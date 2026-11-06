import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error

# eta_data.py lives at the repo root, one level above this file
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NAME = "eta-model"
ALIASES = {"prod": "v1", "candidate": "v2"}
RUSH_HOURS = [7, 8, 9, 16, 17, 18, 19]
EPS = 1e-9  # so a value sitting exactly on a threshold isn't failed by float rounding


def tracking_uri():
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


class EtaPyfunc(mlflow.pyfunc.PythonModel):
    """Serves a joblib bundle; output carries the bundle's version so every answer is traceable."""

    def load_context(self, context):
        b = joblib.load(context.artifacts["bundle"])
        self.model, self.cols, self.version = b["model"], b["feature_columns"], b["version"]

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)[self.cols]
        return pd.DataFrame({"eta_minutes": np.round(self.model.predict(X), 1),
                             "model_version": self.version})


def register(data_dir="data"):
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("eta-serving")
    client = MlflowClient()

    train = pd.read_csv(f"{data_dir}/eta_train.csv")
    reqs = [f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}", f"joblib=={joblib.__version__}",
            f"mlflow=={mlflow.__version__}"]

    registered = {}
    for ver in ("v1", "v2"):
        path = f"{data_dir}/eta_{ver}.joblib"
        b = joblib.load(path)
        X = train[b["feature_columns"]].astype("float64")  # JSON ints and floats both satisfy a double schema
        sig = infer_signature(X.head(5), pd.DataFrame({"eta_minutes": [31.4], "model_version": [ver]}))
        with mlflow.start_run(run_name=f"eta-{ver}"):
            mlflow.log_param("algorithm", type(b["model"]).__name__)
            mlflow.log_param("bundle_version", ver)
            # in-sample, logged for reference only; the gate uses fresh data
            mlflow.log_metric("train_mae", mean_absolute_error(train.eta_minutes, b["model"].predict(X)))
            info = mlflow.pyfunc.log_model(
                name="model", python_model=EtaPyfunc(), artifacts={"bundle": path},
                code_paths=[__file__], signature=sig, input_example=X.head(2),
                pip_requirements=reqs, registered_model_name=NAME)
        registered[ver] = info.registered_model_version

    for alias, ver in ALIASES.items():
        client.set_registered_model_alias(NAME, alias, registered[ver])
    return registry_table(client)


def registry_table(client=None):
    client = client or MlflowClient(tracking_uri())
    aliases = client.get_registered_model(NAME).aliases
    rows = []
    for mv in sorted(client.search_model_versions(f"name='{NAME}'"), key=lambda v: int(v.version)):
        run = client.get_run(mv.run_id)
        rows.append({"version": int(mv.version),
                     "alias": ",".join(a for a, v in aliases.items() if str(v) == str(mv.version)) or "-",
                     "bundle": run.data.params.get("bundle_version"),
                     "algorithm": run.data.params.get("algorithm"),
                     "run_id": mv.run_id[:8]})
    return pd.DataFrame(rows)


def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    """Promotion verdict from two metric dicts (keys mae, mae_rush, mae_rain, p95_ms).

    min_rel_gain and max_slice_regression are fractions of prod's MAE:
    with the defaults the candidate needs >= 3 % lower overall MAE, and each
    slice may be at most 5 % worse than prod's on that slice.
    """
    rel_gain = (prod["mae"] - cand["mae"]) / prod["mae"]
    rush_change = (cand["mae_rush"] - prod["mae_rush"]) / prod["mae_rush"]
    rain_change = (cand["mae_rain"] - prod["mae_rain"]) / prod["mae_rain"]
    criteria = {
        "mae_gain": rel_gain >= min_rel_gain - EPS,
        "p95_latency": cand["p95_ms"] <= max_p95_ms + EPS,
        "rush_slice": rush_change <= max_slice_regression + EPS,
        "rain_slice": rain_change <= max_slice_regression + EPS,
    }
    return {"criteria": criteria,
            "promote": all(criteria.values()),
            "measured": {"rel_mae_gain": round(rel_gain, 4), "cand_p95_ms": cand["p95_ms"],
                         "rush_mae_change": round(rush_change, 4),
                         "rain_mae_change": round(rain_change, 4)},
            "thresholds": {"min_rel_gain": min_rel_gain, "max_p95_ms": max_p95_ms,
                           "max_slice_regression": max_slice_regression}}


def _metrics(model, fresh, features, n_timed=200):
    X = fresh[features].astype("float64")
    pred = model.predict(X)["eta_minutes"].to_numpy()
    err = np.abs(pred - fresh["eta_minutes"].to_numpy())
    rush = fresh["hour"].isin(RUSH_HOURS).to_numpy()
    rain = (fresh["rain"] == 1).to_numpy()

    for i in range(5):  # first calls pay one-off costs; keep them out of the timing
        model.predict(X.iloc[[i]])
    times = []
    for i in range(n_timed):
        row = X.iloc[[i]]
        t0 = time.perf_counter()
        model.predict(row)
        times.append((time.perf_counter() - t0) * 1000)
    return {"mae": round(float(err.mean()), 4), "mae_rush": round(float(err[rush].mean()), 4),
            "mae_rain": round(float(err[rain].mean()), 4),
            "p95_ms": round(float(np.percentile(times, 95)), 3),
            "n_rows": len(fresh), "n_rush": int(rush.sum()), "n_rain": int(rain.sum())}


def promotion_gate(**thresholds):
    import eta_data

    mlflow.set_tracking_uri(tracking_uri())
    fresh = eta_data.make_data(2000, seed=99)
    metrics = {}
    for alias in ("prod", "candidate"):
        model = mlflow.pyfunc.load_model(f"models:/{NAME}@{alias}")
        metrics[alias] = _metrics(model, fresh, eta_data.FEATURES)
    return {**decide(metrics["prod"], metrics["candidate"], **thresholds),
            "prod": metrics["prod"], "candidate": metrics["candidate"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["register", "gate", "table"])
    ap.add_argument("--min-rel-gain", type=float)
    ap.add_argument("--max-p95-ms", type=float)
    ap.add_argument("--max-slice-regression", type=float)
    a = ap.parse_args()
    if a.command == "register":
        print(register().to_string(index=False))
    elif a.command == "table":
        print(registry_table().to_string(index=False))
    else:
        th = {k: v for k, v in [("min_rel_gain", a.min_rel_gain), ("max_p95_ms", a.max_p95_ms),
                                ("max_slice_regression", a.max_slice_regression)] if v is not None}
        print(json.dumps(promotion_gate(**th), indent=2))
