"""Shadow comparison: the same 200 single-row requests to prod (:5101) and candidate (:5102).

    export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    mlflow models serve -m "models:/eta-model@prod"      -p 5101 --env-manager local &
    mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local &
    python partB/shadow.py [--alias-demo]

Follows the Task B shadow-test solution in the Lecture 10 MLflow notebook.
"""
import argparse
import time

import numpy as np
import pandas as pd
import requests

FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]


def ask(sess, port, record):
    t0 = time.perf_counter()
    r = sess.post(f"http://localhost:{port}/invocations",
                  json={"dataframe_records": [record]}, timeout=10)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    p = r.json()["predictions"][0]
    return p["eta_minutes"], p["model_version"], ms


def shadow(prod_port, cand_port, data, n):
    fresh = pd.read_csv(data).head(n)
    records = fresh[FEATURES].to_dict(orient="records")
    sess = requests.Session()
    for rec in records[:5]:  # warm both servers; not measured
        ask(sess, prod_port, rec), ask(sess, cand_port, rec)

    out = {prod_port: ([], [], []), cand_port: ([], [], [])}
    for i, rec in enumerate(records):
        # alternate who goes first so neither server always gets the warmer connection
        order = (prod_port, cand_port) if i % 2 == 0 else (cand_port, prod_port)
        for port in order:
            eta, ver, ms = ask(sess, port, rec)
            out[port][0].append(eta)
            out[port][1].append(ver)
            out[port][2].append(ms)

    p, c = np.array(out[prod_port][0]), np.array(out[cand_port][0])
    diff = np.abs(p - c)
    truth = fresh["eta_minutes"].to_numpy()
    print(f"requests sent to each server : {len(records)} (rows 0-{len(records) - 1} of {data})")
    print(f"versions answering           : prod={sorted(set(out[prod_port][1]))} "
          f"candidate={sorted(set(out[cand_port][1]))}")
    print(f"(i)   mean |prod - candidate|: {diff.mean():.2f} min (median {np.median(diff):.2f}, max {diff.max():.1f})")
    print(f"(ii)  differ by > 10 min     : {(diff > 10).mean():.1%} ({int((diff > 10).sum())} of {len(diff)})")
    print(f"(iii) p95 latency            : prod {np.percentile(out[prod_port][2], 95):.1f} ms, "
          f"candidate {np.percentile(out[cand_port][2], 95):.1f} ms "
          f"(p50 {np.median(out[prod_port][2]):.1f} / {np.median(out[cand_port][2]):.1f} ms)")
    # eta_fresh.csv happens to carry the true ETA; live shadow traffic would only get it later
    print(f"MAE vs actual on these rows  : prod {np.abs(p - truth).mean():.2f} min, "
          f"candidate {np.abs(c - truth).mean():.2f} min")
    big = diff > 10
    if big.any():
        closer = (np.abs(c - truth) < np.abs(p - truth))[big].mean()
        print(f"of the >10 min disagreements, candidate was closer to actual in {closer:.0%}")
    return records[0]


def alias_demo(prod_port, record):
    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient(mlflow.get_tracking_uri())
    sess = requests.Session()
    before = client.get_model_version_by_alias("eta-model", "prod").version
    cand = client.get_model_version_by_alias("eta-model", "candidate").version
    print(f"\nalias demo: prod -> version {before}; server :{prod_port} answers {ask(sess, prod_port, record)[1]}")
    try:
        client.set_registered_model_alias("eta-model", "prod", cand)
        now = client.get_model_version_by_alias("eta-model", "prod").version
        print(f"moved prod -> version {now}; server :{prod_port} still answers {ask(sess, prod_port, record)[1]}")
    finally:
        client.set_registered_model_alias("eta-model", "prod", before)
        print(f"restored prod -> version {before}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-port", type=int, default=5101)
    ap.add_argument("--cand-port", type=int, default=5102)
    ap.add_argument("--data", default="data/eta_fresh.csv")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--alias-demo", action="store_true")
    a = ap.parse_args()
    first = shadow(a.prod_port, a.cand_port, a.data, a.n)
    if a.alias_demo:
        alias_demo(a.prod_port, first)
