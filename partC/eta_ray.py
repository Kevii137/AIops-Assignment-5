"""Ray Serve app: a sticky canary between the v1 (prod) and v2 (candidate) ETA models.

    Router (ingress) --> EtaModelV1: 2 replicas, 0.25 CPU each
                     +-> EtaModelV2: autoscaled 1..3 replicas, 0.25 CPU each

POST /v1/models/eta:predict  {"customer_id": "...", "instances": [{7 features}, ...]}

Deployment/router/build structure follows ray_serve_app.py from Lecture 10 (Ray Serve).
The lecture's router picks an arm per request with random(); this one picks per customer.
"""
import hashlib
import os
import time
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from ray import serve
from ray.serve.handle import DeploymentHandle

FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]

# Worst case 2*0.25 + 3*0.25 + 0.1 = 1.35 CPUs, so the app fits a 2-CPU Ray.
MODEL_CPUS = 0.25
ROUTER_CPUS = 0.1
CANDIDATE_AUTOSCALING = {
    "min_replicas": 1,
    "max_replicas": 3,
    "target_ongoing_requests": 2,
    "upscale_delay_s": 3,
    "downscale_delay_s": 30,
}


def bucket(customer_id: str) -> float:
    """Map an id to [0, 1). Same id -> same value in every process and every run."""
    digest = hashlib.sha256(customer_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def choose_arm(customer_id: str, weight: float, beta_header: str | None = None) -> str:
    if beta_header == "1":
        return "v2"
    return "v2" if bucket(customer_id) < weight else "v1"


@serve.deployment(ray_actor_options={"num_cpus": MODEL_CPUS}, max_ongoing_requests=16,
                  health_check_period_s=5, logging_config={"log_level": "WARNING"})
class EtaModel:
    def __init__(self, path: str, extra_work_ms: float = 0.0):
        b = joblib.load(path)
        self.model, self.cols, self.version = b["model"], b["feature_columns"], b["version"]
        # extra_work_ms stands in for a heavier model. It blocks on purpose (as in the
        # lecture) so one replica really is busy and autoscaling has something to react to.
        self.extra_s = extra_work_ms / 1000.0
        self.replica = serve.get_replica_context().replica_id.unique_id

    async def __call__(self, rows: List[dict]):
        minutes = self.model.predict(pd.DataFrame(rows)[self.cols])
        if self.extra_s:
            time.sleep(self.extra_s)
        return {"predictions": [round(float(m), 1) for m in minutes],
                "model_version": self.version, "served_by": self.replica}

    async def check_health(self):
        if self.model is None:
            raise RuntimeError("model not loaded")


class Delivery(BaseModel):
    distance_km: float
    n_stops: int
    hour: int
    weekday: int
    traffic_index: float
    rain: int
    weight_kg: float


class CanaryRequest(BaseModel):
    customer_id: str = Field(min_length=1)
    instances: List[Delivery] = Field(min_length=1, max_length=500)


api = FastAPI(title="SwiftParcel ETA canary")


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": ROUTER_CPUS},
                  logging_config={"log_level": "WARNING"})
@serve.ingress(api)
class Router:
    def __init__(self, prod: DeploymentHandle, candidate: DeploymentHandle, weight: float):
        self.handles = {"v1": prod, "v2": candidate}
        self.weight = weight

    @api.get("/healthz")
    def healthz(self):
        return {"status": "ok", "weight": self.weight}

    @api.post("/v1/models/eta:predict")
    async def predict(self, body: CanaryRequest, request: Request):
        arm = choose_arm(body.customer_id, self.weight, request.headers.get("x-beta"))
        rows = [d.model_dump() for d in body.instances]
        try:
            out = await self.handles[arm].remote(rows)
        except Exception:
            raise HTTPException(status_code=500, detail="inference failed")
        return {**out, "arm": arm}


def build(weight: float = 0.1, extra_work_ms: float = 0.0, data_dir: str = "data"):
    """Application graph Router -> (v1, v2). `weight` is the share of customers sent to v2."""
    # absolute paths: a replica's working directory is not guaranteed to be ours
    v1_path = os.path.abspath(os.path.join(data_dir, "eta_v1.joblib"))
    v2_path = os.path.abspath(os.path.join(data_dir, "eta_v2.joblib"))
    prod = EtaModel.options(name="EtaModelV1", num_replicas=2).bind(v1_path, extra_work_ms)
    candidate = EtaModel.options(name="EtaModelV2", num_replicas=None,
                                 autoscaling_config=CANDIDATE_AUTOSCALING).bind(v2_path, extra_work_ms)
    return Router.bind(prod, candidate, weight)
