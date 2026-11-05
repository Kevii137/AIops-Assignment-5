import logging
import os
import time
import uuid
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram,
                               disable_created_metrics, generate_latest)
from pydantic import BaseModel, Field, model_validator

MODEL_PATH = os.getenv("ETA_MODEL", "data/eta_v1.joblib")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["feature_columns"]
VERSION = bundle["version"]

log = logging.getLogger("eta_app")


class Delivery(BaseModel):
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 together with n_stops > 20")
        return self


class PredictRequest(BaseModel):
    instances: List[Delivery] = Field(min_length=1, max_length=500)


# Own registry rather than the global default
disable_created_metrics()
registry = CollectorRegistry()
REQUESTS = Counter("eta_requests", "HTTP requests by path and status code",
                   ["path", "status"], registry=registry)
# 0.08 is the SLO boundary, so "share of requests within SLO" is one bucket ratio.
LATENCY = Histogram("eta_request_latency_seconds", "Wall time spent handling a request",
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.08, 0.1, 0.25, 0.5, 1.0),
                    registry=registry)
KNOWN_PATHS = {"/v1/models/eta:predict", "/healthz"}

app = FastAPI(title="SwiftParcel ETA service")


@app.middleware("http")
async def request_id_and_metrics(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    path = request.url.path
    if path != "/metrics":
        # 404s from scanners would otherwise mint a new time series per random URL
        label = path if path in KNOWN_PATHS else "other"
        REQUESTS.labels(path=label, status=str(response.status_code)).inc()
        LATENCY.observe(time.perf_counter() - t0)
    return response


@app.exception_handler(RequestValidationError)
async def validation_failed(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body") or "body",
                "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={
        "error": "validation_failed", "details": details,
        "request_id": request.state.request_id})


@app.post("/v1/models/eta:predict")
def predict(body: PredictRequest, request: Request):
    rid = request.state.request_id
    X = pd.DataFrame([d.model_dump() for d in body.instances])[FEATURES]
    try:
        minutes = model.predict(X)
    except Exception:
        log.exception("inference failed, request_id=%s", rid)
        return JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid})
    return {"predictions": [round(float(m), 1) for m in minutes],
            "model_version": VERSION, "request_id": rid}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": VERSION}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
