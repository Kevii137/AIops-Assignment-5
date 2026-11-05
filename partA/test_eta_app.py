"""Contract tests for eta_app. Run from the repo root after `python eta_data.py`:

    pytest partA -q
"""
import re
import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import eta_app

client = TestClient(eta_app.app)
URL = "/v1/models/eta:predict"

ROW = {"distance_km": 12.5, "n_stops": 4, "hour": 14, "weekday": 2,
       "traffic_index": 0.3, "rain": 0, "weight_kg": 6.0}

def row(**changes):
    return {**ROW, **changes}

def counter_value(text, path, status):
    pat = rf'eta_requests_total\{{path="{re.escape(path)}",status="{status}"\}} (\S+)'
    m = re.search(pat, text)
    return float(m.group(1)) if m else 0.0

def total_requests(text):
    return sum(float(v) for v in re.findall(r"^eta_requests_total\{[^}]*\} (\S+)$", text, re.M))

def test_valid_request():
    r = client.post(URL, json={"instances": [ROW]})
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "v1"
    assert len(body["predictions"]) == 1
    p = body["predictions"][0]
    assert p == round(p, 1) and p > 0
    assert body["request_id"] == r.headers["X-Request-ID"]

def test_batch_order_preserved():
    rows = [row(distance_km=d) for d in (2.0, 80.0, 30.0, 140.0)]
    got = client.post(URL, json={"instances": rows}).json()["predictions"]
    expected = eta_app.model.predict(pd.DataFrame(rows)[eta_app.FEATURES])
    assert got == [round(float(x), 1) for x in expected]
    # reversing the input must reverse the output, not leave it sorted or shuffled
    back = client.post(URL, json={"instances": rows[::-1]}).json()["predictions"]
    assert back == got[::-1]

@pytest.mark.parametrize("instance, field", [
    (row(hour=24), "instances.0.hour"),                                   # out of range
    ({k: v for k, v in ROW.items() if k != "traffic_index"},
     "instances.0.traffic_index"),                                        # missing field
    (row(distance_km="far"), "instances.0.distance_km"),                 # wrong type
    (row(rain=2), "instances.0.rain"),
    (row(weight_kg=0), "instances.0.weight_kg"),                         # weight must be > 0
])
def test_invalid_input_is_422(instance, field):
    r = client.post(URL, json={"instances": [instance]})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_failed"
    assert field in [d["field"] for d in body["details"]]
    assert all(d["msg"] for d in body["details"])

def test_infeasible_combination_rejected():
    r = client.post(URL, json={"instances": [row(distance_km=150.5, n_stops=21)]})
    assert r.status_code == 422
    assert "infeasible" in r.text
    # each condition alone, and the exact boundary, are fine
    ok = [row(distance_km=150.0, n_stops=21), row(distance_km=300.0, n_stops=20)]
    assert client.post(URL, json={"instances": ok}).status_code == 200

def test_batch_size_limits():
    assert client.post(URL, json={"instances": []}).status_code == 422
    assert client.post(URL, json={"instances": [ROW] * 501}).status_code == 422
    r = client.post(URL, json={"instances": [ROW] * 500})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 500

def test_request_id_echoed():
    r = client.post(URL, json={"instances": [ROW]}, headers={"X-Request-ID": "trace-abc-123"})
    assert r.headers["X-Request-ID"] == "trace-abc-123"
    assert r.json()["request_id"] == "trace-abc-123"

def test_request_id_generated_when_absent():
    r = client.post(URL, json={"instances": [ROW]})
    rid = r.headers["X-Request-ID"]
    uuid.UUID(rid)  # raises if it isn't a UUID
    assert r.json()["request_id"] == rid
    assert client.get("/healthz").headers["X-Request-ID"] != rid

def test_request_id_on_validation_error():
    r = client.post(URL, json={"instances": [row(hour=99)]}, headers={"X-Request-ID": "bad-req-7"})
    assert r.status_code == 422
    assert r.headers["X-Request-ID"] == "bad-req-7" and r.json()["request_id"] == "bad-req-7"
    r = client.post(URL, json={"instances": []})
    uuid.UUID(r.headers["X-Request-ID"])
    assert r.json()["request_id"] == r.headers["X-Request-ID"]

def test_metrics_exposition():
    before = client.get("/metrics").text
    client.post(URL, json={"instances": [ROW]})
    client.post(URL, json={"instances": []})
    client.get("/metrics")
    text = client.get("/metrics").text

    assert counter_value(text, URL, 200) == counter_value(before, URL, 200) + 1
    assert counter_value(text, URL, 422) == counter_value(before, URL, 422) + 1
    # two predict calls and two /metrics scrapes happened: only the predicts may count
    assert total_requests(text) == total_requests(before) + 2
    buckets = re.findall(r'eta_request_latency_seconds_bucket\{le="([^"]+)"\}', text)
    assert len(buckets) >= 3 and "+Inf" in buckets
    assert "eta_request_latency_seconds_sum" in text
    count = float(re.search(r"eta_request_latency_seconds_count (\S+)", text).group(1))
    inf = float(re.search(r'eta_request_latency_seconds_bucket\{le="\+Inf"\} (\S+)', text).group(1))
    assert count == inf > 0

def test_inference_failure_is_generic_500(monkeypatch):
    class Exploding:
        def predict(self, X):
            raise RuntimeError("secret: /srv/models/eta_v1.joblib is corrupt")

    monkeypatch.setattr(eta_app, "model", Exploding())
    r = client.post(URL, json={"instances": [ROW]}, headers={"X-Request-ID": "boom-1"})
    assert r.status_code == 500
    assert "secret" not in r.text and "RuntimeError" not in r.text and "joblib" not in r.text
    assert r.json() == {"error": "internal_error", "request_id": "boom-1"}
    assert r.headers["X-Request-ID"] == "boom-1"

def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "v1"}
