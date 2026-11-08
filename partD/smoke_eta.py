"""Release gate for an ETA server. Exit 0 if every check passes, 1 otherwise.

    python partD/smoke_eta.py --url http://localhost:8000 --expect-version v1 [--max-p95-ms 80]

--url may be the server root or the full .../v1/models/eta:predict URL. Works against the
Part A service and the Part C Ray app (the extra customer_id field is ignored by Part A).
Check list follows the smoke_test.py described in Lecture 10 (Minikube Serving, slide 27).
"""
import argparse
import math
import sys
import time

import numpy as np
import requests

PREDICT = "/v1/models/eta:predict"
PROBE = {"distance_km": 30.0, "n_stops": 6, "hour": 17, "weekday": 3,
         "traffic_index": 0.5, "rain": 0, "weight_kg": 10.0}


def predict(sess, url, rows):
    r = sess.post(url, json={"customer_id": "smoke-test", "instances": rows}, timeout=10)
    r.raise_for_status()
    return r.json()


def check_health(sess, base, url, args):
    r = sess.get(base + "/healthz", timeout=5)
    ok = r.status_code == 200 and r.json().get("status") == "ok"
    return ok, f"HTTP {r.status_code} {r.text[:80]}"


def check_version(sess, base, url, args):
    got = predict(sess, url, [PROBE])["model_version"]
    return got == args.expect_version, f"answered by {got}, expected {args.expect_version}"


def check_batch(sess, base, url, args):
    rows = [{**PROBE, "hour": h % 24, "distance_km": 2.0 + 7 * h, "rain": h % 2} for h in range(30)]
    fwd = predict(sess, url, rows)["predictions"]
    back = predict(sess, url, rows[::-1])["predictions"]
    plausible = len(fwd) == 30 and all(isinstance(p, (int, float)) and math.isfinite(p) and 0 < p < 1440
                                       for p in fwd)
    return plausible and back == fwd[::-1], f"{len(fwd)} predictions, range {min(fwd)}-{max(fwd)} min, order kept: {back == fwd[::-1]}"


def check_distance(sess, base, url, args):
    km = [5, 20, 50, 100, 150]
    eta = predict(sess, url, [{**PROBE, "distance_km": d} for d in km])["predictions"]
    return all(b >= a for a, b in zip(eta, eta[1:])), f"distance {km} km -> {eta} min"


def check_traffic(sess, base, url, args):
    light, heavy = predict(sess, url, [{**PROBE, "traffic_index": 0.1},
                                       {**PROBE, "traffic_index": 0.9}])["predictions"]
    return heavy >= light, f"traffic 0.1 -> {light} min, 0.9 -> {heavy} min"


def check_rejects_bad_input(sess, base, url, args):
    r = sess.post(url, json={"customer_id": "smoke-test",
                             "instances": [{**PROBE, "distance_km": "far"}]}, timeout=10)
    return r.status_code == 422, f"distance_km='far' -> HTTP {r.status_code}"


def check_latency(sess, base, url, args):
    for _ in range(5):
        predict(sess, url, [PROBE])
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        predict(sess, url, [PROBE])
        times.append((time.perf_counter() - t0) * 1000)
    p95 = float(np.percentile(times, 95))
    args.measured_p95 = p95
    return p95 <= args.max_p95_ms, f"p95 {p95:.1f} ms over 50 single-row requests (limit {args.max_p95_ms} ms)"


CHECKS = [("health", check_health), ("version", check_version), ("batch", check_batch),
          ("sanity: distance", check_distance), ("sanity: traffic", check_traffic),
          ("bad input -> 422", check_rejects_bad_input), ("latency", check_latency)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--max-p95-ms", type=float, default=80.0)
    args = ap.parse_args()
    args.measured_p95 = None

    base = args.url.split(PREDICT)[0].rstrip("/")
    url = base + PREDICT
    sess = requests.Session()
    failed = []
    for name, fn in CHECKS:
        try:
            ok, detail = fn(sess, base, url, args)
        except Exception as e:  # a dead server or a malformed answer is a failed check, not a crash
            ok, detail = False, f"{type(e).__name__}: {str(e)[:100]}"
        print(f"{'PASS' if ok else 'FAIL'}  {name:18s} {detail}")
        if not ok:
            failed.append(name)

    if failed:
        print(f"SMOKE TEST FAILED: {', '.join(failed)}")
        return 1
    print(f"SMOKE TEST PASSED (version={args.expect_version}, p95={args.measured_p95:.1f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
