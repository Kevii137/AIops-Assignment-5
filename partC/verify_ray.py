"""Deploy eta_ray on a 2-CPU Ray and check the sticky canary. Run from the repo root:

    python partC/verify_ray.py [--autoscale] | tee partC/verify_output.txt

Every check prints its evidence and then asserts; the script stops at the first failure.
"""
import argparse
import collections
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import ray
import requests
from ray import serve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eta_ray  # noqa: E402

APP = "eta"
BASE = "http://127.0.0.1:8100"
URL = f"{BASE}/v1/models/eta:predict"
IDS = [f"cust-{i:04d}" for i in range(1000)]


def deploy(weight, extra_work_ms=0.0):
    serve.run(eta_ray.build(weight=weight, extra_work_ms=extra_work_ms), name=APP, route_prefix="/")
    for _ in range(60):
        try:
            if requests.get(f"{BASE}/healthz", timeout=2).json().get("weight") == weight:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"app with weight {weight} never became ready")


def ask_all(sess, ids, rows, headers=None):
    """customer_id -> arm; also checks the answering model matches the arm."""
    arms = {}
    for i, cid in enumerate(ids):
        r = sess.post(URL, json={"customer_id": cid, "instances": [rows[i % len(rows)]]},
                      headers=headers, timeout=10)
        assert r.status_code == 200, f"{cid}: HTTP {r.status_code} {r.text[:200]}"
        j = r.json()
        assert j["model_version"] == j["arm"], f"{cid}: arm {j['arm']} answered by {j['model_version']}"
        arms[cid] = j["arm"]
    return arms


def share_v2(arms):
    return sum(a == "v2" for a in arms.values()) / len(arms)


def replica_counts():
    deps = serve.status().applications[APP].deployments
    return {name: dict(d.replica_states) for name, d in deps.items()}


def show_setup():
    print("deployments and replica states:")
    for name, states in replica_counts().items():
        print(f"  {name:12s} {states}")
    total = ray.cluster_resources()["CPU"]
    booked = total - ray.available_resources().get("CPU", 0.0)
    print(f"Ray CPUs: {total:.0f}, booked by Serve now: {booked:.2f} "
          f"(worst case with 3 candidate replicas: 2*0.25 + 3*0.25 + 0.1 = 1.35)")
    assert replica_counts()["EtaModelV1"].get("RUNNING") == 2, "prod must run 2 replicas"
    assert booked <= total


def sample_table(ids, *phases):
    df = pd.DataFrame({"customer_id": ids, "bucket": [round(eta_ray.bucket(c), 4) for c in ids]})
    for label, arms in phases:
        df[label] = [arms[c] for c in ids]
    return df.to_string(index=False)


def main(autoscale):
    rows = pd.read_csv("data/eta_sample.csv").to_dict(orient="records")
    sess = requests.Session()

    print("== deploy build(weight=0.2) on ray.init(num_cpus=2)")
    deploy(0.2)
    show_setup()

    print("\n== (a) share of 1000 distinct customers routed to the candidate")
    first = ask_all(sess, IDS, rows)
    s = share_v2(first)
    print(f"v2 share = {s:.3f} ({sum(a == 'v2' for a in first.values())} of 1000), target 0.20 +/- 0.04")
    assert abs(s - 0.2) <= 0.04
    print("PASS (a)")

    print("\n== (b) the first 200 customers ask again")
    again = ask_all(sess, IDS[:200], rows)
    changed = [c for c in IDS[:200] if again[c] != first[c]]
    print(f"customers whose arm changed: {len(changed)} of 200")
    assert not changed
    print("PASS (b)")

    print("\n== (c) delete the app, redeploy the identical app, ask all 1000 again")
    serve.delete(APP)
    time.sleep(3)  # lecture workaround: give the controller time before redeploying
    deploy(0.2)
    show_setup()
    redeployed = ask_all(sess, IDS, rows)
    changed = [c for c in IDS if redeployed[c] != first[c]]
    print(f"customers whose arm changed after redeploy: {len(changed)} of 1000 (incl. the 200 from (b))")
    assert not changed
    print("PASS (c)")

    print("\n== (d) raise weight 0.2 -> 0.5 (router redeployed in place)")
    deploy(0.5)
    raised = ask_all(sess, IDS, rows)
    moved_back = [c for c in IDS if first[c] == "v2" and raised[c] == "v1"]
    moved_up = [c for c in IDS if first[c] == "v1" and raised[c] == "v2"]
    print(f"v2 share now {share_v2(raised):.3f}; moved v1->v2: {len(moved_up)}; moved v2->v1: {len(moved_back)}")
    print("sample:\n" + sample_table(IDS[:10], ("w=0.2", first), ("redeploy", redeployed), ("w=0.5", raised)))
    assert not moved_back
    print("PASS (d)")

    print("\n== (e) X-Beta: 1 on 50 requests from customers who are on v1 at weight 0.5")
    v1_customers = [c for c in IDS if raised[c] == "v1"][:50]
    forced = ask_all(sess, v1_customers, rows, headers={"X-Beta": "1"})
    print(f"with header: {collections.Counter(forced.values())}")
    assert all(a == "v2" for a in forced.values())
    without = ask_all(sess, v1_customers, rows)
    print(f"same customers without header: {collections.Counter(without.values())}")
    assert all(a == "v1" for a in without.values())
    print("PASS (e)")

    if autoscale:
        autoscale_demo(rows)

    print("\nALL CHECKS PASSED")


def autoscale_demo(rows, clients=12, load_s=25, idle_s=45):
    print(f"\n== autoscaling: weight=1.0, extra_work_ms=50, {clients} clients for {load_s} s, then idle {idle_s} s")
    serve.delete(APP)
    time.sleep(3)
    deploy(1.0, extra_work_ms=50)
    timeline, stop = [], threading.Event()
    t0 = time.time()

    def poll():
        while not stop.is_set():
            n = sum(replica_counts()["EtaModelV2"].values())
            timeline.append((round(time.time() - t0), n))
            time.sleep(1)

    def client(k):
        s = requests.Session()
        while time.time() - t0 < load_s:
            s.post(URL, json={"customer_id": f"load-{k}", "instances": [rows[k]]}, timeout=10)

    th = threading.Thread(target=poll)
    th.start()
    workers = [threading.Thread(target=client, args=(k,)) for k in range(clients)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    time.sleep(idle_s)
    stop.set()
    th.join()
    changes = [timeline[0]] + [p for prev, p in zip(timeline, timeline[1:]) if p[1] != prev[1]]
    print("candidate replicas over time (t_s, replicas), changes only:", changes)
    peak = max(n for _, n in timeline)
    print(f"peak {peak}, final {timeline[-1][1]}; load stopped at t={load_s}s")
    assert peak > 1 and peak <= 3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--autoscale", action="store_true")
    a = ap.parse_args()
    ray.init(num_cpus=2, include_dashboard=False, logging_level="ERROR", log_to_driver=False,
             runtime_env={"env_vars": {"PYTHONPATH": str(HERE)}})
    serve.start(http_options={"host": "127.0.0.1", "port": 8100}, logging_config={"log_level": "WARNING"})
    try:
        main(a.autoscale)
    finally:
        serve.shutdown()
        ray.shutdown()
