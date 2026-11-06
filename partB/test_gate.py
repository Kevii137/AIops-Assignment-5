"""Tests for decide() with made-up metrics. No MLflow server or registry needed:

    pytest partB -q
"""
import copy

import pytest

from eta_mlflow import decide

PROD = {"mae": 10.0, "mae_rush": 12.0, "mae_rain": 11.0, "p95_ms": 3.0}


def cand(**changes):
    # default candidate: 20 % better overall, slightly better on both slices, fast
    return {**{"mae": 8.0, "mae_rush": 11.5, "mae_rain": 10.5, "p95_ms": 4.0}, **changes}


def test_clear_improvement_is_promoted():
    out = decide(PROD, cand())
    assert out["promote"] is True
    assert out["criteria"] == {"mae_gain": True, "p95_latency": True,
                               "rush_slice": True, "rain_slice": True}


@pytest.mark.parametrize("cand_mae, ok", [
    (9.69, True),    # 3.1 % better
    (9.70, True),    # exactly 3 %: "at least" includes the boundary
    (9.71, False),   # 2.9 %
    (10.5, False),   # worse than prod
])
def test_overall_gain_boundary(cand_mae, ok):
    out = decide(PROD, cand(mae=cand_mae))
    assert out["criteria"]["mae_gain"] is ok
    assert out["promote"] is ok


@pytest.mark.parametrize("p95, ok", [(9.99, True), (10.0, True), (10.01, False)])
def test_latency_boundary(p95, ok):
    out = decide(PROD, cand(p95_ms=p95))
    assert out["criteria"]["p95_latency"] is ok
    assert out["promote"] is ok


@pytest.mark.parametrize("slice_key", ["mae_rush", "mae_rain"])
@pytest.mark.parametrize("worse_by, ok", [(0.049, True), (0.05, True), (0.051, False)])
def test_slice_regression_boundary(slice_key, worse_by, ok):
    c = cand(**{slice_key: PROD[slice_key] * (1 + worse_by)})
    out = decide(PROD, c)
    name = "rush_slice" if slice_key == "mae_rush" else "rain_slice"
    assert out["criteria"][name] is ok
    assert out["promote"] is ok


def test_big_overall_gain_does_not_hide_a_slice_regression():
    # 40 % better on average, but 20 % worse in the rain: must not ship
    out = decide(PROD, cand(mae=6.0, mae_rain=13.2))
    assert out["criteria"]["mae_gain"] is True
    assert out["criteria"]["rain_slice"] is False
    assert out["promote"] is False


def test_every_failing_criterion_is_reported():
    out = decide(PROD, cand(mae=9.9, p95_ms=25.0, mae_rush=15.0, mae_rain=14.0))
    assert out["criteria"] == {"mae_gain": False, "p95_latency": False,
                               "rush_slice": False, "rain_slice": False}
    assert out["promote"] is False


def test_thresholds_are_respected():
    c = cand()  # 20 % better overall
    assert decide(PROD, c, min_rel_gain=0.60)["promote"] is False
    assert decide(PROD, c, min_rel_gain=0.20)["promote"] is True
    # the gain is 2 min on prod's 10 min (20 %), not on the candidate's 8 min (25 %)
    assert decide(PROD, c, min_rel_gain=0.22)["promote"] is False
    assert decide(PROD, c, max_p95_ms=3.9)["criteria"]["p95_latency"] is False
    worse_rush = cand(mae_rush=13.0)  # 8.3 % worse on rush hour
    assert decide(PROD, worse_rush)["promote"] is False
    assert decide(PROD, worse_rush, max_slice_regression=0.10)["promote"] is True


def test_decide_is_pure():
    p, c = copy.deepcopy(PROD), cand()
    snapshot = copy.deepcopy((p, c))
    first = decide(p, c)
    assert (p, c) == snapshot  # inputs untouched
    assert decide(p, c) == first  # same inputs, same answer
