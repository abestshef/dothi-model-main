"""Smoke tests for the simulation core."""

import numpy as np

from dothi_nursery.model import LENGTH, WIDTH, SimParams, run_simulation


def _params(**kw):
    base = dict(
        beta_choice="Moderate", betap_choice="Moderate", gamma_choice="4 months",
        control_month=16, radius_m=2.0,
    )
    base.update(kw)
    return SimParams.from_choices(**base)


def test_run_shapes_and_labels():
    result = run_simulation(_params(), rng=np.random.default_rng(0))

    assert result["control"] == 16
    assert len(result["series"]) == 4
    assert len(result["labels"]) == 4
    assert result["gridsaves"].shape == (12, LENGTH, WIDTH)

    for s in result["series"]:
        # time, symptomatic and yield-loss series line up
        assert len(s["t"]) == len(s["symptomatic"]) == len(s["yield_loss"])
        # symptomatic counts are never negative
        assert min(s["symptomatic"]) >= 0
        # time runs from planting to (at most) the 30-month horizon
        assert s["t"][0] == 0.0
        assert max(s["t"]) <= 30.0001


def test_reproducible_with_seed():
    a = run_simulation(_params(), rng=np.random.default_rng(42))
    b = run_simulation(_params(), rng=np.random.default_rng(42))
    assert np.array_equal(a["gridsaves"], b["gridsaves"])


def test_grid_only_holds_expected_states():
    result = run_simulation(_params(control_month=12), rng=np.random.default_rng(1))
    allowed = {0, 1, 2, 4}
    assert set(np.unique(result["gridsaves"])).issubset(allowed)


def test_larger_radius_removes_at_least_as_much():
    # Radius controls (rows 3 & 4) should never leave more symptomatic plants
    # standing immediately after control than the infected-only controls.
    result = run_simulation(_params(radius_m=5.0), rng=np.random.default_rng(7))
    gs = result["gridsaves"]
    just_after_inf_only = int((gs[1] == 2).sum())  # control 1, after
    just_after_radius = int((gs[7] == 2).sum())    # control 3, after
    assert just_after_radius <= just_after_inf_only
