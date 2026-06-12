import polars as pl
from f1_predictor.backtest import walk_forward


def _toy_race(session_key, date, n=6, shift=0):
    # n drivers; final order = current order rotated by `shift` (learnable signal).
    rows = []
    for d in range(1, n + 1):
        final = ((d - 1 + shift) % n) + 1
        rows.append({
            "session_key": session_key, "date_start": date, "lap_number": 30,
            "driver_number": d, "position": d, "final_position": final,
            "feat_a": float(d), "feat_b": float(final),
        })
    return pl.DataFrame(rows)


def test_walk_forward_predicts_each_target_on_past_only():
    # 4 races; targets are the last 2. Each fold trains on strictly earlier races.
    races = {
        9001: _toy_race(9001, "2025-01-01", shift=0),
        9002: _toy_race(9002, "2025-02-01", shift=0),
        9003: _toy_race(9003, "2026-03-01", shift=0),
        9004: _toy_race(9004, "2026-04-01", shift=0),
    }
    result = walk_forward(
        races,
        target_keys=[9003, 9004],
        feature_columns=["feat_a", "feat_b"],
        snapshot_laps=[30],
        params={"num_leaves": 7, "min_data_in_leaf": 1, "n_estimators": 20},
    )
    # One result row per target race, each with a model and naive Spearman.
    assert set(result["session_key"].to_list()) == {9003, 9004}
    assert "model_spearman" in result.columns and "naive_spearman" in result.columns
    # Signal is pure persistence (shift=0) -> both near 1.0.
    assert result["naive_spearman"].min() > 0.9


def test_walk_forward_accepts_weighting_and_blend():
    from f1_predictor.models.baseline_gbm import season_weights
    races = {
        9001: _toy_race(9001, "2025-01-01", shift=0),
        9002: _toy_race(9002, "2026-03-01", shift=0),
    }
    res = walk_forward(
        races, target_keys=[9002], feature_columns=["feat_a", "feat_b"],
        snapshot_laps=[30], params={"num_leaves": 7, "min_data_in_leaf": 1, "n_estimators": 10},
        sample_weight_fn=lambda tr: season_weights(tr, upweight_2026=4.0),
        blend_alpha=0.5,
    )
    assert res.height == 1 and res["session_key"][0] == 9002
