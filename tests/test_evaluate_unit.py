import polars as pl
import pytest
from f1_predictor.evaluate import ranking_metrics


def _preds(rows):
    # rows: (session_key, snapshot_lap, driver, final_position, score)
    return pl.DataFrame(
        rows,
        schema=["session_key", "snapshot_lap", "driver_number", "final_position", "score"],
        orient="row",
    )


def test_perfect_ranking_scores_one():
    # Higher score should mean better (lower) final_position. Perfect alignment.
    df = _preds([
        (1, 30, 44, 1, 0.9),
        (1, 30, 11, 2, 0.5),
        (1, 30, 16, 3, 0.1),
    ])
    m = ranking_metrics(df)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["top1_accuracy"] == pytest.approx(1.0)
    assert m["top3_accuracy"] == pytest.approx(1.0)
    assert m["mean_position_error"] == pytest.approx(0.0)


def test_reversed_ranking_is_negative_spearman():
    df = _preds([
        (1, 30, 44, 1, 0.1),
        (1, 30, 11, 2, 0.5),
        (1, 30, 16, 3, 0.9),
    ])
    m = ranking_metrics(df)
    assert m["spearman"] == pytest.approx(-1.0)
    assert m["top1_accuracy"] == pytest.approx(0.0)


def test_metrics_average_over_groups():
    # Two groups: one perfect, one reversed -> mean spearman 0.
    df = _preds([
        (1, 30, 1, 1, 0.9), (1, 30, 2, 2, 0.1),
        (2, 30, 3, 1, 0.1), (2, 30, 4, 2, 0.9),
    ])
    m = ranking_metrics(df)
    assert m["spearman"] == pytest.approx(0.0)
