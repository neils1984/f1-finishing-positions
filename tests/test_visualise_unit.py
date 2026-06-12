"""Unit test for the lap-by-lap prediction-evolution plot."""
import polars as pl

from f1_predictor.visualise import plot_prediction_evolution


def test_plot_prediction_evolution_writes_png(tmp_path):
    rows = []
    for lap in (10, 20, 30, 40):
        for d, fp in [(1, 1), (2, 2), (3, 3)]:
            rows.append({"session_key": 5, "snapshot_lap": lap,
                         "driver_number": d, "final_position": fp,
                         "score": fp * -1.0})
    preds = pl.DataFrame(rows)
    out = tmp_path / "evo.png"
    plot_prediction_evolution(preds, session_key=5, out_path=out)
    assert out.exists() and out.stat().st_size > 0
