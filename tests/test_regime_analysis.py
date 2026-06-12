import polars as pl
from f1_predictor.regime_analysis import position_change_stats


def test_position_change_stats_reports_dnf_and_dispersion():
    # Two races. Race 1: clean, drivers finish near their lap-40 order (small
    # change). Race 2: one DNF (final_position 20 from a front-running lap-40 P2)
    # and a big reshuffle (large change).
    feats = pl.DataFrame({
        "session_key": [1, 1, 2, 2],
        "lap_number":  [40, 40, 40, 40],
        "driver_number": [1, 2, 3, 4],
        "position":      [1, 2, 2, 5],     # race-state position at lap 40
        "final_position":[1, 3, 20, 4],    # driver 3 retired -> classified last
        "dnf": [False, False, True, False],
    })
    stats = position_change_stats(feats, snapshot_lap=40)
    # mean absolute (final_position - lap40 position), per race then averaged
    # race1: |1-1|=0, |3-2|=1 -> 0.5 ; race2: |20-2|=18, |4-5|=1 -> 9.5
    assert stats["mean_abs_change"] == pl.Series([0.5, 9.5]).mean()
    assert stats["dnf_rate"] == 0.25   # 1 of 4 driver-rows is a DNF
