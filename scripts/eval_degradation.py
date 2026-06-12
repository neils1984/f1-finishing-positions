"""CLI: Stage-0 regime-drift diagnostic for the 2026 regulations.

Trains the cross-era LightGBM baseline on the train split and contrasts its
ranking quality on val (same-regime reference) vs test (the 2026 drift probe),
broken down per snapshot lap, against the naive persistence yardstick.

    uv run python scripts/eval_degradation.py
    uv run python scripts/eval_degradation.py --snapshots-dir data/snapshots --out runs

Reads the snapshot splits built by scripts/build_snapshots.py; needs 2026 races
present in the test split to say anything about 2026. Writes a JSON report under
runs/degradation-<ts>/ unless --no-save is given.
"""
from pathlib import Path

import typer

from f1_predictor.drift import degradation_report, write_report

app = typer.Typer(add_completion=False)


def _fmt(m: dict) -> str:
    return (f"spearman={m['spearman']:.4f}  top1={m['top1_accuracy']:.3f}  "
            f"top3={m['top3_accuracy']:.3f}  mpe={m['mean_position_error']:.3f}  "
            f"n_groups={m['n_groups']}")


def _print_split(name: str, s: dict) -> None:
    typer.echo(f"\n=== {name.upper()}  ({s['n_races']} races, {s['n_rows']} rows) ===")
    typer.echo(f"  naive : {_fmt(s['naive'])}")
    typer.echo(f"  model : {_fmt(s['model'])}")
    typer.echo(f"  uplift (model - naive spearman): {s['uplift']:+.4f}")
    typer.echo("  per lap:  lap   naive   model   uplift   n")
    for r in s["per_lap"]:
        typer.echo(f"           {r['snapshot_lap']:>4}  {r['naive_spearman']:+.3f}  "
                   f"{r['model_spearman']:+.3f}  {r['uplift']:+.3f}  {r['n_groups']:>3}")


@app.command()
def main(
    snapshots_dir: Path = typer.Option(Path("data/snapshots"), "--snapshots-dir"),
    out: Path = typer.Option(Path("runs"), "--out", help="Where to write the report"),
    no_save: bool = typer.Option(False, "--no-save", help="Print only, don't write JSON"),
) -> None:
    report = degradation_report(snapshots_dir)

    for split in ("val", "test"):
        if split in report["splits"]:
            _print_split(split, report["splits"][split])

    vt = report.get("val_to_test")
    if vt is not None:
        typer.echo("\n=== VAL -> TEST DRIFT (same-regime -> 2026) ===")
        typer.echo(f"  naive spearman drop : {vt['naive_spearman_drop']:+.4f}")
        typer.echo(f"  model spearman drop : {vt['model_spearman_drop']:+.4f}")
        typer.echo(f"  uplift  val={vt['uplift_val']:+.4f}  test={vt['uplift_test']:+.4f}  "
                   f"retained={vt['uplift_retained']:+.4f}")

    # Regime-awareness sanity check: is the era flag actually usable by the model?
    typer.echo("\n=== REGIME-AWARENESS ===")
    if report["inert_train_features"]:
        typer.echo(f"  inert (constant in train, zero signal): {report['inert_train_features']}")
    else:
        typer.echo("  no inert train features")
    if report["is_2026_regs_inert"] is not None:
        verdict = ("INERT — no 2026 race in train, model cannot split on it"
                   if report["is_2026_regs_inert"] else "active")
        typer.echo(f"  is_2026_regs: {verdict}  "
                   f"(gain importance={report['is_2026_regs_gain_importance']:.1f})")

    if not no_save:
        run_dir = write_report(report, out)
        typer.echo(f"\nWrote {run_dir / 'degradation.json'}")


if __name__ == "__main__":
    app()
