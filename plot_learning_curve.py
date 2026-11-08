"""
plot_learning_curve.py
----------------------
Plot the training learning curve from consistency_evals.csv (rewards and drawdowns
over training timesteps), identifying the best eligible checkpoint.

Usage:
    python plot_learning_curve.py                     # auto-find latest consistency_evals.csv
    python plot_learning_curve.py --fold 1            # inspect fold 1
    python plot_learning_curve.py --fold production   # inspect production fold (models/eval_logs)
    python plot_learning_curve.py --file path/to/consistency_evals.csv
    python plot_learning_curve.py --output outputs/learning_curve.html --no-browser
"""
from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import pandas as pd

from visualize import plot_learning_curve, save_html


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "models").is_dir():
            return candidate
    return Path.cwd()


def resolve_evals_path(root: Path, fold: int | str | None = None, explicit_file: str | None = None) -> Path:
    if explicit_file:
        p = Path(explicit_file)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return p

    models = root / "models"
    if fold == "production":
        p = models / "eval_logs" / "consistency_evals.csv"
        if p.exists():
            return p
    elif fold is not None and str(fold).isdigit():
        p = models / "walk_forward" / f"fold_{fold}" / "eval_logs" / "consistency_evals.csv"
        if p.exists():
            return p
        p_sliding = models / "sliding" / f"fold_{fold}" / "eval_logs" / "consistency_evals.csv"
        if p_sliding.exists():
            return p_sliding

    # Auto search newest
    candidates = sorted(
        models.glob("**/eval_logs/consistency_evals.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    # Also check outputs directory
    out_candidates = sorted(
        (root / "outputs").glob("**/consistency_evals.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if out_candidates:
        return out_candidates[-1]

    raise FileNotFoundError("No consistency_evals.csv found. Train a model first via train_ppo.py.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PPO learning curve from training checkpoints."
    )
    parser.add_argument(
        "--fold", default=None,
        help="Fold number (1, 2, ...) or 'production'",
    )
    parser.add_argument(
        "--file", default=None,
        help="Explicit path to consistency_evals.csv",
    )
    parser.add_argument(
        "--output", "-o", default="outputs/learning_curve.html",
        help="Output HTML path (default: outputs/learning_curve.html)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not automatically open the browser",
    )

    args = parser.parse_args()

    root = find_project_root()
    evals_path = resolve_evals_path(root, fold=args.fold, explicit_file=args.file)
    print(f"Loading learning curve from: {evals_path.resolve()}")

    df = pd.read_csv(evals_path).sort_values("timesteps").reset_index(drop=True)
    print(f"Loaded {len(df)} evaluation points.")

    title = f"PPO Learning Curve — {evals_path.parent.parent.name}"
    fig = plot_learning_curve(df, title=title)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_html(fig, out_path)
    print(f"Saved interactive learning curve -> {out_path.resolve()}")

    if not args.no_browser:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
