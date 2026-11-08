"""
plot_learning_curve.py
----------------------
Script de visualisation de la courbe d'apprentissage de l'agent PPO à partir du fichier
de logs `consistency_evals.csv`.

Permet de suivre l'évolution :
  - Du reward en train eval vs validation
  - Du drawdown maximal (%)
  - Du score composite et de l'identification visuelle du meilleur checkpoint sélectionné

Exemples d'utilisation :
    python plot_learning_curve.py                     # Trouve automatiquement le dernier run de modèles
    python plot_learning_curve.py --fold 1            # Affiche la courbe d'apprentissage du fold 1
    python plot_learning_curve.py --fold production   # Affiche le fold de production (models/eval_logs)
    python plot_learning_curve.py --file chemin/vers/consistency_evals.csv
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
    """Localise le fichier consistency_evals.csv cible selon le fold ou le chemin explicite."""
    if explicit_file:
        p = Path(explicit_file)
        if not p.exists():
            raise FileNotFoundError(f"Fichier introuvable : {p}")
        return p

    models = root / "models"
    # Fold de production (promu après validation walk-forward)
    if fold == "production":
        p = models / "eval_logs" / "consistency_evals.csv"
        if p.exists():
            return p
    # Fold walk-forward spécifique
    elif fold is not None and str(fold).isdigit():
        p = models / "walk_forward" / f"fold_{fold}" / "eval_logs" / "consistency_evals.csv"
        if p.exists():
            return p
        p_sliding = models / "sliding" / f"fold_{fold}" / "eval_logs" / "consistency_evals.csv"
        if p_sliding.exists():
            return p_sliding

    # Recherche automatique du fichier le plus récemment modifié
    candidates = sorted(
        models.glob("**/eval_logs/consistency_evals.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    # Recherche de secours dans outputs/
    out_candidates = sorted(
        (root / "outputs").glob("**/consistency_evals.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if out_candidates:
        return out_candidates[-1]

    raise FileNotFoundError(
        "Aucun fichier consistency_evals.csv trouvé. Entraînez d'abord un modèle avec train_ppo.py."
    )


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
