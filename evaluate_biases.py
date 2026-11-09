"""
evaluate_biases.py
------------------
Script d'évaluation systématique et hors-échantillon des biais directionnels :
  - Test de chaque sous-biais individuellement (A1, A2, B1, B2, C1..C4, D1..D4)
  - Test des biais acheteurs et vendeurs séparément
  - Test des combinaisons et confluences de biais
  - Comparaison rigoureuse avec et sans filtre directionnel (avec frais, spread et slippage)
  - Identification des conditions où le trading doit être restreint ou interdit (filtre de no-trade)

Usage:
    python evaluate_biases.py --csv data/okx_BTCUSDT_7d.csv --symbol BTC/USDT --is-crypto
    python evaluate_biases.py --decision-tf 15min --out-dir outputs/bias_eval
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import CFG
from data_loader import load_mt_ohlcv_csv, resample_ohlcv, split_train_val_test
from directional_biases import DirectionalBiasFilter, compute_directional_biases
from env_bracket import BracketTradingEnv
from evaluate import full_report
from features import prepare_feature_frame


def _make_filtered_policy(base_policy, bias_filter: DirectionalBiasFilter, feature_cols: list[str]):
    """Encapsule une politique de trading avec un filtre de biais directionnel."""
    def policy(state, obs_dict=None):
        action = base_policy(state)
        allowed = bias_filter.get_allowed_direction(state, feature_cols=feature_cols)
        if allowed is not None:
            if allowed == 0:
                return (0, action[1], action[2])
            if allowed == 1 and action[0] == 2:
                return (0, action[1], action[2])
            if allowed == -1 and action[0] == 1:
                return (0, action[1], action[2])
        return action
    return policy


def evaluate_filter_variant(
    filter_name: str,
    bias_filter: DirectionalBiasFilter,
    val_feat: pd.DataFrame,
    val_m1: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Évalue une variante de filtre sur le split de validation hors-échantillon."""
    from baselines import ema_atr_trend_policy, evaluate_policy

    env = BracketTradingEnv(
        val_feat,
        val_m1,
        feature_cols,
        sl_atr_multipliers=CFG.sl_atr_multipliers,
        tp_r_multipliers=CFG.tp_r_multipliers,
        initial_equity=CFG.initial_equity,
        risk_fraction=CFG.risk_fraction,
        spread_price=CFG.spread_price,
        slippage_price=CFG.slippage_price,
        commission_per_trade=CFG.commission_per_trade,
    )

    filtered_pol = _make_filtered_policy(ema_atr_trend_policy, bias_filter, feature_cols)
    eq, trades, rep = evaluate_policy(env, filtered_pol, initial_equity=CFG.initial_equity)
    res = rep["value"].to_dict()
    res["filter_variant"] = filter_name
    return res


def run_bias_benchmark(decision_df: pd.DataFrame, m1_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Exécute l'analyse comparative de toutes les variantes de biais directionnels."""
    train_feat, val_feat, test_feat = split_train_val_test(
        decision_df,
        train_frac=CFG.train_frac,
        val_frac=CFG.val_frac,
        embargo_bars=CFG.split_embargo_bars,
    )

    val_m1 = m1_df.loc[(m1_df.index > val_feat.index.min()) & (m1_df.index <= val_feat.index.max())]

    variants = {
        # 1. Référence sans filtre
        "00_No_Filter": DirectionalBiasFilter(mode="none"),

        # 2. Biais A individuels
        "A1_Daily_Bullish_Only": DirectionalBiasFilter(use_a1=True, use_a2=False, use_b1=False, use_b2=False, use_c1=False, use_c2=False, use_c3=False, use_c4=False, use_d1=False, use_d2=False, use_d3=False, use_d4=False),
        "A2_Daily_Bearish_Only": DirectionalBiasFilter(use_a1=False, use_a2=True, use_b1=False, use_b2=False, use_c1=False, use_c2=False, use_c3=False, use_c4=False, use_d1=False, use_d2=False, use_d3=False, use_d4=False),

        # 3. Biais B individuels (6H RTH)
        "B1_RTH_6H_Bullish_Only": DirectionalBiasFilter(use_a1=False, use_a2=False, use_b1=True, use_b2=False, use_c1=False, use_c2=False, use_c3=False, use_c4=False, use_d1=False, use_d2=False, use_d3=False, use_d4=False),
        "B2_RTH_6H_Bearish_Only": DirectionalBiasFilter(use_a1=False, use_a2=False, use_b1=False, use_b2=True, use_c1=False, use_c2=False, use_c3=False, use_c4=False, use_d1=False, use_d2=False, use_d3=False, use_d4=False),

        # 4. Biais C (Sweeps 07:00-09:30)
        "C1_Sweep_PDH_Sell": DirectionalBiasFilter(use_a1=False, use_a2=False, use_b1=False, use_b2=False, use_c1=True, use_c2=False, use_c3=False, use_c4=False, use_d1=False, use_d2=False, use_d3=False, use_d4=False),
        "C2_Sweep_PDL_Buy": DirectionalBiasFilter(use_a1=False, use_a2=False, use_b1=False, use_b2=False, use_c1=False, use_c2=True, use_c3=False, use_c4=False, use_d1=False, use_d2=False, use_d3=False, use_d4=False),
        "C3_Sweep_PrevHigh_Sell": DirectionalBiasFilter(use_a1=False, use_a2=False, use_b1=False, use_b2=False, use_c1=False, use_c2=False, use_c3=True, use_c4=False, use_d1=False, use_d2=False, use_d3=False, use_d4=False),
        "C4_Sweep_PrevLow_Buy": DirectionalBiasFilter(use_a1=False, use_a2=False, use_b1=False, use_b2=False, use_c1=False, use_c2=False, use_c3=False, use_c4=True, use_d1=False, use_d2=False, use_d3=False, use_d4=False),

        # 5. Biais D (Sweeps 09:30-10:00)
        "D1_Sweep_PDH_Sell": DirectionalBiasFilter(use_a1=False, use_a2=False, use_b1=False, use_b2=False, use_c1=False, use_c2=False, use_c3=False, use_c4=False, use_d1=True, use_d2=False, use_d3=False, use_d4=False),
        "D2_Sweep_PDL_Buy": DirectionalBiasFilter(use_a1=False, use_a2=False, use_b1=False, use_b2=False, use_c1=False, use_c2=False, use_c3=False, use_c4=False, use_d1=False, use_d2=True, use_d3=False, use_d4=False),
        "D3_Sweep_PM_High_Sell": DirectionalBiasFilter(use_a1=False, use_a2=False, use_b1=False, use_b2=False, use_c1=False, use_c2=False, use_c3=False, use_c4=False, use_d1=False, use_d2=False, use_d3=True, use_d4=False),
        "D4_Sweep_PM_Low_Buy": DirectionalBiasFilter(use_a1=False, use_a2=False, use_b1=False, use_b2=False, use_c1=False, use_c2=False, use_c3=False, use_c4=False, use_d1=False, use_d2=False, use_d3=False, use_d4=True),

        # 6. Groupements Acheteurs vs Vendeurs
        "Group_All_Bullish_Biases": DirectionalBiasFilter(use_a1=True, use_a2=False, use_b1=True, use_b2=False, use_c1=False, use_c2=True, use_c3=False, use_c4=True, use_d1=False, use_d2=True, use_d3=False, use_d4=True),
        "Group_All_Bearish_Biases": DirectionalBiasFilter(use_a1=False, use_a2=True, use_b1=False, use_b2=True, use_c1=True, use_c2=False, use_c3=True, use_c4=False, use_d1=True, use_d2=False, use_d3=True, use_d4=False),

        # 7. Combinaisons avec Confluence (ex: nécessite 2 signaux alignés, interdit si contradiction)
        "Confluence_2_Signals": DirectionalBiasFilter(require_confluence=2, mode="filter"),
        "Exclusive_Alignment_Only": DirectionalBiasFilter(require_confluence=1, mode="exclusive"),
    }

    rows = []
    for name, b_filter in variants.items():
        res = evaluate_filter_variant(name, b_filter, val_feat, val_m1, feature_cols)
        rows.append(res)

    df_res = pd.DataFrame(rows)
    cols_order = ["filter_variant", "total_return_pct", "profit_factor", "win_rate_pct", "max_drawdown_pct", "sharpe_like", "n_trades"]
    cols = [c for c in cols_order if c in df_res.columns] + [c for c in df_res.columns if c not in cols_order]
    return df_res[cols]


def main():
    parser = argparse.ArgumentParser(
        description="Évaluation systématique des biais directionnels A, B, C, D."
    )
    parser.add_argument("--symbol", "-s", default=None, help="Symbole de l'instrument")
    parser.add_argument("--csv", "-c", default=None, help="Chemin vers le fichier CSV")
    parser.add_argument("--is-crypto", action="store_true", help="Marché 24/7 crypto")
    parser.add_argument("--time-col", default=None, help="Colonne temporelle")
    parser.add_argument("--source-tz", default=None, help="Fuseau horaire d'origine")
    parser.add_argument("--decision-tf", default=None, help="Timeframe de décision")
    parser.add_argument("--out-dir", "-o", default="outputs/bias_eval", help="Répertoire de sortie")
    args = parser.parse_args()

    if args.symbol: CFG.symbol = args.symbol
    if args.csv: CFG.csv_path = Path(args.csv)
    if args.is_crypto: CFG.is_crypto_24_7 = True
    if args.time_col: CFG.time_col = args.time_col
    if args.source_tz: CFG.source_tz = args.source_tz
    if args.decision_tf: CFG.decision_timeframe = args.decision_tf

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement des données ({CFG.symbol})...")
    m1 = load_mt_ohlcv_csv(
        CFG.csv_path, time_col=CFG.time_col, source_tz=CFG.source_tz,
        timestamp_is_bar_open=CFG.timestamp_is_bar_open,
        bar_duration=CFG.pandas_execution_tf,
    )
    decision = resample_ohlcv(m1, CFG.pandas_tf)
    feat, feature_cols = prepare_feature_frame(
        decision, warmup_bars=CFG.warmup_bars,
        atr_period=CFG.atr_period, rsi_period=CFG.rsi_period,
    )

    print("Évaluation systématique des variantes de filtres directionnels...")
    results = run_bias_benchmark(feat, m1, feature_cols)

    csv_path = out_dir / "bias_benchmark_results.csv"
    results.to_csv(csv_path, index=False)
    print("\n" + "=" * 80)
    print("  RÉSULTATS DE L'ÉVALUATION DES BIAIS DIRECTIONNELS (Validation Hors-Échantillon)")
    print("=" * 80)
    print(results.to_string(index=False))
    print(f"\nRapport complet sauvegardé -> {csv_path}")


if __name__ == "__main__":
    main()
