"""
directional_biases.py
---------------------
Module de décomposition et calcul des biais directionnels en heure de New York (America/New_York).
Gère automatiquement les transitions d'heure d'été et d'hiver (DST).

Chaque biais est modélisé comme un signal/filtre indépendant, causal (sans lookahead bias) :
  - Biais A : Bougie journalière précédente (A1 Bullish, A2 Bearish, A3 Neutral)
  - Biais B : Bougie 6H précédente RTH (Regular Trading Hours de la veille 09:30 - 15:30/16:00 NY)
  - Biais C : Sweeps de liquidité entre 07:00 et 09:30 NY (PDH, PDL, Session High/Low précédent)
  - Biais D : Sweeps de liquidité entre 09:30 et 10:00 NY (PDH, PDL, Session Pre-market 07:00-09:30 High/Low)

Tous les calculs respectent la causalité : pour la barre t, seules les informations
antérieures ou égales à t sont utilisées.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TZ_NY = "America/New_York"


def ensure_ny_timezone(df: pd.DataFrame, source_tz: str = "UTC") -> pd.DataFrame:
    """Convertit l'index temporel du DataFrame vers le fuseau horaire America/New_York
    avec prise en compte automatique de l'heure d'été et d'hiver (DST).
    """
    out = df.copy()
    idx = out.index
    if idx.tz is None:
        idx = idx.tz_localize(source_tz, ambiguous="infer", nonexistent="shift_forward")
    out.index = idx.tz_convert(TZ_NY)
    out.index.name = "Time_NY"
    return out


@dataclass
class BiasSignals:
    """Structure contenant l'état de tous les biais à une barre donnée."""
    # Biais A (Daily)
    a1_daily_bull: bool = False
    a2_daily_bear: bool = False
    a3_daily_neutral: bool = False

    # Biais B (6H RTH)
    b1_rth6h_bull: bool = False
    b2_rth6h_bear: bool = False
    b3_rth6h_neutral: bool = False

    # Biais C (Sweeps 07:00-09:30)
    c1_sweep_pdh: bool = False       # Vendeur potentiel
    c2_sweep_pdl: bool = False       # Acheteur potentiel
    c3_sweep_prev_high: bool = False # Vendeur potentiel
    c4_sweep_prev_low: bool = False  # Acheteur potentiel

    # Biais D (Sweeps 09:30-10:00)
    d1_sweep_pdh: bool = False       # Vendeur potentiel
    d2_sweep_pdl: bool = False       # Acheteur potentiel
    d3_sweep_premarket_high: bool = False  # Vendeur potentiel
    d4_sweep_premarket_low: bool = False   # Acheteur potentiel

    # Résumé numérique : +1 (Acheteur), -1 (Vendeur), 0 (Neutre)
    net_bias: int = 0


def compute_directional_biases(m1_df: pd.DataFrame) -> pd.DataFrame:
    """Calcule tous les sous-biais directionnels A, B, C, D de façon strictement causale
    sur les barres M1 en heure de New York (America/New_York).

    Retourne un DataFrame aligné sur m1_df avec toutes les colonnes de biais.
    """
    df_ny = ensure_ny_timezone(m1_df)

    dates = df_ny.index.date
    times = df_ny.index.time

    # ─────────────────────────────────────────────────────────────────────────
    # 1. BIAIS A : Bougie journalière précédente (Previous Daily Candle)
    # ─────────────────────────────────────────────────────────────────────────
    # Agréger par jour calendaire NY (00:00 à 23:59 NY)
    daily = df_ny.groupby(dates).agg(
        day_open=("Open", "first"),
        day_high=("High", "max"),
        day_low=("Low", "min"),
        day_close=("Close", "last"),
    )
    # Décaler d'un jour pour avoir la bougie de la veille de façon causale
    prev_daily = daily.shift(1)
    prev_daily["daily_bull"] = (prev_daily["day_close"] > prev_daily["day_open"]).astype(float)
    prev_daily["daily_bear"] = (prev_daily["day_close"] < prev_daily["day_open"]).astype(float)
    prev_daily["daily_neutral"] = (prev_daily["day_close"] == prev_daily["day_open"]).astype(float)

    # Mapper sur chaque barre M1 selon la date
    df_dates = pd.Series(dates, index=df_ny.index)
    pdh_map = prev_daily["day_high"].to_dict()
    pdl_map = prev_daily["day_low"].to_dict()
    bull_map = prev_daily["daily_bull"].to_dict()
    bear_map = prev_daily["daily_bear"].to_dict()
    neutral_map = prev_daily["daily_neutral"].to_dict()

    out = pd.DataFrame(index=df_ny.index)
    out["bias_a1_daily_bull"] = df_dates.map(bull_map).fillna(0.0).astype(float)
    out["bias_a2_daily_bear"] = df_dates.map(bear_map).fillna(0.0).astype(float)
    out["bias_a3_daily_neutral"] = df_dates.map(neutral_map).fillna(0.0).astype(float)

    pdh_series = df_dates.map(pdh_map).astype(float)
    pdl_series = df_dates.map(pdl_map).astype(float)
    out["prev_day_high"] = pdh_series
    out["prev_day_low"] = pdl_series

    # ─────────────────────────────────────────────────────────────────────────
    # 2. BIAIS B : Bougie de 6 heures précédente limitée à la RTH (09:30 - 15:30 NY)
    # ─────────────────────────────────────────────────────────────────────────
    # Filtrer uniquement la fenêtre RTH 6H : 09:30 à 15:30 (ou 16:00)
    # Parenthèses explicites pour respecter les priorités des opérateurs bitwise
    is_rth_6h = ((df_ny.index.hour > 9) | ((df_ny.index.hour == 9) & (df_ny.index.minute >= 30))) & \
                ((df_ny.index.hour < 15) | ((df_ny.index.hour == 15) & (df_ny.index.minute <= 30)))

    rth_df = df_ny[is_rth_6h]
    rth_daily = rth_df.groupby(rth_df.index.date).agg(
        rth_open=("Open", "first"),
        rth_close=("Close", "last"),
        rth_high=("High", "max"),
        rth_low=("Low", "min"),
    )
    # Décaler d'un jour (données RTH de la veille)
    prev_rth = rth_daily.shift(1)
    prev_rth["rth_bull"] = (prev_rth["rth_close"] > prev_rth["rth_open"]).astype(float)
    prev_rth["rth_bear"] = (prev_rth["rth_close"] < prev_rth["rth_open"]).astype(float)
    prev_rth["rth_neutral"] = (prev_rth["rth_close"] == prev_rth["rth_open"]).astype(float)

    rth_bull_map = prev_rth["rth_bull"].to_dict()
    rth_bear_map = prev_rth["rth_bear"].to_dict()
    rth_neutral_map = prev_rth["rth_neutral"].to_dict()

    out["bias_b1_rth6h_bull"] = df_dates.map(rth_bull_map).fillna(0.0).astype(float)
    out["bias_b2_rth6h_bear"] = df_dates.map(rth_bear_map).fillna(0.0).astype(float)
    out["bias_b3_rth6h_neutral"] = df_dates.map(rth_neutral_map).fillna(0.0).astype(float)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. NIVEAUX DE SESSION PRÉCÉDENTE (Overnight / Asie / Londres avant 07:00 NY)
    # ─────────────────────────────────────────────────────────────────────────
    # Session précédente = bougies entre 18:00 (veille) et 07:00 (jour même)
    # Pour chaque jour, on calcule le High et Low avant 07:00 NY
    is_pre_0700 = df_ny.index.hour < 7
    pre07_df = df_ny[is_pre_0700]
    pre07_levels = pre07_df.groupby(pre07_df.index.date).agg(
        prev_sess_high=("High", "max"),
        prev_sess_low=("Low", "min"),
    )
    prev_sess_h_map = pre07_levels["prev_sess_high"].to_dict()
    prev_sess_l_map = pre07_levels["prev_sess_low"].to_dict()

    prev_sess_h = df_dates.map(prev_sess_h_map).astype(float)
    prev_sess_l = df_dates.map(prev_sess_l_map).astype(float)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. BIAIS C : Sweeps entre 07:00 et 09:30 NY
    # ─────────────────────────────────────────────────────────────────────────
    # Un "Sweep" se produit lorsque le prix dépasse le niveau de référence
    # (High ou Low) puis réintègre (Close < High pour un High sweep, Close > Low pour un Low sweep).
    minute_total = df_ny.index.hour * 60 + df_ny.index.minute
    in_window_c = (minute_total >= 7 * 60) & (minute_total < 9 * 60 + 30)

    high = df_ny["High"]
    low = df_ny["Low"]
    close = df_ny["Close"]

    # C1 : Sweep du Previous Day High (High > PDH et Close < PDH) -> Biais vendeur
    c1 = in_window_c & (high > pdh_series) & (close < pdh_series)
    # C2 : Sweep du Previous Day Low (Low < PDL et Close > PDL) -> Biais acheteur
    c2 = in_window_c & (low < pdl_series) & (close > pdl_series)
    # C3 : Sweep du High de la session précédente -> Biais vendeur
    c3 = in_window_c & (high > prev_sess_h) & (close < prev_sess_h)
    # C4 : Sweep du Low de la session précédente -> Biais acheteur
    c4 = in_window_c & (low < prev_sess_l) & (close > prev_sess_l)

    # Une fois activé dans la journée/session, le signal de sweep persiste jusqu'à la fin de la journée
    # pour informer les décisions de trading
    out["bias_c1_sweep_pdh"] = c1.groupby(dates).cummax().astype(float)
    out["bias_c2_sweep_pdl"] = c2.groupby(dates).cummax().astype(float)
    out["bias_c3_sweep_prev_high"] = c3.groupby(dates).cummax().astype(float)
    out["bias_c4_sweep_prev_low"] = c4.groupby(dates).cummax().astype(float)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. BIAIS D : Sweeps entre 09:30 et 10:00 NY
    # ─────────────────────────────────────────────────────────────────────────
    # Niveaux de la session Pre-market 07:00–09:30 du jour
    premarket_df = df_ny[in_window_c]
    pm_levels = premarket_df.groupby(premarket_df.index.date).agg(
        pm_high=("High", "max"),
        pm_low=("Low", "min"),
    )
    pm_h_map = pm_levels["pm_high"].to_dict()
    pm_l_map = pm_levels["pm_low"].to_dict()
    pm_high = df_dates.map(pm_h_map).astype(float)
    pm_low = df_dates.map(pm_l_map).astype(float)

    in_window_d = (minute_total >= 9 * 60 + 30) & (minute_total < 10 * 60)

    # D1 : Sweep du Previous Day High (09:30-10:00) -> Biais vendeur
    d1 = in_window_d & (high > pdh_series) & (close < pdh_series)
    # D2 : Sweep du Previous Day Low (09:30-10:00) -> Biais acheteur
    d2 = in_window_d & (low < pdl_series) & (close > pdl_series)
    # D3 : Sweep du High 07:00–09:30 -> Biais vendeur
    d3 = in_window_d & (high > pm_high) & (close < pm_high)
    # D4 : Sweep du Low 07:00–09:30 -> Biais acheteur
    d4 = in_window_d & (low < pm_low) & (close > pm_low)

    out["bias_d1_sweep_pdh"] = d1.groupby(dates).cummax().astype(float)
    out["bias_d2_sweep_pdl"] = d2.groupby(dates).cummax().astype(float)
    out["bias_d3_sweep_pm_high"] = d3.groupby(dates).cummax().astype(float)
    out["bias_d4_sweep_pm_low"] = d4.groupby(dates).cummax().astype(float)

    # Net bullish/bearish score (-1 à +1)
    bull_score = (
        out["bias_a1_daily_bull"] +
        out["bias_b1_rth6h_bull"] +
        out["bias_c2_sweep_pdl"] +
        out["bias_c4_sweep_prev_low"] +
        out["bias_d2_sweep_pdl"] +
        out["bias_d4_sweep_pm_low"]
    )
    bear_score = (
        out["bias_a2_daily_bear"] +
        out["bias_b2_rth6h_bear"] +
        out["bias_c1_sweep_pdh"] +
        out["bias_c3_sweep_prev_high"] +
        out["bias_d1_sweep_pdh"] +
        out["bias_d3_sweep_pm_high"]
    )
    out["net_bias_score"] = (bull_score - bear_score).clip(-5.0, 5.0)

    return out


class DirectionalBiasFilter:
    """Filtre directionnel modulaire permettant d'activer/désactiver indépendamment
    chaque sous-biais, et de contraindre l'exécution des trades :
      - Direction autorisée : Long (+1), Short (-1), ou Neutre/Aucun trade (0).
    """

    def __init__(
        self,
        use_a1: bool = True,
        use_a2: bool = True,
        use_b1: bool = True,
        use_b2: bool = True,
        use_c1: bool = True,
        use_c2: bool = True,
        use_c3: bool = True,
        use_c4: bool = True,
        use_d1: bool = True,
        use_d2: bool = True,
        use_d3: bool = True,
        use_d4: bool = True,
        require_confluence: int = 1,
        mode: str = "filter",  # "filter" (filtre d'action), "exclusive" (trade uniquement si biais aligné)
    ):
        self.use_a1 = use_a1
        self.use_a2 = use_a2
        self.use_b1 = use_b1
        self.use_b2 = use_b2
        self.use_c1 = use_c1
        self.use_c2 = use_c2
        self.use_c3 = use_c3
        self.use_c4 = use_c4
        self.use_d1 = use_d1
        self.use_d2 = use_d2
        self.use_d3 = use_d3
        self.use_d4 = use_d4
        self.require_confluence = require_confluence
        self.mode = mode

    def get_allowed_direction(self, row: pd.Series | dict | np.ndarray, feature_cols: list[str] | None = None) -> int:
        """Détermine la direction autorisée :
            +1 : Uniquement Longs autorisés
            -1 : Uniquement Shorts autorisés
             0 : Aucun trade autorisé (conditions défavorables ou contradiction)
            None / Any : Pas de filtre strict
        """
        if self.mode == "none":
            return None

        # Conversion si row est un environnement Gymnasium (BracketTradingEnv)
        if hasattr(row, "_current_row"):
            current_s = row._current_row()
            row = current_s.to_dict()
        # Conversion si row est un numpy array (venant d'une observation Gym)
        elif isinstance(row, np.ndarray):
            if feature_cols is not None:
                row = {col: float(row[i]) for i, col in enumerate(feature_cols) if i < len(row)}
            else:
                row = {}

        bull_votes = 0
        bear_votes = 0

        # Biais A
        if self.use_a1 and row.get("bias_a1_daily_bull", 0) > 0: bull_votes += 1
        if self.use_a2 and row.get("bias_a2_daily_bear", 0) > 0: bear_votes += 1

        # Biais B
        if self.use_b1 and row.get("bias_b1_rth6h_bull", 0) > 0: bull_votes += 1
        if self.use_b2 and row.get("bias_b2_rth6h_bear", 0) > 0: bear_votes += 1

        # Biais C
        if self.use_c2 and row.get("bias_c2_sweep_pdl", 0) > 0: bull_votes += 1
        if self.use_c4 and row.get("bias_c4_sweep_prev_low", 0) > 0: bull_votes += 1
        if self.use_c1 and row.get("bias_c1_sweep_pdh", 0) > 0: bear_votes += 1
        if self.use_c3 and row.get("bias_c3_sweep_prev_high", 0) > 0: bear_votes += 1

        # Biais D
        if self.use_d2 and row.get("bias_d2_sweep_pdl", 0) > 0: bull_votes += 1
        if self.use_d4 and row.get("bias_d4_sweep_pm_low", 0) > 0: bull_votes += 1
        if self.use_d1 and row.get("bias_d1_sweep_pdh", 0) > 0: bear_votes += 1
        if self.use_d3 and row.get("bias_d3_sweep_pm_high", 0) > 0: bear_votes += 1

        if bull_votes >= self.require_confluence and bear_votes == 0:
            return 1
        elif bear_votes >= self.require_confluence and bull_votes == 0:
            return -1
        elif bull_votes >= self.require_confluence and bear_votes >= self.require_confluence:
            # Contradiction flagrante entre les biais : interdiction de trader
            return 0
        else:
            # Pas assez de confluence
            return 0 if self.mode == "exclusive" else None
