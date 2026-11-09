from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from directional_biases import (
    TZ_NY,
    DirectionalBiasFilter,
    compute_directional_biases,
    ensure_ny_timezone,
)
from leakage_checks import assert_feature_stability_when_future_appended
from features import add_stationary_features


def test_ensure_ny_timezone_dst():
    # Test during summer time (EDT = UTC-4) and winter time (EST = UTC-5)
    # Summer: July
    times_summer = pd.date_range("2024-07-01 12:00:00", periods=2, freq="1h", tz="UTC")
    df_summer = pd.DataFrame({"Close": [100.0, 101.0]}, index=times_summer)
    ny_summer = ensure_ny_timezone(df_summer)
    assert ny_summer.index[0].hour == 8  # 12 UTC - 4h = 8 AM NY

    # Winter: January
    times_winter = pd.date_range("2024-01-01 12:00:00", periods=2, freq="1h", tz="UTC")
    df_winter = pd.DataFrame({"Close": [100.0, 101.0]}, index=times_winter)
    ny_winter = ensure_ny_timezone(df_winter)
    assert ny_winter.index[0].hour == 7  # 12 UTC - 5h = 7 AM NY


def test_directional_biases_computation():
    # Créer 2 jours de données 1m en temps UTC
    dates = pd.date_range("2024-01-01 00:00:00", "2024-01-02 23:59:00", freq="1min", tz="UTC")
    n = len(dates)
    np.random.seed(42)

    # Jour 1 haussier (Close > Open)
    # Jour 2 baissier (Close < Open)
    base_price = 40000.0
    drift = np.linspace(0, 100, n // 2)
    drift2 = np.linspace(100, -50, n - n // 2)
    trend = np.concatenate([drift, drift2])
    closes = base_price + trend
    highs = closes + 10.0
    lows = closes - 10.0
    opens = closes - 1.0

    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 10.0
    }, index=dates)

    biases = compute_directional_biases(df)
    assert "bias_a1_daily_bull" in biases.columns
    assert "bias_a2_daily_bear" in biases.columns
    assert "bias_b1_rth6h_bull" in biases.columns
    assert "bias_c1_sweep_pdh" in biases.columns
    assert "bias_c2_sweep_pdl" in biases.columns
    assert "bias_d1_sweep_pdh" in biases.columns
    assert "bias_d3_sweep_pm_high" in biases.columns
    assert "net_bias_score" in biases.columns


def test_directional_bias_filter_decisions():
    # Test filter decisions
    b_filter = DirectionalBiasFilter(require_confluence=1, mode="filter")

    # Row with Bullish bias A1
    row_bull = {"bias_a1_daily_bull": 1.0, "bias_a2_daily_bear": 0.0}
    assert b_filter.get_allowed_direction(row_bull) == 1

    # Row with Bearish bias A2
    row_bear = {"bias_a1_daily_bull": 0.0, "bias_a2_daily_bear": 1.0}
    assert b_filter.get_allowed_direction(row_bear) == -1

    # Row with contradictory bias (both bull and bear) -> must forbid trading (0)
    row_conflict = {"bias_a1_daily_bull": 1.0, "bias_a2_daily_bear": 1.0}
    assert b_filter.get_allowed_direction(row_conflict) == 0

    # Test with numpy array input (like in Gym observation)
    feat_cols = ["bias_a1_daily_bull", "bias_a2_daily_bear"]
    arr_bull = np.array([1.0, 0.0], dtype=np.float32)
    assert b_filter.get_allowed_direction(arr_bull, feature_cols=feat_cols) == 1

    # Test no filter mode
    no_filter = DirectionalBiasFilter(mode="none")
    assert no_filter.get_allowed_direction(row_bull) is None
    assert no_filter.get_allowed_direction(arr_bull, feature_cols=feat_cols) is None


def test_directional_biases_no_future_leakage():
    # Test mechanical causality guardrail
    n_bars = 400
    times = pd.date_range("2024-01-01", periods=n_bars, freq="5min", tz="UTC")
    np.random.seed(42)
    closes = 50000.0 + np.cumsum(np.random.normal(0, 10, n_bars))
    highs = closes + np.random.uniform(5, 20, n_bars)
    lows = closes - np.random.uniform(5, 20, n_bars)
    opens = closes + np.random.uniform(-5, 5, n_bars)
    vols = np.random.uniform(10, 100, n_bars)

    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols
    }, index=times)

    _, feature_cols = add_stationary_features(df)
    bias_cols = [c for c in feature_cols if c.startswith("bias_")]
    assert len(bias_cols) > 0

    # Ensure no lookahead: appending future bars does not change past bias signals
    assert assert_feature_stability_when_future_appended(df, bias_cols, cut_index=150)


def test_evaluate_biases_benchmark_end_to_end():
    from evaluate_biases import run_bias_benchmark
    from features import prepare_feature_frame

    n_bars = 300
    times = pd.date_range("2024-01-01", periods=n_bars, freq="15min", tz="UTC")
    np.random.seed(42)
    closes = 50000.0 + np.cumsum(np.random.normal(0, 10, n_bars))
    highs = closes + np.random.uniform(5, 20, n_bars)
    lows = closes - np.random.uniform(5, 20, n_bars)
    opens = closes + np.random.uniform(-5, 5, n_bars)
    vols = np.random.uniform(10, 100, n_bars)

    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols}, index=times)
    feat, feature_cols = prepare_feature_frame(df, warmup_bars=50)

    m1_times = pd.date_range("2024-01-01", periods=n_bars * 15, freq="1min", tz="UTC")
    m1_df = pd.DataFrame({"Open": 50000.0, "High": 50020.0, "Low": 49980.0, "Close": 50000.0, "Volume": 1.0}, index=m1_times)

    bench = run_bias_benchmark(feat, m1_df, feature_cols)
    assert len(bench) >= 10
    assert "00_No_Filter" in bench["filter_variant"].values
    assert "A1_Daily_Bullish_Only" in bench["filter_variant"].values
    assert "Confluence_2_Signals" in bench["filter_variant"].values
