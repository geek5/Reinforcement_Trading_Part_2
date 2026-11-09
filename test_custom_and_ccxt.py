from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from config import ProjectConfig
from data_loader import load_mt_ohlcv_csv, resample_ohlcv
from download_ccxt import (
    download_ohlcv,
    get_exchange_instance,
    parse_timestamp_to_ms,
    save_ohlcv_csv,
)
from features import prepare_feature_frame
from plot_learning_curve import resolve_evals_path
from visualize import plot_learning_curve


def test_parse_timestamp_to_ms():
    assert parse_timestamp_to_ms(None) is None
    ms = parse_timestamp_to_ms("2024-01-01 00:00:00")
    assert ms == 1704067200000

    ms_date_only = parse_timestamp_to_ms("2024-01-01")
    assert ms_date_only == 1704067200000

    with pytest.raises(ValueError):
        parse_timestamp_to_ms("invalid-date-string-xyz")


def test_get_exchange_instance():
    ex = get_exchange_instance("binance")
    assert ex.id == "binance"
    assert ex.enableRateLimit is True

    with pytest.raises(ValueError):
        get_exchange_instance("non_existent_crypto_exchange_123")


def test_download_ohlcv_mocked():
    mock_exchange = MagicMock()
    mock_exchange.id = "mock_exchange"
    mock_exchange.has = {"fetchOHLCV": True}
    mock_exchange.parse_timeframe.return_value = 60  # 1 min in sec
    mock_exchange.milliseconds.return_value = 1704070800000

    # 3 mock candles
    c1 = [1704067200000, 40000.0, 40100.0, 39950.0, 40050.0, 10.0]
    c2 = [1704067260000, 40050.0, 40150.0, 40020.0, 40120.0, 15.0]
    c3 = [1704067320000, 40120.0, 40200.0, 40100.0, 40180.0, 20.0]

    mock_exchange.fetch_ohlcv.side_effect = [
        [c1, c2],
        [c3],
        [],
    ]

    df = download_ohlcv(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        timeframe="1m",
        since_ms=1704067200000,
        until_ms=1704067320000,
        limit_per_request=2,
        verbose=False,
    )

    assert len(df) == 3
    assert list(df.columns) == ["Timestamp", "Time", "Open", "High", "Low", "Close", "Volume"]
    assert df["Close"].iloc[-1] == 40180.0
    assert df["Time"].iloc[0] == "2024-01-01 00:00:00"


def test_save_and_load_ccxt_csv(tmp_path: Path):
    df_raw = pd.DataFrame({
        "Timestamp": [1704067200000, 1704067260000, 1704067320000, 1704067380000],
        "Time": [
            "2024-01-01 00:00:00",
            "2024-01-01 00:01:00",
            "2024-01-01 00:02:00",
            "2024-01-01 00:03:00",
        ],
        "Open": [40000.0, 40050.0, 40100.0, 40150.0],
        "High": [40100.0, 40120.0, 40180.0, 40200.0],
        "Low": [39950.0, 40020.0, 40080.0, 40120.0],
        "Close": [40050.0, 40100.0, 40150.0, 40190.0],
        "Volume": [10.0, 12.0, 8.0, 14.0],
    })
    csv_file = tmp_path / "btc_1m.csv"
    save_ohlcv_csv(df_raw, csv_file)

    # 1. Test load with time_col='Time' (ISO string UTC)
    loaded_iso = load_mt_ohlcv_csv(csv_file, time_col="Time", source_tz="UTC")
    assert len(loaded_iso) == 4
    assert str(loaded_iso.index.tz) in ("UTC", "datetime.timezone.utc", "UTC+00:00")
    assert "Close" in loaded_iso.columns

    # 2. Test load with time_col='Timestamp' (Unix ms numeric)
    loaded_ms = load_mt_ohlcv_csv(csv_file, time_col="Timestamp", source_tz="UTC")
    assert len(loaded_ms) == 4
    assert loaded_ms.index.equals(loaded_iso.index)


def test_custom_instrument_config_and_pipeline(tmp_path: Path):
    # Verify 24/7 vs 23/5 periods_per_year
    cfg_trad = ProjectConfig(is_crypto_24_7=False, decision_timeframe="H1")
    assert cfg_trad.periods_per_year == 6_003

    cfg_crypto = ProjectConfig(is_crypto_24_7=True, decision_timeframe="H1")
    assert cfg_crypto.periods_per_year == 8_766

    # Create dummy 1m candles for 1500 bars to test feature pipeline (ema200 requires >= 200 bars)
    n_bars = 1500
    times = pd.date_range("2024-01-01", periods=n_bars, freq="1min", tz="UTC")
    np.random.seed(42)
    base_price = 50000.0
    changes = np.random.normal(0, 10, n_bars)
    closes = base_price + np.cumsum(changes)
    highs = closes + np.random.uniform(5, 15, n_bars)
    lows = closes - np.random.uniform(5, 15, n_bars)
    opens = closes + np.random.uniform(-5, 5, n_bars)
    vols = np.random.uniform(1, 10, n_bars)

    df_synth = pd.DataFrame({
        "Time": times.strftime("%Y-%m-%d %H:%M:%S"),
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": vols,
    })
    synth_csv = tmp_path / "custom_crypto_1m.csv"
    df_synth.to_csv(synth_csv, index=False)

    loaded = load_mt_ohlcv_csv(synth_csv, time_col="Time", source_tz="UTC")
    resampled = resample_ohlcv(loaded, "5min")
    assert len(resampled) == n_bars // 5

    feat, feature_cols = prepare_feature_frame(resampled, warmup_bars=200)
    assert len(feat) > 0
    assert "whale_buy_zscore" in feature_cols
    assert "whale_sell_zscore" in feature_cols
    assert "whale_net_vol_ratio" in feature_cols
    assert "whale_buy_tier" in feature_cols
    assert "whale_sell_tier" in feature_cols


def test_plot_learning_curve(tmp_path: Path):
    evals = pd.DataFrame({
        "timesteps": [100000, 200000, 300000],
        "train_eval_r": [15.2, 25.4, 40.1],
        "val_r": [10.1, 20.3, 35.0],
        "train_dd_pct": [10.0, 7.0, 5.0],
        "val_dd_pct": [12.0, 8.0, 6.0],
        "score": [5.2, 18.4, 30.1],
        "eligible": [True, True, True],
    })

    fig = plot_learning_curve(evals)
    assert fig is not None

    csv_path = tmp_path / "consistency_evals.csv"
    evals.to_csv(csv_path, index=False)

    resolved = resolve_evals_path(tmp_path, explicit_file=str(csv_path))
    assert resolved == csv_path


def test_cli_parsing_and_model_slug():
    from train_ppo import parse_args as parse_train_args
    from run_pipeline import parse_args as parse_pipeline_args
    from training_diagnostics import parse_args as parse_diag_args
    import sys

    # Test train_ppo CLI parsing
    orig_argv = sys.argv
    try:
        sys.argv = [
            "train_ppo.py",
            "--symbol", "ETH/USDT",
            "--csv", "data/eth.csv",
            "--is-crypto",
            "--time-col", "Time",
            "--source-tz", "UTC",
            "--decision-tf", "15min",
            "--model-name", "MyBestAgent",
            "--mode", "single",
        ]
        args = parse_train_args()
        assert args.symbol == "ETH/USDT"
        assert args.csv == "data/eth.csv"
        assert args.is_crypto is True
        assert args.time_col == "Time"
        assert args.source_tz == "UTC"
        assert args.decision_tf == "15min"
        assert args.model_name == "MyBestAgent"
        assert args.mode == "single"

        # Test run_pipeline CLI parsing
        sys.argv = [
            "run_pipeline.py",
            "--symbol", "SOL/USDT",
            "--is-crypto",
            "--decision-tf", "5min",
        ]
        p_args = parse_pipeline_args()
        assert p_args.symbol == "SOL/USDT"
        assert p_args.is_crypto is True
        assert p_args.decision_tf == "5min"

        # Test training_diagnostics CLI parsing
        sys.argv = [
            "training_diagnostics.py",
            "--models-dir", "custom_models",
            "--symbol", "BTC/USDT",
        ]
        d_args = parse_diag_args()
        assert d_args.models_dir == "custom_models"
        assert d_args.symbol == "BTC/USDT"
    finally:
        sys.argv = orig_argv


def test_big_trades_whale_leakage():
    from leakage_checks import assert_feature_stability_when_future_appended
    from features import add_stationary_features

    n_bars = 300
    times = pd.date_range("2024-01-01", periods=n_bars, freq="5min", tz="UTC")
    np.random.seed(42)
    closes = 40000.0 + np.cumsum(np.random.normal(0, 10, n_bars))
    highs = closes + np.random.uniform(5, 20, n_bars)
    lows = closes - np.random.uniform(5, 20, n_bars)
    opens = closes + np.random.uniform(-5, 5, n_bars)
    vols = np.random.uniform(10, 100, n_bars)

    # Spike a whale buy and sell
    vols[50] = 5000.0
    vols[150] = 6000.0

    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols
    }, index=times)

    _, feature_cols = add_stationary_features(df)
    whale_cols = [c for c in feature_cols if "whale" in c]
    assert len(whale_cols) == 5

    # Check causality guardrail: no future leakage
    assert assert_feature_stability_when_future_appended(df, whale_cols, cut_index=100)
