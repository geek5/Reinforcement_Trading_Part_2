# RL-ready feature + bracket-trading pipeline (Multi-instrument: XAUUSD, Crypto & Custom)

This project provides a complete reinforcement learning trading research pipeline for minute data (MT4/MT5 exports or CCXT crypto downloads):

```csv
Time (EET),Open,High,Low,Close,Volume
2020.01.09 01:00:00,1557.152,1557.452,1555.202,1555.302,0.045
```
or CCXT / UTC format:
```csv
Time,Open,High,Low,Close,Volume
2024-01-01 00:00:00,42250.0,42280.0,42240.0,42270.0,12.54
```

It gives you a complete research pipeline:

1. Load and validate raw M1 data.
2. Parse broker/EET time safely.
3. Resample M1 execution data into M5/M15 decision bars.
4. Build causal stationary features normalized by ATR/price.
5. Visualize candles, indicators, volatility, sessions, feature correlations, trades, equity, and drawdown.
6. Tune a simple trend-following baseline on train/validation splits while keeping the test split sealed.
7. Optionally train PPO with a `MultiDiscrete([direction, SL bucket, TP/R bucket])` action space.
8. Validate with **walk-forward** cross-validation (rolling out-of-sample windows) rather than a single train/val split.
9. Reveal the test split only once via `final_holdout_eval.py` after the model is frozen.

## Quick start

```bash
pip install -r requirements.txt
```

### Télécharger des bougies 1m via CCXT (Custom instrument)

Vous pouvez télécharger l'historique 1m pour n'importe quel actif crypto supporté par CCXT (Binance, Bybit, OKX, Kraken, etc.) :

```bash
# Exemple : Télécharger 90 jours de bougies 1m BTC/USDT depuis Binance
python download_ccxt.py --symbol BTC/USDT --exchange binance --days 90

# Exemple : Télécharger une plage de dates spécifique pour ETH/USDT
python download_ccxt.py --symbol ETH/USDT --exchange binance --since 2023-01-01 --until 2024-01-01 --output data/ETHUSDT_1m.csv
```

### Configuration d'un instrument custom dans `config.py`

Pour utiliser un instrument personnalisé (par ex. `BTC/USDT`), modifiez simplement `config.py` :

```python
CFG.symbol = "BTC/USDT"
CFG.is_crypto_24_7 = True                         # Active le calcul annualisé 24/7 (365 jours/an)
CFG.csv_path = Path("data/binance_BTCUSDT_1m.csv")
CFG.time_col = "Time"                             # Colonne générée par download_ccxt.py
CFG.source_tz = "UTC"                             # UTC pour CCXT
CFG.spread_price = 1.0                            # Ajuster selon l'actif
CFG.slippage_price = 0.5
CFG.commission_per_trade = 0.0                    # Ou frais d'échange
```

Pour XAUUSD ou le Forex classique :

Put your full file here (the long Bid history is the primary dataset; the short
Ask file is kept as a fast smoke-test fallback — switch between them in `config.py`):

```text
data/XAUUSD_1 Min_Bid_2003.05.05_2026.05.31.csv   # primary (~23 years)
data/XAUUSD_1 Min_Ask_2020.01.09_2026.01.15.csv   # smoke-test fallback (~6 years)
```

Then run:

```bash
python run_pipeline.py
```

This keeps the test split sealed and writes validation-only artifacts, including
a `walk_forward_baseline.csv` showing baseline stability across the rolling folds.

Train the RL agent with walk-forward validation (one model per fold, aggregate
out-of-sample report; the final fold is saved as the production model):

```bash
python train_ppo.py            # == train_ppo.train_walk_forward()
```

When you are ready for the one-time holdout reveal, run:

```bash
python final_holdout_eval.py
```

### Visualiser la courbe d'apprentissage (Learning Curve)

Pour afficher et exporter la courbe d'apprentissage interactive (récompenses train/val, drawdowns, score de consistance et meilleur checkpoint) :

```bash
python plot_learning_curve.py                     # Détecte automatiquement le dernier run
python plot_learning_curve.py --fold 1            # Courbe d'apprentissage du fold 1
python plot_learning_curve.py --output curve.html # Enregistre en fichier HTML interactif
```

Vous pouvez aussi exécuter le diagnostic complet :
```bash
python training_diagnostics.py
```

## Important defaults

- `SOURCE_TZ = Europe/Helsinki`, because many brokers label server time as EET/EEST.
- MT4/MT5 M1 timestamps are treated as candle-open timestamps and shifted to candle-close timestamps before resampling/backtesting.
- `DECISION_TIMEFRAME = H1`, while M1 is retained for intrabar TP/SL simulation.
- Features are causal and mostly stationary: ATR-normalized distances, ratios, session flags, candle shape ratios.
- The observation set is **25 features**. Four redundant ones were dropped in the 2026-06-02 collinearity audit (`rsi_centered`, `roc5_atr`, `body_atr`, `ema50_slope5_atr`); each was |r| ≥ 0.96 with a retained feature (two were exact duplicates). The underlying `ema20/50/200` columns are still computed for the baselines and charts.
- **Sliding-window walk-forward** (`train_ppo.train_sliding_walk_forward`, the default `python train_ppo.py` entry point) is the realistic validator: each fold trains on `sliding_train_years` (5y), selects its checkpoint on the next `sliding_val_months` (6m), and is judged out-of-sample on the following `sliding_test_months` (6m); the window then slides `sliding_step_months` (6m) and repeats. This simulates *retraining every 6 months and trading the next 6 months live*. Every fold's test window is stitched into one continuous out-of-sample equity curve (`models/sliding_oos_equity.csv`); the gate runs on the **test** metrics. Because every train window is the same length, equal timesteps per fold = equal passes. ~35 folds over the 23y dataset (raise `sliding_step_months` to reduce).
- **Block walk-forward** (`train_ppo.train_walk_forward`; `config.py`: `n_walk_forward_folds`, `walk_forward_anchored`, `test_frac`) is the alternative: it seals the last `test_frac` of bars and rolls `n_folds` train→val windows over the rest. `test_frac == 1 - train_frac - val_frac`, so the sealed holdout matches the single-split test exactly.
- When TP and SL are both inside the same M1 candle, the simulator assumes SL first. This is deliberately pessimistic.
- Position size is fixed-fractional risk-based; the RL agent controls direction and bracket shape, not size.
- `run_pipeline.py` now performs a temporal train/validation/test split, tunes on train/val, and keeps test sealed by default.
- `training_diagnostics.py` and `train_ppo.py` also keep the test split sealed by default.
- RL rewards are normalized by risk budget rather than raw cash PnL, which is materially more stable for PPO.

## Files

- `config.py`: all main parameters.
- `data_loader.py`: CSV parsing, timezone handling, validation, resampling.
- `features.py`: causal indicators and stationary feature matrix.
- `leakage_checks.py`: simple future-append stability test.
- `env_bracket.py`: Gymnasium-compatible bracket trading environment.
- `baselines.py`: random and EMA/ATR rule policies.
- `evaluate.py`: metrics, trade-log summary, drawdown.
- `visualize.py`: Plotly visualization functions.
- `train_ppo.py`: optional PPO training scaffold.
- `run_pipeline.py`: one-command pre-test pipeline with validation-only outputs.
- `final_holdout_eval.py`: explicit one-time holdout evaluation entry point.
- `notebooks/XAUUSD_RL_Pipeline_Demo.ipynb`: guided notebook.
