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

## Guide des commandes et exemples d'utilisation (CLI)

Le projet permet de tout configurer directement par **options en ligne de commande** sans avoir à modifier le code, que ce soit pour le téléchargement, le pipeline de test, l'entraînement RL ou la visualisation de la courbe d'apprentissage.

---

### 1. Télécharger des bougies 1m via CCXT (`download_ccxt.py`)

Téléchargez des données historiques 1-minute depuis n'importe quelle plateforme crypto (Binance, Bybit, OKX, Kraken, Coinbase, etc.) :

```bash
# Exemple 1 : Télécharger les 90 derniers jours de BTC/USDT depuis Binance
python download_ccxt.py --symbol BTC/USDT --exchange binance --days 90

# Exemple 2 : Télécharger une année complète d'ETH/USDT avec chemin de sortie personnalisé
python download_ccxt.py --symbol ETH/USDT --exchange binance --since 2023-01-01 --until 2024-01-01 --output data/binance_ETHUSDT_1m.csv

# Exemple 3 : Télécharger SOL/USDT sur Bybit pour les 180 derniers jours
python download_ccxt.py --symbol SOL/USDT --exchange bybit --days 180

# Exemple 4 : Télécharger EUR/USD sur Kraken pour un test
python download_ccxt.py --symbol EUR/USD --exchange kraken --days 30
```

---

### 2. Entraîner l'agent RL avec PPO (`train_ppo.py`)

Les options en ligne de commande permettent de spécifier :
- L'instrument (`--symbol`)
- Le fichier CSV (`--csv`)
- Le mode de marché (`--is-crypto` pour calendrier continu 24/7)
- Le fuseau horaire (`--source-tz UTC`)
- Le nom du modèle sauvegardé (`--model-name`), qui est automatiquement combiné avec le symbole et la date/heure (`YYYYMMDD_HHMMSS`)

```bash
# Exemple 1 : Entraînement sur BTC/USDT avec nom personnalisé et horodatage automatique
python train_ppo.py --symbol BTC/USDT --csv data/binance_BTCUSDT_1m.csv --is-crypto --source-tz UTC --time-col Time --decision-tf 15min --model-name MonModeleBTC --mode single --timesteps 500000

# Exemple 2 : Entraînement glissant (sliding walk-forward) sur ETH/USDT
python train_ppo.py --symbol ETH/USDT --csv data/binance_ETHUSDT_1m.csv --is-crypto --source-tz UTC --time-col Time --decision-tf 5min --mode sliding --timesteps 1000000

# Exemple 3 : Entraînement par blocs walk-forward sur instrument classique (XAUUSD)
python train_ppo.py --symbol XAUUSD --csv "data/XAUUSD_1 Min_Bid_2003.05.05_2026.05.31.csv" --time-col "Time (EET)" --source-tz Europe/Helsinki --decision-tf H1 --model-name XAU_H1_Agent --mode walk_forward

# Exemple 4 : Test rapide (smoke-test) avec 1 seul environnement CPU et budget réduit
python train_ppo.py --symbol BTC/USDT --csv data/binance_BTCUSDT_1m.csv --is-crypto --source-tz UTC --decision-tf 5min --timesteps 50000 --n-envs 1 --mode single
```

---

### 3. Exécuter le pipeline de validation (`run_pipeline.py`)

Exécute le resampling, la création des features, la vérification de fuite temporelle, et l'optimisation des baselines avec un instrument personnalisé :

```bash
# Exemple 1 : Pipeline pour BTC/USDT en barres de décision 15min
python run_pipeline.py --symbol BTC/USDT --csv data/binance_BTCUSDT_1m.csv --is-crypto --source-tz UTC --time-col Time --decision-tf 15min --out-dir outputs/btc_15m

# Exemple 2 : Pipeline pour SOL/USDT en 5min
python run_pipeline.py --symbol SOL/USDT --csv data/bybit_SOLUSDT_1m.csv --is-crypto --source-tz UTC --decision-tf 5min

# Exemple 3 : Pipeline standard sur XAUUSD
python run_pipeline.py
```

---

### 4. Visualiser la courbe d'apprentissage (`plot_learning_curve.py`)

Affiche le graphique interactif Plotly montrant les récompenses de train vs validation, les drawdowns max %, et le meilleur checkpoint sélectionné :

```bash
# Exemple 1 : Détection automatique du run le plus récent et ouverture dans le navigateur
python plot_learning_curve.py

# Exemple 2 : Inspecter un fold walk-forward spécifique (ex. fold 1)
python plot_learning_curve.py --fold 1

# Exemple 3 : Sauvegarder en fichier HTML sans ouvrir le navigateur
python plot_learning_curve.py --output outputs/courbe_apprentissage.html --no-browser

# Exemple 4 : Spécifier explicitement un fichier consistency_evals.csv
python plot_learning_curve.py --file models/walk_forward/fold_2/eval_logs/consistency_evals.csv
```

---

### 5. Diagnostics complets d'entraînement (`training_diagnostics.py`)

Analyse la courbe d'apprentissage, la qualité des features (stationnarité, corrélation) et le surapprentissage :

```bash
# Exemple 1 : Diagnostics sur le modèle par défaut dans models/
python training_diagnostics.py

# Exemple 2 : Diagnostics sur un modèle entraîné sur un instrument custom
python training_diagnostics.py --models-dir models --symbol BTC/USDT --csv data/binance_BTCUSDT_1m.csv --is-crypto --source-tz UTC --time-col Time

# Exemple 3 : Révéler les diagnostics sur le split test holdout scellé
python training_diagnostics.py --reveal-test
```

---

### 6. Visualiser les trades et les résultats (`view_results.py`)

Génère les graphiques chandeliers, trades exécutés et courbes d'équité :

```bash
# Exemple 1 : Voir les 500 dernières barres
python view_results.py --csv data/binance_BTCUSDT_1m.csv --symbol BTC/USDT --source-tz UTC --bars 500

# Exemple 2 : Visualiser une période spécifique
python view_results.py --start 2024-01-01 --end 2024-03-01 --no-browser

# Exemple 3 : Visualiser le split de validation
python view_results.py --split val
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
