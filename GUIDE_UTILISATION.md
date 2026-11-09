# Guide Complet d'Utilisation : Entraînement, Exécution et Interprétation des Résultats

Ce guide détaille pas à pas la procédure complète pour :
1. Configurer le fuseau horaire global `America/New_York` avec gestion automatique de l'heure d'été et d'hiver (DST).
2. Télécharger des données historiques 1 minute via CCXT pour vos contrats futures / cryptos / devises.
3. Exécuter le pipeline de feature engineering (avec détection des **Big Trades / Baleines** et des **biais directionnels A, B, C, D**).
4. Entraîner l'agent RL (PPO) avec nom, date personnalisés et filtrage directionnel strict des futures.
5. Visualiser et interpréter la courbe d'apprentissage interactive.
6. Analyser les diagnostics d'overfitting et visualiser les trades sur les chandeliers.
7. Évaluer systématiquement l'impact des biais directionnels hors-échantillon.

---

## Sommaire
- [Fuseau horaire global : America/New_York et gestion DST](#fuseau-horaire-global--americanew_york-et-gestion-dst)
- [Étape 1 : Téléchargement des bougies 1m via CCXT](#étape-1--téléchargement-des-bougies-1m-via-ccxt)
- [Étape 2 : Validation des données et de la baseline](#étape-2--validation-des-données-et-de-la-baseline)
- [Étape 3 : Entraînement de l'agent RL (PPO) avec filtrage des biais](#étape-3--entraînement-de-lagent-rl-ppo-avec-filtrage-des-biais)
- [Étape 4 : Visualisation de la courbe d'apprentissage](#étape-4--visualisation-de-la-courbe-dapprentissage)
- [Étape 5 : Diagnostics complets d'entraînement](#étape-5--diagnostics-complets-dentraînement)
- [Étape 6 : Visualisation des trades et chandeliers](#étape-6--visualisation-des-trades-et-chandeliers)
- [Étape 7 : Évaluation systématique des biais directionnels A, B, C, D](#étape-7--évaluation-systématique-des-biais-directionnels-a-b-c-d)
- [Grille de lecture : comment savoir si la stratégie fonctionne](#grille-de-lecture--comment-savoir-si-la-stratégie-fonctionne)

---

## Fuseau horaire global : America/New_York et gestion DST

Le projet utilise de manière centralisée le fuseau horaire **`America/New_York`** (EST/EDT) :
- **Gestion automatique DST** : Les passages à l'heure d'été (EDT, UTC-4) et à l'heure d'hiver (EST, UTC-5) sont calculés dynamiquement sans ambiguïté.
- **Préservation UTC** : Les données sources en UTC ou timestamps Unix sont converties proprement vers `America/New_York` pour tous les calculs de sessions (Asie, Londres, New York RTH 09:30-16:00, fenêtres de sweep 07:00-09:30 et 09:30-10:00).

---

## Étape 1 : Téléchargement des bougies 1m via CCXT

Le script `download_ccxt.py` permet de télécharger des bougies historiques 1-minute depuis n'importe quelle plateforme crypto supportée par CCXT (Binance, Bybit, OKX, Kraken, Coinbase, etc.).

### Commande
```bash
# Exemple : Télécharger les 90 derniers jours de BTC/USDT depuis Binance
python download_ccxt.py --symbol BTC/USDT --exchange binance --days 90
```

### Autres exemples de téléchargement
```bash
# Télécharger une période calendaire précise
python download_ccxt.py --symbol ETH/USDT --exchange binance --since 2023-01-01 --until 2024-01-01 --output data/binance_ETHUSDT_1m.csv

# Télécharger SOL/USDT sur Bybit
python download_ccxt.py --symbol SOL/USDT --exchange bybit --days 180
```

### Ce que vous devez attendre
- **Barre de progression (`tqdm`)** affichant le nombre de bougies téléchargées par blocs de 1 000.
- **Message de confirmation** :
  ```text
  Téléchargement terminé : 129,600 bougies uniques.
  Période couverte : 2023-11-01 00:00:00 -> 2024-01-30 00:00:00 UTC
  Fichier sauvegardé -> /app/data/binance_BTCUSDT_1m.csv
  ```
- **Validation** : Le fichier CSV est créé dans le dossier `data/` avec les colonnes `Timestamp`, `Time`, `Open`, `High`, `Low`, `Close`, `Volume`.

---

## Étape 2 : Validation des données et de la baseline

Le script `run_pipeline.py` prépare les features causales (incluant les Z-scores et ratios de volume des **Big Trades / Baleines**), vérifie l'absence de fuite temporelle (*data leakage*), et calibre une stratégie de référence (baseline).

### Commande
```bash
python run_pipeline.py \
  --symbol BTC/USDT \
  --csv data/binance_BTCUSDT_1m.csv \
  --is-crypto \
  --source-tz UTC \
  --time-col Time \
  --decision-tf 15min \
  --out-dir outputs/btc_15m
```

### Ce qu'il faut observer
- `Leakage check passed: appending future bars did not change past features.` : Confirme que le feature engineering est strictement causal sans regard vers le futur.
- Un tableau récapitulatif `Performance report` s'affiche avec le rendement, le profit factor et le drawdown de la baseline.
- Des graphiques HTML interactifs sont générés dans `outputs/btc_15m/`.

---

## Étape 3 : Entraînement de l'agent RL (PPO) avec filtrage des biais

Le script `train_ppo.py` gère l'apprentissage par renforcement. Vous pouvez personnaliser :
- Le symbole (`--symbol`)
- Le fichier source (`--csv`)
- Le mode de marché (`--is-crypto` pour calendrier continu 24/7)
- Le fuseau horaire (`--source-tz America/New_York` ou `UTC`)
- Le nom du modèle sauvegardé (`--model-name`), combiné automatiquement avec le symbole et un horodatage (`YYYYMMDD_HHMMSS`).
- Le filtrage directionnel strict (`--use-bias-filter`) : PPO ne peut ouvrir que dans le sens autorisé par le biais actif (Longs si bullish, Shorts si bearish, aucun trade si conflit).

### Commande
```bash
python train_ppo.py \
  --symbol BTC/USDT \
  --csv data/binance_BTCUSDT_1m.csv \
  --is-crypto \
  --source-tz UTC \
  --time-col Time \
  --decision-tf 15min \
  --model-name MonAgentBTC \
  --mode single \
  --use-bias-filter \
  --timesteps 300000 \
  --n-envs 4
```

### Ce qui s'affiche pendant l'entraînement
À chaque intervalle d'évaluation (`eval_freq`), une ligne de progression est imprimée :
```text
[  100,000]  train=  +18.50/dd 4.2%  val=  +12.30/dd 5.1%  score=   +7.20  ← BEST
[  150,000]  train=  +22.10/dd 3.8%  val=  +15.40/dd 4.5%  score=  +10.90  ← BEST
[  200,000]  train=  +35.00/dd 3.1%  val=   -2.10/dd 9.2%  score=   -2.10  (ineligible)
```
- **`train`** : Récompense moyenne obtenue sur la période d'entraînement et son drawdown (`dd`).
- **`val`** : Récompense obtenue sur la période de validation hors-échantillon (*Out-Of-Sample*).
- **`score`** : Score composite pénalisé par le drawdown : $\min(Q_{train}, Q_{val})$.
- **`← BEST`** : Indique qu'un nouveau meilleur checkpoint a été validé et enregistré dans `models/best_model/`.
- **`(ineligible)`** : Le modèle a enregistré une performance négative sur l'une des périodes ou n'a pas pris assez de trades ; le système l'ignore pour protéger la sélection finale.

### Fichiers produits à la fin
- Modèle sauvegardé avec nom et date : `models/MonAgentBTC_BTCUSDT_YYYYMMDD_HHMMSS_15min_...zip`
- Normaliseur de statistiques d'état : `..._vecnorm.pkl`
- Meilleur checkpoint validé : `models/best_model/best_model.zip`
- Historique des évaluations : `models/eval_logs/consistency_evals.csv`
- Métadonnées du run : `models/run_info.json`

---

## Étape 4 : Visualisation de la courbe d'apprentissage

Pour afficher et exporter la courbe d'apprentissage interactive Plotly :

### Commande
```bash
python plot_learning_curve.py
```
*(Sur un serveur sans interface graphique, utilisez `--no-browser -o outputs/courbe.html`)*.

### Comment lire le graphique interactif
1. **Courbe bleue (`Train eval reward`)** : Progression des récompenses d'entraînement. Une pente ascendante montre que l'agent apprend à exploiter les features de prix et de volume.
2. **Courbe orange (`Validation reward`)** : Performance sur des données non vues durant le gradient.
   - Si elle progresse avec la courbe bleue : **l'agent généralise correctement**.
   - Si la courbe bleue monte fortement tandis que la courbe orange devient négative : **l'agent fait du surapprentissage (overfitting)**.
3. **Lignes en pointillés (`Drawdown %`)** : Pourcentage de perte maximale encourue. Un niveau bas et stable (< 10-15%) est gage de régularité.
4. **Ligne verte verticale (`Best Checkpoint`)** : Marque le timestep exact du meilleur modèle sélectionné (celui enregistré dans `models/best_model/`).

---

## Étape 5 : Diagnostics complets d'entraînement

Pour générer un rapport chiffré approfondi sur l'apprentissage, les corrélations de features et la stabilité :

### Commande
```bash
python training_diagnostics.py \
  --symbol BTC/USDT \
  --csv data/binance_BTCUSDT_1m.csv \
  --is-crypto \
  --source-tz UTC \
  --time-col Time
```

### Ce qu'il faut vérifier
- **Section [1] LEARNING CURVE** :
  ```text
  Validation reward trend: +2.10 -> +18.40 (Δ: +16.30)
  Is learning: YES
  ```
- **Section [2] FEATURE QUALITY** : Vérifie la stationnarité et la corrélation des 30 features (notamment les signaux d'anomalies de volume baleines).
- **Section [3] OVERFITTING DIAGNOSTICS** : Compare le Profit Factor et le ratio de Sharpe entre Train et Validation. Le message `✓ No severe overfitting flags triggered` valide la stabilité du modèle.

---

## Étape 6 : Visualisation des trades et chandeliers

Pour examiner graphiquement les positions prises, les prix d'entrée, les Take-Profit (TP) et Stop-Loss (SL) :

### Commande
```bash
python view_results.py \
  --symbol BTC/USDT \
  --csv data/binance_BTCUSDT_1m.csv \
  --source-tz UTC \
  --time-col Time \
  --bars 500
```

### Ce qui est affiché
- **Triangles verts / rouges** : Entrées à l'achat (*Long*) / à la vente (*Short*).
- **Lignes horizontales pointillées** : Brackets dynamiques de TP (vert) et SL (rouge).
- **Cercles verts** : Sorties gagnantes ayant touché le Take-Profit.
- **Croix rouges** : Sorties stoppées ayant touché le Stop-Loss.

---

## Étape 7 : Évaluation systématique des biais directionnels A, B, C, D

Le script `evaluate_biases.py` évalue automatiquement et de manière causale (hors-échantillon avec frais et slippage) chaque sous-biais directionnel en heure `America/New_York` :
- **Biais A** : Bougie journalière précédente (A1 Bullish, A2 Bearish, A3 Neutral)
- **Biais B** : Bougie 6H précédente RTH (Regular Trading Hours de la veille)
- **Biais C** : Sweeps de liquidité entre 07:00 et 09:30 NY (PDH, PDL, Session High/Low précédent)
- **Biais D** : Sweeps de liquidité entre 09:30 et 10:00 NY (PDH, PDL, Session Pre-market 07:00–09:30)

### Commande
```bash
python evaluate_biases.py \
  --symbol BTC/USDT \
  --csv data/binance_BTCUSDT_1m.csv \
  --is-crypto \
  --source-tz UTC \
  --time-col Time \
  --decision-tf 15min \
  --out-dir outputs/bias_eval
```

Le script teste :
1. Chaque sous-biais individuellement.
2. Les groupements d'acheteurs et de vendeurs.
3. Les confluences (ex. 2 biais alignés requis, interdiction formelle de trader en cas de contradiction).
4. La comparaison avec et sans filtre directionnel.

---

## Grille de lecture : comment savoir si la stratégie fonctionne

| Métrique / Signal | Objectif recherché | Signe d'un problème |
| :--- | :--- | :--- |
| **Validation Reward** | Positif et croissant au fil des timesteps ($\Delta > 0$) | Négatif ou plat (politique inactive ou perdante) |
| **Profit Factor (PF)** | $> 1.15$ (idéalement $> 1.30$) | $< 1.00$ (stratégie en perte nette) |
| **Max Drawdown** | Faible et maîtrisé ($< 10-15\%$) | Pertes sévères ($> 25\%$) |
| **Volume de Trades** | Régulier ($> 30$ trades par période) | Moins de 5 trades (l'agent n'ose pas entrer) |
| **Win Rate & Ratio R** | Soit WR $> 50\%$ avec $R \ge 1.0$, soit pay-off élevé ($R > 1.5$) | Espérance mathématique négative |
