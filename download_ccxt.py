"""
download_ccxt.py
----------------
Script de téléchargement de bougies OHLCV 1m (ou autre timeframe) via la bibliothèque CCXT.
Exporte les données au format CSV directement exploitable par le pipeline RL (Gymnasium / SB3).

Compatible avec tous les exchanges gérés par CCXT (Binance, Bybit, OKX, Kraken, Coinbase, etc.)
et n'importe quelle paire / instrument (ex. BTC/USDT, ETH/USDT, SOL/USDT, EUR/USD, etc.).

Exemples d'utilisation :
    python download_ccxt.py --symbol BTC/USDT --exchange binance --since 2023-01-01 --until 2024-01-01
    python download_ccxt.py --symbol ETH/USDT --exchange bybit --days 90
    python download_ccxt.py --symbol SOL/USDT --output data/SOLUSDT_1m.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

try:
    import ccxt
except ImportError as exc:
    raise ImportError("ccxt is required. Install it with: pip install ccxt") from exc


def get_exchange_instance(exchange_id: str, rate_limit: bool = True) -> ccxt.Exchange:
    """Initialise et configure l'instance de l'exchange CCXT demandé."""
    exchange_class = getattr(ccxt, exchange_id.lower(), None)
    if exchange_class is None:
        raise ValueError(
            f"L'exchange '{exchange_id}' n'est pas reconnu par ccxt. "
            f"Exchanges courants : binance, bybit, okx, kraken, coinbase, etc."
        )
    # Activation du rate limiting pour éviter d'être banni par l'API de l'exchange
    return exchange_class({
        "enableRateLimit": rate_limit,
        "timeout": 30000,
    })


def parse_timestamp_to_ms(dt_str: Optional[str]) -> Optional[int]:
    """Convertit une chaîne de date/heure (ex: '2024-01-01') en millisecondes epoch UTC."""
    if dt_str is None:
        return None
    dt_str = dt_str.strip()
    try:
        dt = pd.to_datetime(dt_str, utc=True)
        return int(dt.timestamp() * 1000)
    except Exception as exc:
        raise ValueError(f"Impossible de parser la date : {dt_str!r}") from exc


def format_ms_to_iso(ms: int) -> str:
    """Formate un timestamp en millisecondes en chaîne ISO UTC lisible."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def download_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = "1m",
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
    limit_per_request: int = 1000,
    max_retries: int = 5,
    retry_delay_sec: float = 2.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Télécharge les bougies OHLCV par pagination sur une plage temporelle donnée.

    Retourne un DataFrame avec les colonnes :
    ['Timestamp', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    où 'Time' est une chaîne UTC formatée : 'YYYY-MM-DD HH:MM:SS'.
    """
    if not exchange.has.get("fetchOHLCV", False):
        raise ValueError(f"L'exchange {exchange.id} ne supporte pas l'endpoint fetchOHLCV.")

    # Durée de chaque bougie en millisecondes
    tf_seconds = exchange.parse_timeframe(timeframe)
    tf_ms = tf_seconds * 1000

    # Si 'since_ms' n'est pas renseigné, par défaut on prend 30 jours dans le passé
    now_ms = exchange.milliseconds()
    if until_ms is None:
        until_ms = now_ms
    if since_ms is None:
        since_ms = until_ms - (30 * 24 * 60 * 60 * 1000)

    if since_ms >= until_ms:
        raise ValueError(f"since_ms ({since_ms}) doit être strictement antérieur à until_ms ({until_ms})")

    # Estimation du nombre total de bougies pour la barre de progression
    total_expected = max(1, (until_ms - since_ms) // tf_ms)

    all_ohlcv = []
    current_since = since_ms

    if verbose:
        print(f"Téléchargement de {symbol} ({timeframe}) sur {exchange.id.upper()}...")
        print(f"Du   : {format_ms_to_iso(since_ms)} UTC")
        print(f"Au   : {format_ms_to_iso(until_ms)} UTC")
        print(f"Bougies estimées : ~{total_expected:,}")

    # Barre de progression interactive tqdm
    pbar = tqdm(total=total_expected, unit="bars", disable=not verbose)

    # Boucle de pagination tant qu'on n'a pas atteint la date de fin
    while current_since <= until_ms:
        candles = None
        # Mécanisme de retry exponentiel en cas de déconnexion réseau ou rate limit
        for retry in range(max_retries):
            try:
                candles = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=limit_per_request,
                )
                break
            except (ccxt.NetworkError, ccxt.RateLimitExceeded, ccxt.ExchangeError) as exc:
                if retry == max_retries - 1:
                    print(f"\nÉchec de récupération au timestamp {current_since} après {max_retries} essais : {exc}")
                    raise
                time.sleep(retry_delay_sec * (2 ** retry))

        if not candles:
            # L'exchange ne renvoie plus de bougies (fin de l'historique disponible)
            break

        # On ignore les bougies qui dépassent la date de fin souhaitée
        valid_candles = [c for c in candles if c[0] <= until_ms]
        if not valid_candles:
            break

        all_ohlcv.extend(valid_candles)
        pbar.update(len(valid_candles))

        last_candle_ms = valid_candles[-1][0]
        # La prochaine requête débute juste après la dernière bougie reçue
        next_since = last_candle_ms + tf_ms

        if next_since <= current_since:
            # Sécurité anti-boucle infinie si l'exchange renvoie les mêmes timestamps
            next_since = current_since + tf_ms

        current_since = next_since

        # Si l'exchange a retourné moins de bougies que la limite et qu'on a dépassé until_ms
        if len(candles) < limit_per_request and last_candle_ms >= until_ms:
            break

    pbar.close()

    if not all_ohlcv:
        print("Attention : Aucune bougie téléchargée.")
        return pd.DataFrame(columns=["Timestamp", "Time", "Open", "High", "Low", "Close", "Volume"])

    # Construction du DataFrame pandas et déduplication
    df = pd.DataFrame(all_ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df = df.drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # Création de la colonne de date lisible UTC "YYYY-MM-DD HH:MM:SS"
    df["Time"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Ordre standard des colonnes
    df = df[["Timestamp", "Time", "Open", "High", "Low", "Close", "Volume"]]

    if verbose:
        print(f"Téléchargement terminé : {len(df):,} bougies uniques.")
        if len(df):
            print(f"Période couverte : {df['Time'].iloc[0]} -> {df['Time'].iloc[-1]} UTC")

    return df


def save_ohlcv_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Enregistre le DataFrame de bougies sous forme de fichier CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Fichier sauvegardé -> {path.resolve()}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Download 1m OHLCV candles via CCXT for custom instruments."
    )
    parser.add_argument(
        "--symbol", "-s", default="BTC/USDT",
        help="Trading symbol / pair, e.g. BTC/USDT, ETH/USDT, EUR/USD (default: BTC/USDT)",
    )
    parser.add_argument(
        "--exchange", "-e", default="binance",
        help="CCXT exchange name, e.g. binance, bybit, okx, kraken (default: binance)",
    )
    parser.add_argument(
        "--timeframe", "-t", default="1m",
        help="Candle timeframe, e.g. 1m, 5m, 1h (default: 1m)",
    )
    parser.add_argument(
        "--since", default=None,
        help="Start date UTC (e.g. 2023-01-01 or 2023-01-01 00:00:00)",
    )
    parser.add_argument(
        "--until", default=None,
        help="End date UTC (e.g. 2024-01-01 or 'now')",
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Number of past days to download up to now (convenience shortcut)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path (default: data/{exchange}_{symbol_clean}_{timeframe}.csv)",
    )
    parser.add_argument(
        "--limit", type=int, default=1000,
        help="Limit of candles per request (default: 1000)",
    )

    args = parser.parse_args()

    exchange = get_exchange_instance(args.exchange)

    now_ms = exchange.milliseconds()
    until_ms = parse_timestamp_to_ms(args.until) if (args.until and args.until.lower() != "now") else now_ms

    if args.days is not None:
        since_ms = until_ms - (args.days * 24 * 60 * 60 * 1000)
    else:
        since_ms = parse_timestamp_to_ms(args.since)

    # Output filename
    if args.output:
        out_path = Path(args.output)
    else:
        clean_symbol = args.symbol.replace("/", "").replace(":", "-").upper()
        out_path = Path("data") / f"{args.exchange.lower()}_{clean_symbol}_{args.timeframe}.csv"

    df = download_ohlcv(
        exchange=exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        since_ms=since_ms,
        until_ms=until_ms,
        limit_per_request=args.limit,
    )

    if not df.empty:
        save_ohlcv_csv(df, out_path)
    else:
        print("No data retrieved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
