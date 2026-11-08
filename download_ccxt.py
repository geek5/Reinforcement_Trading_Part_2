"""
download_ccxt.py
----------------
Download 1-minute (or custom timeframe) OHLCV candles using CCXT and export
to CSV compatible with the RL trading pipeline.

Supports any exchange available in CCXT (Binance, OKX, Bybit, Kraken, Coinbase, etc.)
and any pair/symbol (e.g. BTC/USDT, ETH/USDT, SOL/USDT, EUR/USD, etc.).

Usage:
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
    """Create and return a configured CCXT exchange instance."""
    exchange_class = getattr(ccxt, exchange_id.lower(), None)
    if exchange_class is None:
        raise ValueError(
            f"Exchange '{exchange_id}' is not recognized by ccxt. "
            f"Available exchanges include: binance, bybit, okx, kraken, coinbase, etc."
        )
    return exchange_class({
        "enableRateLimit": rate_limit,
        "timeout": 30000,
    })


def parse_timestamp_to_ms(dt_str: Optional[str]) -> Optional[int]:
    """Parse a date/datetime string into UTC epoch milliseconds."""
    if dt_str is None:
        return None
    dt_str = dt_str.strip()
    try:
        dt = pd.to_datetime(dt_str, utc=True)
        return int(dt.timestamp() * 1000)
    except Exception as exc:
        raise ValueError(f"Could not parse datetime string: {dt_str!r}") from exc


def format_ms_to_iso(ms: int) -> str:
    """Format millisecond timestamp into ISO 8601 string."""
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
    """Download OHLCV candles paginated over a date range.

    Returns a DataFrame with columns: ['Timestamp', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    where 'Time' is ISO UTC 'YYYY-MM-DD HH:MM:SS'.
    """
    if not exchange.has.get("fetchOHLCV", False):
        raise ValueError(f"Exchange {exchange.id} does not support fetchOHLCV.")

    # Timeframe duration in milliseconds
    tf_seconds = exchange.parse_timeframe(timeframe)
    tf_ms = tf_seconds * 1000

    # If since_ms is not given, default to 30 days ago
    now_ms = exchange.milliseconds()
    if until_ms is None:
        until_ms = now_ms
    if since_ms is None:
        since_ms = until_ms - (30 * 24 * 60 * 60 * 1000)

    if since_ms >= until_ms:
        raise ValueError(f"since_ms ({since_ms}) must be earlier than until_ms ({until_ms})")

    # Estimated total candles for progress bar
    total_expected = max(1, (until_ms - since_ms) // tf_ms)

    all_ohlcv = []
    current_since = since_ms

    if verbose:
        print(f"Downloading {symbol} ({timeframe}) from {exchange.id.upper()}...")
        print(f"From : {format_ms_to_iso(since_ms)} UTC")
        print(f"To   : {format_ms_to_iso(until_ms)} UTC")
        print(f"Estimated bars: ~{total_expected:,}")

    pbar = tqdm(total=total_expected, unit="bars", disable=not verbose)

    while current_since <= until_ms:
        candles = None
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
                    print(f"\nFailed to fetch at timestamp {current_since} after {max_retries} attempts: {exc}")
                    raise
                time.sleep(retry_delay_sec * (2 ** retry))

        if not candles:
            # No more data returned by the exchange
            break

        # Filter out candles beyond until_ms
        valid_candles = [c for c in candles if c[0] <= until_ms]
        if not valid_candles:
            break

        all_ohlcv.extend(valid_candles)
        pbar.update(len(valid_candles))

        last_candle_ms = valid_candles[-1][0]
        # Next request begins at the candle immediately following the last one
        next_since = last_candle_ms + tf_ms

        if next_since <= current_since:
            # Protect against infinite loop if exchange returns same timestamp
            next_since = current_since + tf_ms

        current_since = next_since

        # If exchange returned fewer candles than requested, we might have reached the end
        if len(candles) < limit_per_request and last_candle_ms >= until_ms:
            break

    pbar.close()

    if not all_ohlcv:
        print("Warning: No candles downloaded.")
        return pd.DataFrame(columns=["Timestamp", "Time", "Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(all_ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df = df.drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # Format human-readable UTC Time column: "YYYY-MM-DD HH:MM:SS"
    df["Time"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Column order
    df = df[["Timestamp", "Time", "Open", "High", "Low", "Close", "Volume"]]

    if verbose:
        print(f"Downloaded {len(df):,} unique candles.")
        if len(df):
            print(f"Span: {df['Time'].iloc[0]} -> {df['Time'].iloc[-1]} UTC")

    return df


def save_ohlcv_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the downloaded candles DataFrame to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved dataset -> {path.resolve()}")
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
