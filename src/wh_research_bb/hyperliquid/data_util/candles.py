import logging
import time
from datetime import UTC, datetime, timedelta

import polars as pl
from hyperliquid.info import Info
from hyperliquid.utils.constants import MAINNET_API_URL
from hyperliquid.utils.error import ClientError, ServerError
from tenacity import (
    before,  # TODO change this
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

# Create Info instance
info = Info(base_url=MAINNET_API_URL, skip_ws=True)  # TODO: make info not run on import

# Candles column name map
COLUMN_NAME_MAP = {
    "T": "timestamp",
    "s": "symbol",
    "i": "interval",
    "o": "open",
    "c": "close",
    "h": "high",
    "l": "low",
    "v": "volume",
    "n": "trades",
}

# Candles dataframe schema
CANDLES_SCHEMA = pl.Schema(
    {
        "timestamp": pl.Datetime(time_zone="UTC"),
        "symbol": pl.String(),
        "interval": pl.String(),
        "open": pl.Float64(),
        "close": pl.Float64(),
        "high": pl.Float64(),
        "low": pl.Float64(),
        "volume": pl.Float64(),
        "trades": pl.Int64(),
    }
)

# Maximum number of candles API can download
MAX_CANDLES = 5000

# Wait time between API requests
WAIT = 5.5

# timedelta map
TIME_DELTA_MAP = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "8h": timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
}

# Supported intervals for candle download
TIME_INTERVALS = tuple(TIME_DELTA_MAP.keys())


def _is_retryable(exc: BaseException) -> bool:
    """
    function for determining if
    retry is warranted for API
    error
    """
    if isinstance(exc, ServerError):
        return True
    elif isinstance(exc, ClientError):
        return exc.status_code == 429
    else:
        return False


def _get_perp_universe_list() -> list[str]:
    """
    Gets list of all symbols in hyperliquid perp
    universe, both listed and unlisted
    """
    symbols = []
    universe_obj = info.meta()["universe"]
    for _ in universe_obj:
        symbols.append(_["name"])
    return symbols


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(7),
    wait=wait_exponential_jitter(5, 60),
    reraise=True,
)
def _download_candles_df(
    symbol: str, candle_interval: str, start: int, end: int, now: datetime
) -> pl.DataFrame:
    """
    download candles from HL api
    from specified period into df
    """
    if candle_interval not in TIME_INTERVALS:
        raise ValueError("Chosen time interval is not supported")

    candles = info.candles_snapshot(
        name=symbol, interval=candle_interval, startTime=start, endTime=end
    )

    if not isinstance(candles, list):
        raise TypeError(f"Unexpected candles response for {symbol}: {candles!r}")

    if not candles:
        return pl.DataFrame(schema=CANDLES_SCHEMA)
    df = pl.DataFrame(candles)
    df = _format_raw_candles_df(df, now)
    return df


def _format_raw_candles_df(df: pl.DataFrame, now: datetime) -> pl.DataFrame:
    """
    converts raw pl.DataFrame(candles)
    into a more useful format
    """
    df = df.drop("t")
    df = df.rename(COLUMN_NAME_MAP)
    df = df.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.replace_time_zone(
            time_zone="UTC"
        )
    )
    df = df.with_columns(pl.col("timestamp") + timedelta(milliseconds=1))
    df = df.filter(pl.col("timestamp") <= now)
    df = (
        df.select(list(CANDLES_SCHEMA)).cast(CANDLES_SCHEMA).sort("timestamp")
    )  # TODO: check logic on list(CANDLES_SCHEMA)
    return df


def download_list_candles_max(  # TODO: remove symbols arg, pass _get_perp_universe_list instead
    symbols: list[str], candle_interval: str
) -> tuple[pl.DataFrame, list[str]]:
    """
    downloads a long panel dataframe
    of all symbols in the perp universe
    """
    if candle_interval not in TIME_INTERVALS:
        raise ValueError("Chosen time interval is not supported")

    time_delta = TIME_DELTA_MAP[candle_interval]
    now = datetime.now(UTC)
    start = now - (time_delta * MAX_CANDLES)
    frames, failed_symbols = [], []

    for i, symbol in enumerate(symbols):
        try:
            frames.append(
                _download_candles_df(
                    symbol=symbol,
                    candle_interval=candle_interval,
                    start=int(start.timestamp() * 1000),
                    end=int(now.timestamp() * 1000),
                    now=now,
                )
            )
            logger.info(
                f"{symbol} {candle_interval} downloaded (symbol {i + 1} / {len(symbols)})"
            )
        except (
            ClientError,
            ServerError,
            KeyError,
            TypeError,
        ) as e:  # TODO: consider other exc such as connection error & timeout
            logger.warning(
                f"Download for symbol: {symbol} failed, Error: {e} (symbol {i + 1} / {len(symbols)})"
            )
            failed_symbols.append(symbol)
        time.sleep(
            WAIT
        )  # TODO: add logic to prevent sleep after final symbol is downlaoded
    if frames:
        df = pl.concat(frames).sort(["symbol", "timestamp"])
        return (df, failed_symbols)
    else:
        df = pl.DataFrame(schema=CANDLES_SCHEMA)
        return (df, failed_symbols)
