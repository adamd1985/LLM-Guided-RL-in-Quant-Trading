from __future__ import annotations

import csv
import io
import logging
import math
import pickle  # noqa: S403  # nosec: trusted internal artefacts
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Literal

import backoff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype
import polars as pl
import statsmodels.api as sm
import yaml
from pydantic import BaseModel, Field

try:
    import html2text
except ImportError:  # pragma: no cover - dependency is optional for tests
    html2text = None  # type: ignore[assignment]
from tqdm.auto import tqdm
try:
    pd.set_option("future.no_silent_downcasting", True)
except KeyError:
    pass

try:
    from rl_agent_utils import PerformanceEstimator
except ImportError:  # pragma: no cover - optional dependency for RL evaluation
    PerformanceEstimator = None  # type: ignore[assignment]

# =============================================================================
# Constants
# =============================================================================

MIN_CONFIDENCE = 0
MAX_CONFIDENCE = 3


logger = logging.getLogger(__name__)


class NewsCleaningResponse(BaseModel):
    """Validated response payload for cleaned news factors."""

    news_factors: list[str]


class Action(str, Enum):
    """Supported trade directions."""

    LONG = "LONG"
    SHORT = "SHORT"


class Correction(BaseModel):
    """Structured correction returned by downstream evaluation models."""

    correction_type: Literal[
        "feature_weight",
        "signal_conflict",
        "scenario_omission",
        "methodology_flaw",
    ]
    target: str
    suggestion: str


class Evaluation(BaseModel):
    """Evaluation artefact produced by the judge model."""

    action: Action
    long_conf_score: int = Field(..., ge=MIN_CONFIDENCE, le=MAX_CONFIDENCE)
    short_conf_score: int = Field(..., ge=MIN_CONFIDENCE, le=MAX_CONFIDENCE)
    evaluation_score: int = Field(..., ge=MIN_CONFIDENCE, le=MAX_CONFIDENCE)
    explanation: str
    corrections: list[Correction] | None = None


class TradeStrategy(BaseModel):
    """Structured trade recommendation returned by the strategy model."""

    action: Action
    action_confidence: int = Field(..., ge=MIN_CONFIDENCE, le=MAX_CONFIDENCE)
    explanation: str
    features: list[str]


def enum_to_str_representer(dumper: yaml.Dumper, data: Enum) -> yaml.Node:
    """Represent enums as plain strings when serialising to YAML."""

    return dumper.represent_str(data.value)


yaml.add_representer(Action, enum_to_str_representer)
RATES_INDEX = "^TNX"
VOLATILITY_INDEX = "^VIX"
SMALLCAP_INDEX = "^RUT"
GOLD_FUTURES = "GC=F"
OIL_FUTURES = "CL=F"
MARKET = "^SPX"
SECTOR_INDEX = "^IXIC"
HV_THRESHOLD = 0.05

ADDITIONAL_FIN_FEATURES = [RATES_INDEX, VOLATILITY_INDEX, SMALLCAP_INDEX, GOLD_FUTURES, OIL_FUTURES, MARKET, SECTOR_INDEX]

TICKER_COMPANY_NAME_MAP = {
    "DIA": "Dow Jones Industrial Average ETF",
    "SPY": "S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust (NASDAQ 100 ETF)",
    "EZU": "FTSE 100 ETF",
    "EWJ": "iShares MSCI Japan ETF (Nikkei 225)",
    "GOOGL": "Alphabet Inc. (Google)",
    "AAPL": "Apple Inc.",
    "META": "Meta Platforms Inc. (Facebook)",
    "AMZN": "Amazon.com Inc.",
    "MSFT": "Microsoft Corporation",
    "TWTR": "Twitter Inc.",
    "NOK": "Nokia Corporation",
    "PHIA.AS": "Koninklijke Philips N.V.",
    "SIE.DE": "Siemens AG",
    "BIDU": "Baidu Inc.",
    "BABA": "Alibaba Group Holding Limited",
    "0700.HK": "Tencent Holdings Limited",
    "6758.T": "Sony Group Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "HSBC": "HSBC Holdings plc",
    "0939.HK": "China Construction Bank Corporation",
    "XOM": "Exxon Mobil Corporation",
    "RDSA.AS": "Royal Dutch Shell plc (Shell)",
    "PTR": "PetroChina Company Limited",
    "TSLA": "Tesla Inc.",
    "VWAGY": "Volkswagen AG",
    "TM": "Toyota Motor Corporation",
    "KO": "The Coca-Cola Company",
    "ABI.BR": "Anheuser-Busch InBev SA/NV",
    "2503.T": "Kirin Holdings Company Limited",
}

HIGH_RISK_PROFILE = "HIGH RISK"
HIGH_OBJECTIVES = "Develop a 30-day trading strategy based on volatility, events, and price action."
LOW_RISK_PROFILE = "LOW RISK"
LOW_OBJECTIVES = "Develop a low-risk trading strategy with controlled volatility and minimal drawdowns, designed for a 30-90 day horizon."

PERSONA = "You are an equities trader with a strong quantitative background."
CLASSIFICATION = "High-Growth Tech Stock"

GPT_VOCAB_SIZE = 50_257  # 50_128 or 100_256


DataFrame = pd.DataFrame
Series = pd.Series

# =============================================================================
# Utility Functions
# =============================================================================


def simple_moving_average(series: Series, window: int) -> Series:
    """Calculate the simple moving average with a fixed lookback window."""

    if window <= 0:
        raise ValueError("window must be positive")
    return series.rolling(window=window, min_periods=window).mean()


def exponential_moving_average(series: Series, span: int) -> Series:
    """Return the exponential moving average using Welles Wilder style smoothing."""

    if span <= 0:
        raise ValueError("span must be positive")
    return series.ewm(span=span, adjust=False).mean()


def relative_strength_index(series: Series, period: int = 14) -> Series:
    """Compute the Relative Strength Index (RSI)."""

    if period <= 0:
        raise ValueError("period must be positive")

    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0.0)


def average_true_range(high: Series, low: Series, close: Series, period: int = 14) -> Series:
    """Calculate the Average True Range (ATR)."""

    if period <= 0:
        raise ValueError("period must be positive")

    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def macd(series: Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple[Series, Series, Series]:
    """Return MACD, signal line, and histogram using exponential moving averages."""

    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("MACD periods must be positive")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be smaller than slow_period")

    ema_fast = exponential_moving_average(series, span=fast_period)
    ema_slow = exponential_moving_average(series, span=slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: Series, window: int, num_std: float = 2.0) -> tuple[Series, Series, Series]:
    """Compute Bollinger Bands (upper, middle, lower)."""

    if window <= 0:
        raise ValueError("window must be positive")
    if num_std <= 0:
        raise ValueError("num_std must be positive")

    middle_band = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper_band = middle_band + num_std * rolling_std
    lower_band = middle_band - num_std * rolling_std
    return upper_band, middle_band, lower_band


def safe_pickle_load(path: Path | str, *, trusted_source: bool = True) -> Any:
    """Safely load pickle artefacts generated within the project.

    Parameters
    ----------
    path:
        Location of the pickle file.
    trusted_source:
        Guard flag ensuring the caller explicitly acknowledges that the pickle
        data originates from a trusted pipeline. Set to ``False`` to raise.

    Returns
    -------
    Any
        The deserialised Python object.

    Raises
    ------
    ValueError
        If ``trusted_source`` is ``False``.
    FileNotFoundError
        When the path does not exist on disk.
    RuntimeError
        If the file cannot be deserialised even after fallback handling.
    """

    if not trusted_source:
        raise ValueError("Refusing to load pickle data from an untrusted source.")

    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    if file_path.suffix not in {".pkl", ".pickle"}:
        raise ValueError(f"Expected a pickle file, received: {file_path.suffix}")

    try:
        with file_path.open("rb") as file:
            return pickle.load(file)  # noqa: S301  # nosec: trusted internal artefacts
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning(
            "Standard pickle load failed for %s: %s. Falling back to pandas handling.",
            file_path,
            exc,
        )
        try:
            with file_path.open("rb") as file:
                raw_bytes = file.read()
            return pd.read_pickle(io.BytesIO(raw_bytes), compression=None)  # noqa: S301  # nosec: trusted internal artefacts
        except Exception as fallback_exc:  # pragma: no cover - defensive path
            raise RuntimeError(f"Unable to deserialize pickle artefact at {file_path}.") from fallback_exc


def get_fundamentals(target: str, fundamentals_dir: Path | str) -> DataFrame:
    """Load and enhance quarterly fundamentals with QoQ and YoY growth factors.

    Parameters
    ----------
    target:
        Equity ticker to load.
    fundamentals_dir:
        Directory containing ``*-aggregated_fundamentals.csv`` artefacts.

    Returns
    -------
    pandas.DataFrame
        Time-indexed fundamentals with additional growth columns. All timestamps
        are normalised to UTC midnight.
    """

    fundamentals_path = Path(fundamentals_dir).expanduser().resolve()
    fundamentals_file = fundamentals_path / f"{target}-aggregated_fundamentals.csv"
    if not fundamentals_file.exists():
        raise FileNotFoundError(f"No fundamentals file found at {fundamentals_file}.")

    logger.info("Loading fundamentals from %s", fundamentals_file)
    fundamentals = pd.read_csv(fundamentals_file, parse_dates=["Date"])
    fundamentals["Date"] = pd.to_datetime(fundamentals["Date"], utc=True).dt.normalize()

    fundamentals = fundamentals.sort_values("Date").set_index("Date").replace([np.inf, -np.inf], np.nan)
    quarterly_features = [
        "Quick Ratio",
        "Current Ratio",
        "Debt to Equity Ratio",
        "Gross Margin",
        "Operating Margin",
        "EBIT Margin",
        "Net Profit Margin",
        "Asset Turnover",
        "Inventory Turnover Ratio",
        "Price to Book Ratio",
        "PE Ratio",
        "EPS",
        "Revenue",
        "EBIT",
        "Net Income",
        "Free Cash Flow Per Share",
        "Operating Cash Flow Per Share",
        "Return on Equity",
        "Return on Assets",
        "Earnings Yield",
    ]

    for col in quarterly_features:
        if col not in fundamentals.columns:
            continue

        fundamentals[f"{col}_QoQ_Growth"] = fundamentals[col].pct_change(periods=1).replace([np.inf, -np.inf], np.nan)

        no_change_mask = fundamentals[col] == fundamentals[col].shift(1)
        fundamentals.loc[no_change_mask, f"{col}_QoQ_Growth"] = np.nan

        fundamentals[f"{col}_YoY_Growth"] = fundamentals[col].pct_change(periods=4).replace([np.inf, -np.inf], np.nan)

        fundamentals[f"{col}_QoQ_Growth"] = fundamentals[f"{col}_QoQ_Growth"].ffill()
        fundamentals[f"{col}_YoY_Growth"] = fundamentals[f"{col}_YoY_Growth"].ffill()

    return fundamentals.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def engineer_optionfeatures(
    options_data: pl.DataFrame,
    skew_threshold: float = 0.015,
) -> DataFrame:
    """Generate volatility skew features across moneyness buckets.

    Parameters
    ----------
    options_data:
        Polars DataFrame containing option chains enriched with ``moneyness`` and
        implied volatility columns.
    skew_threshold:
        Minimum absolute skew difference required to flag structural regimes.

    Returns
    -------
    pandas.DataFrame
        Daily skew summary indexed by date.
    """

    if not isinstance(options_data, pl.DataFrame):
        raise TypeError("options_data must be a polars.DataFrame instance")

    if skew_threshold <= 0:
        raise ValueError("skew_threshold must be positive")

    # Define ranges for OTM, ATM, and ITM
    otm_calls = options_data.filter((pl.col("call_put") == "C") & (pl.col("moneyness") >= 5.0))
    atm_calls = options_data.filter((pl.col("call_put") == "C") & (pl.col("moneyness") > -1) & (pl.col("moneyness") <= 1))
    itm_calls = options_data.filter((pl.col("call_put") == "C") & (pl.col("moneyness") > -1) & (pl.col("moneyness") < 5))
    otm_puts = options_data.filter((pl.col("call_put") == "P") & (pl.col("moneyness") >= 5.0))
    atm_puts = options_data.filter((pl.col("call_put") == "P") & (pl.col("moneyness") > -1) & (pl.col("moneyness") <= 1))
    itm_puts = options_data.filter((pl.col("call_put") == "P") & (pl.col("moneyness") > -1) & (pl.col("moneyness") < 5))

    # OTM Skew computations
    otm_combined = (
        otm_calls.group_by("t_date")
        .agg([pl.last("iv").alias("OTM_IV_Call"), pl.last("expiration_date")])
        .join(otm_puts.group_by("t_date").agg([pl.last("iv").alias("OTM_IV_Put"), pl.last("expiration_date")]), on="t_date", how="inner")
        .with_columns(((pl.col("OTM_IV_Put") - pl.col("OTM_IV_Call")) / pl.col("OTM_IV_Call")).alias("OTM_Skew"))
    ).drop("expiration_date_right")

    # ATM Skew computations
    atm_combined = (
        atm_calls.group_by("t_date")
        .agg([pl.last("iv").alias("ATM_IV_Call"), pl.last("expiration_date")])
        .join(atm_puts.group_by("t_date").agg([pl.last("iv").alias("ATM_IV_Put"), pl.last("expiration_date")]), on="t_date", how="inner")
        .with_columns(((pl.col("ATM_IV_Put") - pl.col("ATM_IV_Call")) / pl.col("ATM_IV_Call")).alias("ATM_Skew"))
    ).drop("expiration_date_right")

    # ITM Skew computations
    itm_combined = (
        itm_calls.group_by("t_date")
        .agg([pl.last("iv").alias("ITM_IV_Call"), pl.last("expiration_date")])
        .join(itm_puts.group_by("t_date").agg([pl.last("iv").alias("ITM_IV_Put"), pl.last("expiration_date")]), on="t_date", how="inner")
        .with_columns(((pl.col("ITM_IV_Put") - pl.col("ITM_IV_Call")) / pl.col("ITM_IV_Call")).alias("ITM_Skew"))
    ).drop("expiration_date_right")

    # Fallback logic for ATM and ITM
    if atm_combined.is_empty():
        atm_combined = otm_combined.select(
            [
                pl.col("OTM_IV_Call").alias("ATM_IV_Call"),
                pl.col("OTM_IV_Put").alias("ATM_IV_Put"),
                pl.col("OTM_Skew").alias("ATM_Skew"),
                pl.col("t_date"),
                pl.col("expiration_date"),
            ]
        )

    if itm_combined.is_empty():
        itm_combined = otm_combined.select(
            [
                pl.col("OTM_IV_Call").alias("ITM_IV_Call"),
                pl.col("OTM_IV_Put").alias("ITM_IV_Put"),
                pl.col("OTM_Skew").alias("ITM_Skew"),
                pl.col("t_date"),
                pl.col("expiration_date"),
            ]
        )

    # Combine all skews
    combined_skew = otm_combined.join(atm_combined, on="t_date", how="outer").drop(["expiration_date_right", "t_date_right"])
    combined_skew = combined_skew.join(itm_combined, on="t_date", how="outer").drop(["expiration_date_right", "t_date_right"])

    # Add Skew column
    combined_skew = combined_skew.with_columns((pl.col("ATM_IV_Put") / pl.col("ATM_IV_Call")).alias("Skew"))

    # Add Vol_Surface column
    combined_skew = combined_skew.with_columns(
        pl.when(((pl.col("OTM_Skew") - pl.col("ATM_Skew")).abs() > skew_threshold) & (pl.col("ATM_Skew") < pl.col("OTM_Skew")))
        .then(pl.lit("SMILE"))
        .when(
            ((pl.col("OTM_Skew") - pl.col("ATM_Skew")).abs() > skew_threshold)
            & (pl.col("ATM_Skew") < pl.col("OTM_Skew"))
            & (pl.col("ATM_IV_Put") > pl.col("ATM_IV_Call"))
        )
        .then(pl.lit("PUT_SKEW"))
        .when(
            ((pl.col("OTM_Skew") - pl.col("ATM_Skew")).abs() > skew_threshold)
            & (pl.col("ATM_Skew") < pl.col("OTM_Skew"))
            & (pl.col("ATM_IV_Call") > pl.col("ATM_IV_Put"))
        )
        .then(pl.lit("CALL_SKEW"))
        .otherwise(pl.lit("FLAT"))
        .alias("Vol_Surface")
        .cast(pl.Utf8)
    )

    # Calculate DTE (Days to Expiration)
    combined_skew = combined_skew.with_columns(
        (pl.col("expiration_date").str.strptime(pl.Date, "%Y-%m-%d") - pl.col("t_date")).dt.total_days().alias("DTE")
    )

    # Final adjustments
    combined_skew = (
        combined_skew.with_columns(
            [
                pl.col("t_date").alias("Date"),
            ]
        )
        .drop_nulls()
        .drop("t_date")
    )

    return combined_skew.sort("Date").to_pandas()


def get_historic_optionschain_summary(target: str, options_dir: Path | str) -> DataFrame:
    """Summarise historical option chain skews for the specified ticker."""

    options_path = Path(options_dir).expanduser().resolve()
    candidates = sorted(options_path.glob(f"{target}*options_chain.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No option chain parquet files found for target '{target}' in {options_path}.")

    selected_file = candidates[0]
    options_df = pl.read_parquet(selected_file).unique().sort(["t_date", "price_strike", "call_put"]).drop("comment", strict=False)
    options_df = options_df.with_columns(pl.col("t_date").str.strptime(pl.Date, format="%Y-%m-%d").alias("t_date"))
    options_df = options_df.with_columns(
        ((pl.col("price_strike") - pl.col("start_date_forward_price")) / pl.col("start_date_forward_price") * 100)
        .round(0)
        .cast(pl.Float64)
        .alias("moneyness")
    )

    optionchain_data = engineer_optionfeatures(options_df.sort("t_date"))
    optionchain_data["Date"] = pd.to_datetime(optionchain_data["Date"], utc=True).dt.normalize()
    return optionchain_data.set_index("Date")


def get_economic_indicators(directory_path: Path | str) -> DataFrame:
    """Load and engineer economic indicator time series from CSV snapshots."""

    directory = Path(directory_path).expanduser()
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {directory}")

    processed_frames: dict[str, DataFrame] = {}

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        data_name = re.sub(r"_\d{4}_\d{4}$", "", file_path.stem)

        if "date" in df.columns:
            df["Date"] = pd.to_datetime(df["date"], utc=True)
        elif "release_date" in df.columns:
            df["Date"] = pd.to_datetime(df["release_date"], utc=True)
        elif {"year", "periodName"}.issubset(df.columns):
            df["Date"] = pd.to_datetime(df["year"].astype(str) + " " + df["periodName"], format="%Y %B", utc=True)
        else:
            raise ValueError(f"Unrecognized date format in file: {file_path}")

        df["Date"] = pd.to_datetime(df["Date"].dt.to_period("M").dt.to_timestamp(), utc=True).dt.normalize()
        df = df.set_index("Date")
        df = df[~df.index.duplicated(keep="last")]

        column_name = data_name
        if "value" in df.columns:
            column_name = data_name
            df = df.rename(columns={"value": column_name})
            df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
        elif "actual" in df.columns:
            df = df.rename(columns={"actual": column_name})

        df = df[[column_name]].dropna()
        processed_frames[data_name] = df

    merged_df = pd.concat(processed_frames.values(), axis=1).sort_index()
    yoy_features = merged_df.pct_change(periods=12).add_suffix("_YoY")
    qoq_features = merged_df.pct_change(periods=3).add_suffix("_QoQ")
    final_df = pd.concat([merged_df, yoy_features, qoq_features], axis=1).bfill().ffill()

    return final_df


def get_historic(target: str, historic_dir: Path | str) -> DataFrame:
    """Load historical OHLC data exported from IBKR."""

    historic_path = Path(historic_dir).expanduser().resolve()
    data_file = historic_path / f"{target}-STK-3652-20210101-1 day.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Historical data file not found: {data_file}")

    df = pd.read_csv(data_file)
    df["Ticker"] = target
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.normalize()
    return df.sort_values("Date").set_index("Date").dropna()


def get_macro_data(
    macro_path: Path | str,
    selected_tickers: Sequence[str] = ("SPX", "NDX", "VIX", "TNX", "IRX"),
) -> DataFrame:
    """Load macro benchmarks (index levels, rates, vol) for correlation features."""

    base_path = Path(macro_path).expanduser().resolve()
    frames: list[DataFrame] = []

    for ticker in selected_tickers:
        matches = sorted(base_path.glob(f"{ticker}-*.csv"))
        if not matches:
            logger.warning("No macro data found for ticker: %s", ticker)
            continue

        df = pd.read_csv(matches[0])
        df = df.rename(columns=lambda col: f"{ticker}_{col}" if col != "Date" else col)
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.normalize()
        frames.append(df.sort_values("Date"))

    if not frames:
        raise FileNotFoundError(f"No macro data loaded from {base_path}")

    macro_data = frames[0]
    for frame in frames[1:]:
        macro_data = macro_data.merge(frame, on="Date", how="outer")

    macro_data = macro_data.sort_values("Date").reset_index(drop=True).dropna()
    macro_data.set_index("Date", inplace=True)
    return macro_data


def calculate_ma_slope(series: Series) -> float:
    """Compute the slope of a moving average over the last two observations."""

    return float((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]) if len(series) > 1 else float("nan")


def calculate_trend_slope(series: Series) -> float:
    """Estimate the linear trend slope for the provided series."""

    x = np.arange(len(series))
    return float(np.polyfit(x, series, 1)[0]) if len(series) > 1 else float("nan")


def feature_engineer(
    df: DataFrame,
    market_ticker: str = "SPX",
    roll_window: int = 5 * 4,
) -> DataFrame:
    """Utility to get a DF with all features for a weekly strategy."""

    df.reset_index(inplace=True, drop=False)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.normalize()

    # Technical Indicators
    df["20MA"] = simple_moving_average(df["Close"], window=20)
    df["50MA"] = simple_moving_average(df["Close"], window=50)
    df["100MA"] = simple_moving_average(df["Close"], window=100)
    df["200MA"] = simple_moving_average(df["Close"], window=200)
    df["RSI"] = relative_strength_index(df["Close"], period=14)
    df["ATR"] = average_true_range(df["High"], df["Low"], df["Close"], period=14)
    df["MACD"], df["Signal_Line"], _ = macd(df["Close"], fast_period=12, slow_period=26, signal_period=9)

    # Slope of Moving Averages and Z-Scores
    for ma in ["20MA", "50MA", "100MA", "200MA"]:
        df[f"{ma}_Slope"] = df[ma].rolling(window=roll_window).apply(calculate_trend_slope, raw=False)
        rolling_mean = df[ma].rolling(window=roll_window).mean()
        rolling_std = df[ma].rolling(window=roll_window).std()
        df[f"{ma}_Z"] = ((df[ma] - rolling_mean) / rolling_std).bfill()

    # Rolling Features for Daily and Macro Data
    daily_features = ["Open", "High", "Low", "Close", "Volume", "IV_Close", "HV_Close"]
    macro_features = ["SPX_Close", "NDX_Close", "VIX_Close", "TNX_Close"]
    for col in daily_features + macro_features:
        if col in df.columns:
            df[f"{col}_MA"] = df[col].rolling(window=roll_window).mean()
            df[f"{col}_Slope"] = df[col].rolling(window=roll_window).apply(calculate_ma_slope, raw=False).bfill()
            rolling_mean = df[col].rolling(window=roll_window).mean()
            rolling_std = df[col].rolling(window=roll_window).std()
            df[f"{col}_Z"] = ((df[col] - rolling_mean) / rolling_std).bfill()

    # Actionable Interpretable Features
    df["Gap_Open"] = df["Open"] - df["Close"].shift(1)
    df["Gap_Percentage"] = ((df["Gap_Open"]) / df["Close"].shift(1)) * 100
    df["Range_Percentage"] = ((df["High"] - df["Low"]) / df["Close"]) * 100

    # Weekly Range Feature
    df["Weekly_High"] = df["High"].rolling(window=roll_window).max()
    df["Weekly_Low"] = df["Low"].rolling(window=roll_window).min()
    df["Weekly_Range_Width"] = ((df["Weekly_High"] - df["Weekly_Low"]) / df["Close"]) * 100

    # Rolling GAP Features
    df["Rolling_Gap"] = df["Gap_Open"].rolling(window=roll_window).mean()
    df["Gap_Holding_Strength"] = df["Gap_Open"] / df["Close"].shift(1).rolling(window=roll_window).max()

    # Rolling Returns for Close
    if "Close" in df.columns:
        daily_returns = df["Close"].pct_change()
        df["Daily_Cumulative_Return"] = (daily_returns.add(1).rolling(window=roll_window).apply(np.prod, raw=True)).bfill()
        df["Daily_Past_Returns"] = [
            ", ".join([f"{val * 100:.2f}%" for val in daily_returns[i - roll_window + 1 : i + 1]]) if i >= roll_window - 1 else "None"
            for i in range(len(daily_returns))
        ]
        df.set_index("Date", inplace=True)
        weekly_close = df["Close"].resample("W").last()
        weekly_returns = weekly_close.pct_change()
        expanded_weekly_returns = weekly_returns.reindex(df.index, method="ffill")
        df["Weekly_Past_Returns"] = [
            ", ".join([f"{expanded_weekly_returns[i - (5 * week)]:.2%}" for week in range(4, 0, -1) if i - (5 * week) >= 0])
            if i >= 20
            else "None"
            for i in range(len(df))
        ]
        df.reset_index(inplace=True)

    # Market Beta Calculation
    df["Market_Returns"] = df[f"{market_ticker}_Close"].pct_change().fillna(0)
    df["Close_Returns"] = df["Close"].pct_change().fillna(0)
    df["Market_Beta"] = np.nan
    for i in range(roll_window, len(df)):
        y = df["Close_Returns"].iloc[i - roll_window : i]
        X = df["Market_Returns"].iloc[i - roll_window : i]
        if len(y) == len(X):
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            df.loc[df.index[i], "Market_Beta"] = model.params[1]

    # Rolling Options data
    if "OTM_Skew" in df.columns:
        for moneyness in ["OTM", "ATM", "ITM"]:
            for feature in ["IV_Call", "IV_Put", "Skew"]:
                df[f"MA_{moneyness}_{feature}"] = df[f"{moneyness}_{feature}"].rolling(window=roll_window).mean()

    # Additional Advanced Features
    df["MACD_Strength"] = df["MACD"] - df["Signal_Line"]
    df["MACD_Strength_Slope"] = df["MACD_Strength"].rolling(window=roll_window).apply(calculate_trend_slope, raw=False)
    df["RSI_Divergence"] = df["RSI"].diff().rolling(window=roll_window).mean()
    df["Volume_Momentum"] = df["Volume"].pct_change().rolling(window=roll_window).mean()
    df["Volume_Weighted_Returns"] = (df["Close_Returns"] * df["Volume"]).rolling(window=roll_window).mean()
    upper_band, middle_band, lower_band = bollinger_bands(df["Close"], window=roll_window)
    df["BB_Upper"] = upper_band
    df["BB_Middle"] = middle_band
    df["BB_Lower"] = lower_band
    df["BB_Width"] = (upper_band - lower_band) / df["Close"]
    df["IV_Percentile"] = (df["IV_Close"] - df["IV_Close"].rolling(window=roll_window).min()) / (
        df["IV_Close"].rolling(window=roll_window).max() - df["IV_Close"].rolling(window=roll_window).min()
    )
    df["VIX_Impact"] = df["VIX_Close"] * df["Market_Beta"]
    df["Momentum_Long"] = (df["RSI"] > 50) & (df["MACD_Strength"] > 0)
    df["Momentum_Short"] = (df["RSI"] < 50) & (df["MACD_Strength"] < 0)

    # Finalize and Return DataFrame
    engineered_df = df.drop_duplicates().loc[:, ~df.columns.duplicated(keep="first")]
    engineered_df.set_index("Date", inplace=True)
    engineered_df.columns = [col.replace(" ", "_").replace("-", "_") for col in engineered_df.columns]
    return engineered_df.bfill().ffill().fillna(0)


def clean_strategy_text(text: str | None) -> str:
    """Normalise strategy text to avoid YAML parsing errors."""

    if not isinstance(text, str):
        return ""

    cleaned_text = text.strip().replace("\n", " ").replace("\r", " ")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = cleaned_text.replace(":", " -")
    return cleaned_text


def update_historical_data_context(
    engineered_df: DataFrame,
    persona: str,
    HIGH_RISK_PROFILE: str,
    HIGH_OBJECTIVES: str,
    Last_LLM_Strat: str | None = None,
    Last_LLM_Strat_Action: str | None = None,
    Last_LLM_Strat_Returns: float | None = None,
    Last_LLM_Strat_Cum_Returns: float | None = None,
    Last_LLM_Strat_Days: int | None = None,
    Last_LLM_Strat_Action_Confidence: float | None = None,
    Peak_Returns: float | None = None,
    Trough_Returns: float | None = None,
    expert_df: DataFrame | None = None,
    classification: str = "High-Growth Tech Stock",
    news_factors: Sequence[str] | None = None,
    news_sentiment: float | None = None,
    news_impact_score: float | None = None,
) -> dict[str, Any]:
    """Populate trading prompt placeholders from engineered context data.

    Parameters
    ----------
    news_sentiment:
        Aggregated sentiment score for the latest news context. Defaults to ``0``
        when no sentiment is available.
    news_impact_score:
        Likert-style impact score summarising the expected market influence of the
        news factors. Defaults to ``0`` when unavailable.
    """

    Last_LLM_Strat = clean_strategy_text(Last_LLM_Strat)

    def _round_value(value):
        return round(value, 2) if isinstance(value, (int, float)) else value

    if engineered_df.empty:
        month_number = None
        week_number = None
    else:
        latest_index = engineered_df.index[-1]
        month_number = latest_index.month
        week_number = latest_index.isocalendar().week

    context: dict[str, Any] = {
        "month_number": month_number,
        "week_number": week_number,
        "classification": classification,
        "persona": persona,
        "risk_profile": HIGH_RISK_PROFILE,
        "portfolio_objectives": HIGH_OBJECTIVES,
        "Last_LLM_Strat": Last_LLM_Strat,
        "Last_LLM_Strat_Returns": Last_LLM_Strat_Returns,
        "Last_LLM_Strat_Days": Last_LLM_Strat_Days,
        "Last_LLM_Strat_Action": Last_LLM_Strat_Action,
        "Last_LLM_Strat_Action_Confidence": Last_LLM_Strat_Action_Confidence,
        "Last_LLM_Strat_Cum_Returns": Last_LLM_Strat_Cum_Returns,
        "Last_LLM_Strat_Best_Returns": Peak_Returns,
        "Last_LLM_Strat_Worse_Returns": Trough_Returns,
        "news_factors": list(news_factors) if news_factors else None,
        "news_sentiment": news_sentiment if news_sentiment is not None else 0,
        "news_impact_score": news_impact_score if news_impact_score is not None else 0,
    }

    # Add features from the specified list
    features = [
        # Basic Information
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Average",
        "Barcount",
        # Implied Volatility (IV) and Historical Volatility (HV) Metrics
        "IV_Open",
        "IV_High",
        "IV_Low",
        "IV_Close",
        "IV_Volume",
        "IV_Average",
        "IV_Barcount",
        "HV_Open",
        "HV_High",
        "HV_Low",
        "HV_Close",
        "HV_Volume",
        "HV_Average",
        "HV_Barcount",
        # General Ticker and Quarter Information
        "Quarter",
        "Stock_Price",
        # Economic Data
        "PMI",
        "PPI",
        "Retail Sales",
        "Employment",
        "Yield_Curve",
        "Treasury Yields",
        "CPI",
        "Consumer Confidence",
        "Consumer_Confidence",
        "Housing Starts",
        "Housing_Starts",
        "M2_Money_Supply",
        "Retail_Sales",
        "Treasury_Yields",
        "GDP",
        "Yield Curve",
        "PMI_YoY",
        "PPI_YoY",
        "Retail Sales_YoY",
        "Employment_YoY",
        "Yield_Curve_YoY",
        "Treasury Yields_YoY",
        "CPI_YoY",
        "Consumer Confidence_YoY",
        "Consumer_Confidence_YoY",
        "Housing Starts_YoY",
        "Housing_Starts_YoY",
        "M2_Money_Supply_YoY",
        "Retail_Sales_YoY",
        "Treasury_Yields_YoY",
        "GDP_YoY",
        "Yield Curve_YoY",
        "PMI_QoQ",
        "PPI_QoQ",
        "Retail Sales_QoQ",
        "Employment_QoQ",
        "Yield_Curve_QoQ",
        "Treasury Yields_QoQ",
        "CPI_QoQ",
        "Consumer Confidence_QoQ",
        "Consumer_Confidence_QoQ",
        "Housing Starts_QoQ",
        "Housing_Starts_QoQ",
        "M2_Money_Supply_QoQ",
        "Retail_Sales_QoQ",
        "Treasury_Yields_QoQ",
        "GDP_QoQ",
        "Yield Curve_QoQ"
        # Fundamental Metrics
        "TTM_FCF_per_Share",
        "Price_to_FCF_Ratio",
        "Net_Income",
        "Shareholder's_Equity",
        "Return_on_Equity",
        "TTM_Sales_per_Share",
        "Price_to_Sales_Ratio",
        "Book_Value_per_Share",
        "Price_to_Book_Ratio",
        "Total_Assets",
        "Return_on_Assets",
        "Current_Assets",
        "Current_Liabilities",
        "Current_Ratio",
        "EPS",
        "Current_Assets_Inventory",
        "Quick_Ratio",
        "TTM_Net_EPS",
        "PE_Ratio",
        "Long_term_Debt_/_Capital",
        "Gross_Margin",
        "Operating_Margin",
        "EBIT_Margin",
        "EBITDA_Margin",
        "Pre_Tax_Profit_Margin",
        "Net_Profit_Margin",
        "Asset_Turnover",
        "Inventory_Turnover_Ratio",
        "Days_Sales_In_Receivables",
        "Operating_Cash_Flow_Per_Share",
        "Free_Cash_Flow_Per_Share",
        "Long_Term_Debt",
        "Debt_to_Equity_Ratio",
        "Invested_Capital",
        "Return_on_Investment",
        # Quarter-over-Quarter (QoQ) Growth Features
        "Quick_Ratio_QoQ_Growth",
        "Current_Ratio_QoQ_Growth",
        "Debt_to_Equity_Ratio_QoQ_Growth",
        "Gross_Margin_QoQ_Growth",
        "Operating_Margin_QoQ_Growth",
        "EBIT_Margin_QoQ_Growth",
        "Net_Profit_Margin_QoQ_Growth",
        "Asset_Turnover_QoQ_Growth",
        "Inventory_Turnover_Ratio_QoQ_Growth",
        "Price_to_Book_Ratio_QoQ_Growth",
        "PE_Ratio_QoQ_Growth",
        "EPS_QoQ_Growth",
        "Net_Income_QoQ_Growth",
        "Free_Cash_Flow_Per_Share_QoQ_Growth",
        "Operating_Cash_Flow_Per_Share_QoQ_Growth",
        "Return_on_Equity_QoQ_Growth",
        "Return_on_Assets_QoQ_Growth",
        # Year-over-Year (YoY) Growth Features
        "Quick_Ratio_YoY_Growth",
        "Current_Ratio_YoY_Growth",
        "Debt_to_Equity_Ratio_YoY_Growth",
        "Gross_Margin_YoY_Growth",
        "Operating_Margin_YoY_Growth",
        "EBIT_Margin_YoY_Growth",
        "Net_Profit_Margin_YoY_Growth",
        "Asset_Turnover_YoY_Growth",
        "Inventory_Turnover_Ratio_YoY_Growth",
        "Price_to_Book_Ratio_YoY_Growth",
        "PE_Ratio_YoY_Growth",
        "EPS_YoY_Growth",
        "Net_Income_YoY_Growth",
        "Free_Cash_Flow_Per_Share_YoY_Growth",
        "Operating_Cash_Flow_Per_Share_YoY_Growth",
        "Return_on_Equity_YoY_Growth",
        "Return_on_Assets_YoY_Growth",
        # Index Metrics (SPX, NDX, VIX, TNX, IRX)
        "SPX_Open",
        "SPX_High",
        "SPX_Low",
        "SPX_Close",
        "SPX_Volume",
        "SPX_Average",
        "SPX_Barcount",
        "SPX_IV_Open",
        "SPX_IV_High",
        "SPX_IV_Low",
        "SPX_IV_Close",
        "SPX_IV_Volume",
        "SPX_IV_Average",
        "SPX_IV_Barcount",
        "SPX_HV_Open",
        "SPX_HV_High",
        "SPX_HV_Low",
        "SPX_HV_Close",
        "SPX_HV_Volume",
        "SPX_HV_Average",
        "SPX_HV_Barcount",
        "NDX_Open",
        "NDX_High",
        "NDX_Low",
        "NDX_Close",
        "NDX_Volume",
        "NDX_Average",
        "NDX_Barcount",
        "NDX_IV_Open",
        "NDX_IV_High",
        "NDX_IV_Low",
        "NDX_IV_Close",
        "NDX_IV_Volume",
        "NDX_IV_Average",
        "NDX_IV_Barcount",
        "NDX_HV_Open",
        "NDX_HV_High",
        "NDX_HV_Low",
        "NDX_HV_Close",
        "NDX_HV_Volume",
        "NDX_HV_Average",
        "NDX_HV_Barcount",
        "VIX_Open",
        "VIX_High",
        "VIX_Low",
        "VIX_Close",
        "VIX_Volume",
        "VIX_Average",
        "VIX_Barcount",
        "VIX_IV_Open",
        "VIX_IV_High",
        "VIX_IV_Low",
        "VIX_IV_Close",
        "VIX_IV_Volume",
        "VIX_IV_Average",
        "VIX_IV_Barcount",
        "VIX_HV_Open",
        "VIX_HV_High",
        "VIX_HV_Low",
        "VIX_HV_Close",
        "VIX_HV_Volume",
        "VIX_HV_Average",
        "VIX_HV_Barcount",
        "TNX_Open",
        "TNX_High",
        "TNX_Low",
        "TNX_Close",
        "TNX_Volume",
        "TNX_Average",
        "TNX_Barcount",
        "TNX_HV_Open",
        "TNX_HV_High",
        "TNX_HV_Low",
        "TNX_HV_Close",
        "TNX_HV_Volume",
        "TNX_HV_Average",
        "TNX_HV_Barcount",
        "IRX_Open",
        "IRX_High",
        "IRX_Low",
        "IRX_Close",
        "IRX_Volume",
        "IRX_Average",
        "IRX_Barcount",
        "IRX_HV_Open",
        "IRX_HV_High",
        "IRX_HV_Low",
        "IRX_HV_Close",
        "IRX_HV_Volume",
        "IRX_HV_Average",
        "IRX_HV_Barcount",
        # Option Metrics
        "OTM_IV_Call",
        "expiration_date",
        "OTM_IV_Put",
        "OTM_Skew",
        "ATM_IV_Call",
        "ATM_IV_Put",
        "ATM_Skew",
        "ITM_IV_Call",
        "ITM_IV_Put",
        "ITM_Skew",
        "Skew",
        "Vol_Surface",
        "DTE",
        # Moving Averages (MA) and Technical Indicators
        "20MA",
        "50MA",
        "100MA",
        "200MA",
        "RSI",
        "ATR",
        "MACD",
        "Signal_Line",
        "20MA_Slope",
        "20MA_Z",
        "50MA_Slope",
        "50MA_Z",
        "100MA_Slope",
        "100MA_Z",
        "200MA_Slope",
        "200MA_Z",
        # Price Metrics
        "Open_MA",
        "Open_Slope",
        "Open_Z",
        "High_MA",
        "High_Slope",
        "High_Z",
        "Low_MA",
        "Low_Slope",
        "Low_Z",
        "Close_MA",
        "Close_Slope",
        "Close_Z",
        "Volume_MA",
        "Volume_Slope",
        "Volume_Z",
        "IV_Close_MA",
        "IV_Close_Slope",
        "IV_Close_Z",
        "HV_Close_MA",
        "HV_Close_Slope",
        "HV_Close_Z",
        "SPX_Close_MA",
        "SPX_Close_Slope",
        "SPX_Close_Z",
        "NDX_Close_MA",
        "NDX_Close_Slope",
        "NDX_Close_Z",
        "VIX_Close_MA",
        "VIX_Close_Slope",
        "VIX_Close_Z",
        "TNX_Close_MA",
        "TNX_Close_Slope",
        "TNX_Close_Z",
        # Market Dynamics and Return Metrics
        "Gap_Open",
        "Gap_Percentage",
        "Range_Percentage",
        "Weekly_High",
        "Weekly_Low",
        "Weekly_Range_Width",
        "Rolling_Gap",
        "Gap_Holding_Strength",
        "Daily_Cumulative_Return",
        "Daily_Past_Returns",
        "Weekly_Past_Returns",
        "Market_Returns",
        "Close_Returns",
        "Market_Beta",
        # Option Implied Volatility Averages
        "MA_OTM_IV_Call",
        "MA_OTM_IV_Put",
        "MA_OTM_Skew",
        "MA_ATM_IV_Call",
        "MA_ATM_IV_Put",
        "MA_ATM_Skew",
        "MA_ITM_IV_Call",
        "MA_ITM_IV_Put",
        "MA_ITM_Skew",
        # Additional Metrics and Indicators
        "MACD_Strength",
        "MACD_Strength_Slope",
        "RSI_Divergence",
        "Volume_Momentum",
        "Volume_Weighted_Returns",
        "BB_Upper",
        "BB_Middle",
        "BB_Lower",
        "BB_Width",
        "IV_Percentile",
        "VIX_Impact",
        "Momentum_Long",
        "Momentum_Short",
    ]

    for feature in features:
        if feature in engineered_df.columns:
            context[feature] = _round_value(engineered_df[feature].iloc[-1])

    if expert_df is not None:
        expert_features = [
            "Expert_Action",
            "Entry_Point",
            "Take_Profit",
            "Stop_Loss",
            "Timeframe",
            "Next_Day_Returns",
            "Next_Week_Returns",
            "Next_Month_Returns",
        ]
        for feature in expert_features:
            if feature in expert_df.columns:
                context[feature] = _round_value(expert_df[feature].iloc[-1])

    return context


def load_yaml_template(file_path: Path | str) -> str:
    """Read a YAML template file into memory."""

    return Path(file_path).expanduser().read_text(encoding="utf-8")


def sanitize(val: Any) -> str:
    """Remove non-printable characters from arbitrary values."""

    return "".join(c for c in str(val) if c.isprintable())


def fill_yaml_template(data: Mapping[str, Any], yaml_template: str) -> Any:
    """Fill a YAML template with sanitised values before parsing."""

    clean_data = {k: sanitize(v) for k, v in data.items()}
    return yaml.safe_load(yaml_template.format(**clean_data))


def convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy containers to native Python types."""

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(convert_numpy(item) for item in obj)
    return obj


def calc_uncertainty_metrics(token_topk_logprobs: Sequence[Any]) -> dict[str, float]:
    """Compute entropy-based diagnostics from log probability traces."""

    if not token_topk_logprobs:
        raise ValueError("token_topk_logprobs must contain at least one element")

    first = token_topk_logprobs[0]
    top_logprobs = getattr(first, "top_logprobs", None)
    if not top_logprobs:
        raise ValueError("Each entry must expose a non-empty 'top_logprobs' attribute")

    token_count = len(token_topk_logprobs)
    k = len(top_logprobs)
    if k <= 1:
        raise ValueError("top_logprobs must contain at least two alternatives to compute entropy")

    total_entropy = 0.0
    total_logprob = 0.0
    max_entropy = math.log(k)

    for topk in token_topk_logprobs:
        probs = [math.exp(lp.logprob) for lp in topk.top_logprobs]
        topk_mass = sum(probs)
        if topk_mass > 1.0 + 1e-6:
            raise ValueError(f"Top-k mass exceeds unity: {topk_mass}")

        tail_mass = max(0.0, 1.0 - topk_mass)
        entropy_k = -sum(p * math.log(p) for p in probs if p > 0)
        entropy_tail = -tail_mass * math.log(tail_mass) if tail_mass > 0 else 0.0

        total_entropy += entropy_k + entropy_tail
        total_logprob += getattr(topk, "logprob")

    avg_logprob = total_logprob / token_count
    perplexity = math.exp(-avg_logprob)
    avg_entropy = total_entropy / token_count
    normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "entropy": avg_entropy,
        "normalized_entropy": normalized_entropy,
        "perplexity": perplexity,
        "avg_logprob": avg_logprob,
    }


@backoff.on_exception(backoff.expo, (Exception,), max_tries=5, jitter=backoff.full_jitter)
def call_openai_to_extract_news(
    articles: Sequence[str],
    news_yml_file: Path | str,
    ticker: str,
    target_name: str,
    date: str,
    client: Any,
    model: str,
    response_format: type[NewsCleaningResponse] = NewsCleaningResponse,
    LLM_OUTPUT_PATH: Path | str = "./data",
) -> dict[str, Any]:
    """Summarise relevant news factors for a ticker using the news-cleaning LLM prompt.

    Parameters
    ----------
    articles:
        Sequence of article bodies to summarise.
    news_yml_file:
        Path to the YAML template used to construct the prompt payload.
    ticker:
        Symbol that anchors the strategy context.
    target_name:
        Readable security name used for prompt grounding.
    date:
        Date stamp (``YYYY-MM``) applied to cached responses.
    client:
        OpenAI-compatible client exposing the beta chat completions API.
    model:
        Target LLM model identifier.
    response_format:
        Expected response schema returned by the LLM.
    LLM_OUTPUT_PATH:
        Directory used to persist cached responses.

    Returns
    -------
    dict[str, Any]
        Serialisable object containing the extracted factors and token metadata.
    """

    output_root = Path(LLM_OUTPUT_PATH).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    proba_path = output_root / f"{date}_news.yml"

    if proba_path.exists():
        return yaml.safe_load(proba_path.read_text(encoding="utf-8"))

    news_template = load_yaml_template(news_yml_file)
    prompt = prepare_yaml_with_articles(
        articles=articles,
        news_template=news_template,
        ticker=ticker,
        target_name=target_name,
        date=date,
    )

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        top_p=1,
        response_format=response_format,
        max_tokens=7500,
    )

    parsed_response = response.choices[0].message.parsed
    total_tokens = response.usage.total_tokens
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    cost = ((prompt_tokens / 1_000_000) * 0.15) + ((completion_tokens / 1_000_000) * 0.6)

    result = {
        "response": parsed_response.news_factors,
        "tokens_meta": {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
        },
    }

    proba_path.write_text(yaml.dump(result, default_flow_style=False), encoding="utf-8")
    return result


@backoff.on_exception(backoff.expo, (Exception,), max_tries=5, jitter=backoff.full_jitter)
def call_openai_for_strategy(
    context: Mapping[str, Any],
    prompt_file_name: str,
    strategy_file_name: str,
    LLM_OUTPUT_PATH: Path | str,
    client: Any,
    model: str,
    yaml_file: Path | str,
    top_k_tokens: int = 5,
) -> dict[str, Any]:
    """Generate a trade strategy suggestion via the strategy LLM prompt.

    Parameters
    ----------
    context:
        Feature dictionary injected into the strategy prompt template.
    prompt_file_name:
        Name of the cached prompt YAML artefact.
    strategy_file_name:
        Name of the cached strategy YAML artefact.
    LLM_OUTPUT_PATH:
        Output directory for prompt/strategy caches.
    client:
        OpenAI-compatible client instance capable of chat completions.
    model:
        Target LLM model identifier.
    yaml_file:
        Strategy prompt template to be populated.
    top_k_tokens:
        Number of log-probability entries captured per generated token.

    Returns
    -------
    dict[str, Any]
        Parsed strategy metadata enriched with uncertainty diagnostics.
    """

    output_root = Path(LLM_OUTPUT_PATH).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    strat_path = output_root / strategy_file_name
    prompt_file_path = output_root / prompt_file_name

    if strat_path.exists():
        return yaml.safe_load(strat_path.read_text(encoding="utf-8"))

    template = load_yaml_template(yaml_file)
    filled_template_raw = fill_yaml_template(context, template)
    prompt_text = yaml.dump(filled_template_raw, default_flow_style=True, allow_unicode=True)

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        response_format=TradeStrategy,
        temperature=0.7,
        max_completion_tokens=3500,
        frequency_penalty=1,
        presence_penalty=0.25,
        logprobs=True,
        top_logprobs=top_k_tokens,
    )

    trade_strategy_response = response.choices[0].message
    trade_strategy = (
        trade_strategy_response.parsed if isinstance(trade_strategy_response.parsed, dict) else trade_strategy_response.parsed.__dict__
    )
    if trade_strategy is None:
        raise RuntimeError("Parsed trade strategy is None")

    total_tokens = response.usage.total_tokens
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    cost = ((prompt_tokens / 1_000_000) * 0.15) + ((completion_tokens / 1_000_000) * 0.6)

    token_logprobs = response.choices[0].logprobs.content
    uncertainty_metrics = calc_uncertainty_metrics(token_logprobs)
    long_token_proba = max((tp.logprob for tp in token_logprobs if Action.LONG.value in tp.token), default=-float("inf"))
    short_token_proba = max((tp.logprob for tp in token_logprobs if Action.SHORT.value in tp.token), default=-float("inf"))

    response_data = {
        "explanation": trade_strategy["explanation"],
        "action_confidence": trade_strategy["action_confidence"],
        "action": trade_strategy["action"].value,
        "long_token_proba": long_token_proba,
        "short_token_proba": short_token_proba,
        "perplexity": uncertainty_metrics["perplexity"],
        "entropy": uncertainty_metrics["entropy"],
        "normalized_entropy": uncertainty_metrics["normalized_entropy"],
        "tokens_meta": {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
        },
    }

    response_yml = convert_numpy(response_data)
    strat_path.write_text(
        yaml.dump(response_yml, default_flow_style=False, allow_unicode=True, indent=2),
        encoding="utf-8",
    )
    prompt_file_path.write_text(
        yaml.safe_dump(filled_template_raw, default_flow_style=False, allow_unicode=True, indent=2),
        encoding="utf-8",
    )

    return response_data


def reset_llm_signals(ticker_df: DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> DataFrame:
    """Initialise columns used for LLM strategy overlays within the date range."""

    ticker_df = ticker_df.sort_index()
    columns_to_fill = [
        "strategy",
        "explanation",
        "news_factors",
        "trade_action",
        "trade_signal",
        "action_confidence",
        "entropy",
        "perplexity",
        "strat_signal_long",
        "strat_signal_short",
        "long_token_proba",
        "short_token_proba",
        "tokens_meta_strat",
        "tokens_meta_news",
    ]
    ticker_df[columns_to_fill] = np.nan

    filtered_ticker_df = ticker_df[(ticker_df.index >= start_date) & (ticker_df.index <= end_date)].copy()

    return filtered_ticker_df


def feature_engineer_llm_signals(
    filtered_ticker_df: DataFrame,
    date: pd.Timestamp,
    strategy_file_name: str,
    trade_strategy: Mapping[str, Any],
    strat_history: Mapping[str, Any],
    news_results: Mapping[str, Any] | None = None,
    eps: float = 0.0001,
) -> DataFrame:
    """Project LLM strategy metadata forward until the next evaluation point."""

    if eps <= 0 or eps >= 1:
        raise ValueError("eps must be within (0, 1)")

    is_llm_long = trade_strategy["action"] == Action.LONG
    index = filtered_ticker_df.loc[date:].index

    entropy = trade_strategy["entropy"]
    confidence = trade_strategy["action_confidence"]
    if not 0 <= entropy <= 1:
        raise ValueError(f"Entropy must be in [0, 1], received {entropy}")
    if not MIN_CONFIDENCE <= confidence <= MAX_CONFIDENCE:
        raise ValueError(f"Confidence must be between {MIN_CONFIDENCE} and {MAX_CONFIDENCE}, received {confidence}")

    certainty = eps + (1 - eps) * (1 - entropy)
    if not 0 < certainty <= 1:
        raise ValueError(f"Certainty must be within (0, 1], received {certainty}")

    conf = confidence / MAX_CONFIDENCE
    llm_signal = (conf * certainty) ** 0.5

    object_columns = (
        "strategy",
        "explanation",
        "tokens_meta_strat",
        "news_factors",
        "tokens_meta_news",
    )
    for column in object_columns:
        if column not in filtered_ticker_df.columns:
            filtered_ticker_df[column] = pd.Series(index=filtered_ticker_df.index, dtype="object")
        elif not is_object_dtype(filtered_ticker_df[column].dtype):
            filtered_ticker_df[column] = filtered_ticker_df[column].astype("object")

    filtered_ticker_df.loc[index, "strategy"] = strategy_file_name
    filtered_ticker_df.loc[index, "trade_action"] = int(is_llm_long)
    filtered_ticker_df.loc[index, "action_confidence"] = confidence
    filtered_ticker_df.loc[index, "explanation"] = trade_strategy["explanation"]
    filtered_ticker_df.loc[index, "trade_signal"] = llm_signal
    trade_strategy["trade_signal"] = llm_signal

    filtered_ticker_df.loc[index, "strat_signal_long"] = llm_signal if is_llm_long else 0.0
    filtered_ticker_df.loc[index, "strat_signal_short"] = 0.0 if is_llm_long else llm_signal

    filtered_ticker_df.loc[index, "perplexity"] = trade_strategy["perplexity"]
    filtered_ticker_df.loc[index, "entropy"] = entropy
    filtered_ticker_df.loc[index, "long_token_proba"] = trade_strategy["long_token_proba"]
    filtered_ticker_df.loc[index, "short_token_proba"] = trade_strategy["short_token_proba"]
    filtered_ticker_df.loc[index, "tokens_meta_strat"] = [trade_strategy["tokens_meta"]] * len(index)
    filtered_ticker_df.loc[index, "cummulative_returns"] = strat_history["cummulative_returns"]
    filtered_ticker_df.loc[index, "last_returns"] = strat_history["last_returns"]
    max_days = strat_history["elapsed_days"] if strat_history["elapsed_days"] is not None else 0
    arr = np.arange(0, max_days + 1)
    sub_index = index
    if len(arr) > len(sub_index):
        arr = arr[: len(sub_index)]
    elif len(arr) < len(sub_index):
        pad = np.full(len(sub_index) - len(arr), max_days)
        arr = np.concatenate([arr, pad])
    filtered_ticker_df.loc[sub_index, "elapsed_days"] = arr
    if news_results is not None:
        news_response = news_results.get("response")
        if isinstance(news_response, Sequence) and not isinstance(news_response, (str, bytes)):
            news_payload = list(news_response)
        else:
            news_payload = news_response

        tokens_meta = news_results.get("tokens_meta")
        for idx in index:
            filtered_ticker_df.at[idx, "news_factors"] = news_payload
            filtered_ticker_df.at[idx, "tokens_meta_news"] = tokens_meta

    return filtered_ticker_df


def generate_strategy_for_ticker(
    ticker_df: DataFrame,
    ticker: str,
    LLM_OUTPUT_PATH: Path | str,
    persona: str,
    HIGH_RISK_PROFILE: str,
    HIGH_OBJECTIVES: str,
    client: Any,
    model: str,
    news_yaml_file: Path | str | None = None,
    strategy_yaml_file: Path | str | None = None,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    classification: str = CLASSIFICATION,
    time_horizon: Literal["monthly", "weekly"] = "monthly",
    max_news: int = 5,
    show_progress: bool = True,
) -> DataFrame:
    """Iterate over the price history to generate LLM strategies per evaluation window."""

    def get_date_condition(date: pd.Timestamp) -> tuple[int, int]:
        return (date.year, date.month) if time_horizon == "monthly" else date.isocalendar()[:2]

    output_root = Path(LLM_OUTPUT_PATH).expanduser().resolve()
    ticker_output_path = output_root / ticker
    ticker_output_path.mkdir(parents=True, exist_ok=True)
    output_file = ticker_output_path / f"{ticker}_aug.csv"
    start_date_ts = pd.to_datetime(start_date, utc=True)
    end_date_ts = pd.to_datetime(end_date, utc=True)
    filtered_ticker_df = reset_llm_signals(ticker_df, start_date_ts, end_date_ts)

    current_time_condition = None
    previous_strategy = None
    strategy_start_date = None
    strategy_change_price = filtered_ticker_df["Close"].iloc[0]
    cum_returns = np.array([])
    prev_cum_rets = None
    days_elapsed = None
    best_rets = 0
    worse_rets = 0

    date_iterator = filtered_ticker_df.iterrows()
    if show_progress:
        date_iterator = tqdm(
            date_iterator,
            total=len(filtered_ticker_df),
            desc=f"Generating strategies for {ticker}",
            leave=False,
        )

    for date, day_data in date_iterator:
        time_condition = get_date_condition(date)
        if time_condition != current_time_condition:
            current_time_condition = time_condition

            if time_horizon == "monthly":
                anchor_date = date.replace(day=1)
            elif time_horizon == "weekly":
                anchor_date = pd.to_datetime(date.to_period("W").start_time).tz_localize("UTC")
            else:
                raise Exception("Unsupported time_horizon")

            first_business_day = pd.date_range(anchor_date, periods=1, freq="B")[0]
            start_idx = ticker_df.index.get_indexer([first_business_day], method="nearest")[0]
            context_row = ticker_df.iloc[[start_idx]]

            if previous_strategy:
                current_price = day_data["Close"]
                ret = (
                    (current_price - strategy_change_price) / strategy_change_price
                    if previous_strategy["action"] == Action.LONG.value
                    else (strategy_change_price - current_price) / strategy_change_price
                )
                cum_returns = np.append(cum_returns, ret)
                prev_cum_rets = np.cumprod(1 + cum_returns)[-1] - 1
                days_elapsed = np.busday_count(strategy_start_date.date(), date.date())
                best_rets = np.max(cum_returns)
                worse_rets = np.min(cum_returns)

            news_results = None
            if news_yaml_file is not None:
                last_n_news = list(
                    dict.fromkeys(
                        f
                        for f in chain.from_iterable(
                            ticker_df.loc[date:].iloc[: 20 if time_horizon == "monthly" else 5]["content"].dropna().tolist()
                        )
                        if f and f.strip()
                    )
                )
                if last_n_news:
                    news_results = call_openai_to_extract_news(
                        articles=last_n_news[-max_news:],
                        news_yml_file=news_yaml_file,
                        ticker=ticker,
                        target_name=TICKER_COMPANY_NAME_MAP[ticker],
                        date=f"{date.year}-{date.strftime('%m')}",
                        client=client,
                        model=model,
                        LLM_OUTPUT_PATH=str(ticker_output_path),
                    )

            news_factors_payload: Sequence[str] | None = None
            news_sentiment_value: float | None = None
            news_impact_value: float | None = None
            if news_results:
                raw_response = news_results.get("response")
                if isinstance(raw_response, Sequence) and not isinstance(raw_response, (str, bytes)):
                    if raw_response and isinstance(raw_response[0], Mapping):
                        news_factors_payload = [str(item.get("factor") or item) for item in raw_response if isinstance(item, Mapping)]
                        sentiments = [
                            item.get("sentiment")
                            for item in raw_response
                            if isinstance(item, Mapping) and item.get("sentiment") is not None
                        ]
                        impacts = [
                            item.get("market_impact")
                            for item in raw_response
                            if isinstance(item, Mapping) and item.get("market_impact") is not None
                        ]
                        if sentiments:
                            news_sentiment_value = float(np.mean(sentiments))
                        if impacts:
                            news_impact_value = float(np.mean(impacts))
                    else:
                        news_factors_payload = [str(item) for item in raw_response]
                elif raw_response is not None:
                    news_factors_payload = [str(raw_response)]

                news_sentiment_value = news_results.get("news_sentiment", news_sentiment_value)
                news_impact_value = news_results.get("news_impact_score", news_impact_value)

            strat_history = {
                "last_strategy": previous_strategy["explanation"] if previous_strategy else None,
                "last_action": previous_strategy["action"] if previous_strategy else None,
                "last_returns": ret if previous_strategy else None,
                "cummulative_returns": prev_cum_rets if previous_strategy else None,
                "elapsed_days": days_elapsed if previous_strategy else None,
                "best_returns": best_rets if previous_strategy else None,
                "worst_returns": worse_rets if previous_strategy else None,
                "last_action_confidence": previous_strategy["trade_signal"] if previous_strategy else None,
            }

            context = update_historical_data_context(
                engineered_df=context_row,
                persona=persona,
                HIGH_RISK_PROFILE=HIGH_RISK_PROFILE,
                HIGH_OBJECTIVES=HIGH_OBJECTIVES,
                classification=classification,
                Last_LLM_Strat=strat_history["last_strategy"],
                Last_LLM_Strat_Action=strat_history["last_action"],
                Last_LLM_Strat_Action_Confidence=strat_history["last_action_confidence"],
                Last_LLM_Strat_Returns=strat_history["last_returns"],
                Last_LLM_Strat_Cum_Returns=strat_history["cummulative_returns"],
                Last_LLM_Strat_Days=strat_history["elapsed_days"],
                Peak_Returns=strat_history["best_returns"],
                Trough_Returns=strat_history["worst_returns"],
                news_factors=news_factors_payload,
                news_sentiment=news_sentiment_value,
                news_impact_score=news_impact_value,
            )

            strategy_file_name = f"{date.strftime('%Y-%m%d')}-{ticker}-1.yml"
            prompt_file_name = f"{date.strftime('%Y-%m%d')}-{ticker}-1_prompt.yml"

            trade_strategy = call_openai_for_strategy(
                context=context,
                prompt_file_name=prompt_file_name,
                strategy_file_name=strategy_file_name,
                LLM_OUTPUT_PATH=str(ticker_output_path),
                client=client,
                model=model,
                yaml_file=strategy_yaml_file,
            )

            if previous_strategy is None or trade_strategy["action"] != previous_strategy["action"]:
                strategy_change_price = day_data["Close"]
                strategy_start_date = date
                cum_returns = np.array([])
                best_rets = 0
                worse_rets = 0

                previous_strategy = trade_strategy
                filtered_ticker_df = feature_engineer_llm_signals(
                    filtered_ticker_df,
                    date,
                    strategy_file_name,
                    trade_strategy,
                    strat_history,
                    news_results,
                )

            filtered_ticker_df = filtered_ticker_df.ffill().bfill()
            filtered_ticker_df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")

    return filtered_ticker_df

def generate_random_sample_dates(dataframe: DataFrame, num_samples: int = 252) -> pd.DatetimeIndex:
    """Sample unique timestamps from a DataFrame index for inspection."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    timestamps = pd.to_datetime(dataframe.index, utc=True)
    if num_samples > len(timestamps):
        raise ValueError("num_samples cannot exceed the number of available timestamps")

    random_dates = np.random.choice(timestamps, size=num_samples, replace=False)
    return pd.to_datetime(random_dates)


def evaluate_trading_metrics(
    trades_df: DataFrame,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
    rl_env: Any | None = None,
) -> tuple[dict[str, float | None], DataFrame]:
    """Calculate classical and LLM-specific trading diagnostics."""

    if rl_env is not None:
        if PerformanceEstimator is None:
            raise RuntimeError(
                "rl_agent_utils.PerformanceEstimator is unavailable. Install the rl environment dependencies to enable RL metrics."
            )
        trades_df["trade_action"] = rl_env.data["action"] == 0
        trades_df["reward"] = rl_env.data["returns"]
        trades_df["returns"] = rl_env.data["returns"]

        analyser = PerformanceEstimator(rl_env.data)
        df_metrics = analyser.getComputedPerformance()
        metrics: dict[str, float | None] = dict(zip(df_metrics["Metric"], df_metrics["Value"], strict=False))
    else:
        trades_df = trades_df.copy()
        trades_df["Position"] = trades_df["trade_action"].apply(lambda value: 1 if value == 1 else -1)
        trades_df["Lagged_Position"] = trades_df["Position"].shift(1)
        trades_df["returns"] = trades_df["Lagged_Position"] * trades_df["Close"].pct_change()
        trades_df["returns"] = trades_df["returns"].fillna(0)

        excess_daily_returns = trades_df["returns"] - (risk_free_rate / trading_days)
        sharpe_ratio = (
            (excess_daily_returns.mean() / excess_daily_returns.std()) * np.sqrt(trading_days)
            if excess_daily_returns.std() != 0
            else float("nan")
        )
        trades_df["Sharpe_Ratio"] = sharpe_ratio

        trades_df["Absolute_Position_Change"] = trades_df["Position"].diff().abs()
        total_trading_volume = trades_df["Absolute_Position_Change"].sum()
        average_position = trades_df["Position"].abs().mean()
        portfolio_turnover = total_trading_volume / average_position if average_position != 0 else float("nan")
        trades_df["Portfolio_Turnover"] = portfolio_turnover

        trades_df["Cumulative_Wealth"] = (1 + trades_df["returns"]).cumprod()
        rolling_max = trades_df["Cumulative_Wealth"].cummax()
        trades_df["Drawdown"] = 1 - trades_df["Cumulative_Wealth"] / rolling_max
        max_drawdown = trades_df["Drawdown"].max()
        trades_df["Max_Drawdown"] = max_drawdown

        drawdown_periods = (trades_df["Drawdown"] > 0).astype(int)
        drawdown_durations = drawdown_periods.groupby((drawdown_periods != drawdown_periods.shift()).cumsum()).cumsum()
        mean_drawdown_duration = drawdown_durations[trades_df["Drawdown"] > 0].mean()
        trades_df["Mean_Drawdown_Duration"] = mean_drawdown_duration

        cumulative_returns = trades_df["Cumulative_Wealth"] - 1
        trades_df["cumulative_returns"] = cumulative_returns

        metrics = {
            "Sharpe Ratio (Annualized SR)": sharpe_ratio,
            "Portfolio Turnover (PTR)": portfolio_turnover,
            "Maximum Drawdown (MDD)": max_drawdown,
            "Mean Drawdown Duration (MDDur)": mean_drawdown_duration,
            "Cumulative Returns": cumulative_returns.iloc[-1],
        }

    mean_ppl = mean_entropy = max_ppl = max_entropy = norm_entropy = max_norm_entropy = None
    mean_tokens = mean_costs = max_tokens = max_costs = total_tokens = total_costs = None

    if "tokens_meta_strat" in trades_df.columns:
        strat_data = trades_df["tokens_meta_strat"].apply(pd.Series)
        trades_df["total_tokens"] = strat_data["total_tokens"]
        trades_df["cost"] = strat_data["cost"]
        trades_df["prompt_tokens"] = strat_data["prompt_tokens"]
        trades_df["completion_tokens"] = strat_data["completion_tokens"]

        news_data = trades_df["tokens_meta_news"].apply(pd.Series)
        if not news_data.empty and news_data.notna().any().any():
            trades_df["total_tokens"] += news_data["total_tokens"].fillna(0)
            trades_df["cost"] += news_data["cost"].fillna(0)
            trades_df["prompt_tokens"] += news_data["prompt_tokens"].fillna(0)
            trades_df["completion_tokens"] += news_data["completion_tokens"].fillna(0)

        if "perplexity" in trades_df.columns:
            trades_df["perplexity"] = trades_df["perplexity"].clip(lower=-3, upper=3)
            mean_ppl = trades_df["perplexity"].mean()
            max_ppl = trades_df["perplexity"].max()

        if "entropy" in trades_df.columns:
            mean_entropy = trades_df["entropy"].mean()
            max_entropy = trades_df["entropy"].max()

        if "normalized_entropy" in trades_df.columns:
            norm_entropy = trades_df["normalized_entropy"].mean()
            max_norm_entropy = trades_df["normalized_entropy"].max()

        mean_tokens = trades_df["total_tokens"].mean()
        mean_costs = trades_df["cost"].mean()
        max_tokens = trades_df["total_tokens"].max()
        max_costs = trades_df["cost"].max()
        total_tokens = trades_df["total_tokens"].sum()
        total_costs = trades_df["cost"].sum()

        trades_df["Mean_Entropy"] = mean_entropy
        trades_df["Max_Perplexity"] = max_ppl
        trades_df["Max_Entropy"] = max_entropy
        trades_df["Mean_Tokens"] = mean_tokens
        trades_df["Mean_Costs"] = mean_costs
        trades_df["Max_Tokens"] = max_tokens
        trades_df["Max_Costs"] = max_costs
        trades_df["Total_Tokens"] = total_tokens
        trades_df["Total_Costs"] = total_costs
        trades_df["mean_normalized_entropy"] = norm_entropy
        trades_df["max_normalized_entropy"] = max_norm_entropy
        month_index = trades_df.index
        if getattr(month_index, "tz", None) is not None:
            month_index = month_index.tz_convert("UTC").tz_localize(None)
        trades_df["month"] = month_index.to_period("M")

    metrics.update(
        {
            "Mean Perplexity": mean_ppl,
            "Mean Entropy": mean_entropy,
            "Max Perplexity": max_ppl,
            "Max Entropy": max_entropy,
            "Mean Normalized Entropy": norm_entropy,
            "Max Normalized Entropy": max_norm_entropy,
            "Mean Tokens": mean_tokens,
            "Mean Costs": mean_costs,
            "Max Tokens": max_tokens,
            "Max Costs": max_costs,
            "Total Tokens": total_tokens,
            "Total Costs": total_costs,
        }
    )

    return metrics, trades_df


def expert_trades(ticker_df: DataFrame, high_risk: bool = True) -> DataFrame:
    """Generate heuristic expert trades aligned with the desired risk profile."""

    target_factor = 0.025 if high_risk else 0.05
    loss_factor = 0.025 if high_risk else 0.15
    ticker_df = ticker_df.sort_index()

    ticker_df['Entry_Point'] = np.nan
    ticker_df['Take_Profit'] = np.nan
    ticker_df['Stop_Loss'] = np.nan
    ticker_df['Timeframe'] = np.nan
    ticker_df['Expert_Action'] = None # Below are for the RF
    ticker_df['trade_action'] = 0
    ticker_df['Next_Day_Returns'] = np.nan
    ticker_df['Next_Week_Returns'] = np.nan
    ticker_df['Next_Month_Returns'] = np.nan
    for i in range(len(ticker_df)):
        current_date = ticker_df.index[i]
        current_close = ticker_df['Close'].iloc[i]

        # Safely calculate future returns based on the predefined timeframes
        next_day_close = ticker_df['Close'].iloc[i + 1] if i + 1 < len(ticker_df) else np.nan
        next_week_close = ticker_df['Close'].iloc[i + 3] if i + 3 < len(ticker_df) else np.nan
        next_month_close = ticker_df["Close"].iloc[i + 20] if i + 20 < len(ticker_df) else np.nan

        ticker_df.loc[current_date, 'Next_Day_Returns'] = (
            (next_day_close / current_close - 1) if not np.isnan(next_day_close) else np.nan
        )
        ticker_df.loc[current_date, 'Next_Week_Returns'] = (
            (next_week_close / current_close - 1) if not np.isnan(next_week_close) else np.nan
        )
        ticker_df.loc[current_date, 'Next_Month_Returns'] = (
            (next_month_close / current_close - 1) if not np.isnan(next_month_close) else np.nan
        )
        next_day_returns = ticker_df.loc[current_date, 'Next_Day_Returns']
        next_week_returns = ticker_df.loc[current_date, 'Next_Week_Returns']
        next_month_returns = ticker_df.loc[current_date, 'Next_Month_Returns']
        if high_risk:
            weights = [0.1, 0.9, 0]
        else:
            weights = [0, 0.6, 0.4]
        weighted_average = sum(
            weights[j] * r if not np.isnan(r) else 0
            for j, r in enumerate([next_day_returns, next_week_returns, next_month_returns])
        )
        trade_action = Action.LONG if weighted_average > 0 else Action.SHORT
        if trade_action == Action.LONG:
            entry = current_close * (1 - target_factor / 5)
            target = current_close * (1 + target_factor)
            stop_loss = entry * (1 - loss_factor)
            ticker_df.loc[current_date, 'trade_action'] = 1
        else:
            entry = current_close * (1 + target_factor / 5)
            target = current_close * (1 - target_factor)
            stop_loss = entry * (1 + loss_factor)
            ticker_df.loc[current_date, 'trade_action'] = 0
        ticker_df.loc[current_date, 'Entry_Point'] = entry
        ticker_df.loc[current_date, 'Take_Profit'] = target
        ticker_df.loc[current_date, 'Stop_Loss'] = stop_loss
        ticker_df.loc[current_date, 'Timeframe'] = 5 if high_risk else 20
        ticker_df.loc[current_date, 'Expert_Action'] = trade_action.value

    return ticker_df


def plot_llm_trade(
    llm_df: DataFrame,
    plot: bool = True,
    expert_trader_df: DataFrame | None = None,
    plot_other_trades: bool = True,
) -> tuple[plt.Figure, plt.Figure | None, plt.Figure]:
    """Visualise LLM-driven trades, uncertainty diagnostics, and auxiliary metrics."""

    llm_df = llm_df.copy()
    if "long_token_proba" in llm_df.columns:
        has_probas = (
            'long_conf_score' in llm_df.columns and
            'short_conf_score' in llm_df.columns and
            not (llm_df['long_conf_score'].isnull().all() and llm_df['short_conf_score'].isnull().all())
        )

        # Dynamically adjust the number of subplots
        num_subplots = 5 if has_probas else 4
        fig1, axes1 = plt.subplots(num_subplots, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2] + [1] * (num_subplots - 1)}, sharex=True)
    else:
        fig2 = None
        fig3 = None
        fig1, axes1 = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2,1]}, sharex=True)

    # Plot 1: Price with Technical Indicators
    ax = axes1[0]
    ax.set_title("Trades, Price, and Technical Indicators", fontsize=12)
    longs_llm = llm_df[llm_df['trade_action'] == 1]
    shorts_llm = llm_df[llm_df['trade_action'] == 0]
    ax.scatter(longs_llm.index, longs_llm['Close'], marker='^', edgecolors='cyan', label='LONG', alpha=0.8, facecolors='none')
    ax.scatter(shorts_llm.index, shorts_llm['Close'], marker='v', edgecolors='magenta', label='SHORT', alpha=0.8, facecolors='none')
    if expert_trader_df is not None and plot_other_trades:
        offset = 2
        longs_llm = expert_trader_df[expert_trader_df['trade_action'] == 1]
        shorts_llm = expert_trader_df[expert_trader_df['trade_action'] == 0]

        ax.scatter(longs_llm.index, longs_llm['Close'] - offset,
                marker='^', color='blue', label='Expert LONG', alpha=0.8)
        ax.scatter(shorts_llm.index, shorts_llm['Close'] - offset,
                marker='v', color='red', label='Expert SHORT', alpha=0.8)


    ax.plot(llm_df.index, llm_df['Close'], label='Close', color='blue', linewidth=1.5)
    ax.plot(llm_df.index, llm_df['20MA'], label='20 Day MA', color='gray', linestyle="-.", linewidth=2)
    ax.plot(llm_df.index, llm_df['50MA'], label='50 Day MA', color='black', linestyle="-.", linewidth=2)
    ax.plot(llm_df.index, llm_df['100MA'], label='100 Day MA', color='gray', linestyle="--", linewidth=2)
    ax.plot(llm_df.index, llm_df['200MA'], label='200 Day MA', color='black', linestyle="--", linewidth=2)

    if 'LLM_Trade_Action' in llm_df and plot_other_trades:
        change_points = llm_df['LLM_Trade_Action'].shift(1) != llm_df['LLM_Trade_Action']
        llm_changes = llm_df[change_points]

        longs = llm_changes[llm_changes['LLM_Trade_Action'] == 1]
        flats = llm_changes[llm_changes['LLM_Trade_Action'] == 0]

        fig1.axes[0].scatter(
            longs.index,
            longs['Close'],
            marker='^',
            color='green',
            label='LLM Long',
            s=100
        )

        fig1.axes[0].scatter(
            flats.index,
            flats['Close'],
            marker='v',
            color='red',
            label='LLM Short',
            s=100
        )

    ax.legend( bbox_to_anchor=(1.15, 1))
    ax.grid(True)

    axes1[1].plot(llm_df.index, llm_df['cumulative_returns'], label="Cumulative Returns", alpha=0.7)
    axes1[1].fill_between(
        llm_df.index,
        llm_df['cumulative_returns'],
        where=llm_df['cumulative_returns'] >= 0,
        color='green',
        alpha=0.3,
        label="Positive Returns"
    )
    axes1[1].fill_between(
        llm_df.index,
        llm_df['cumulative_returns'],
        where=llm_df['cumulative_returns'] < 0,
        color='red',
        alpha=0.3,
        label="Negative Returns"
    )
    axes1[1].set_title("Cumulative Returns")
    axes1[1].legend(bbox_to_anchor=(1.15, 1))
    axes1[1].grid(True)

    if "reward" in llm_df.columns:
        ax2 = axes1[1].twinx()
        ax2.plot(
            llm_df.index,
            llm_df['reward'].cumsum(),
            color='blue',
            label="Reward",
            linestyle='--',
            alpha=0.7
        )
        """ax2.plot(
            llm_df.index,
            llm_df['other_reward'].cumsum(),
            color='green',
            label="Other",
            linestyle='dashdot',
            alpha=0.8
        )"""
        ax2.set_ylabel("Cumulative Rewards", color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        axes1[1].set_title("Cumulative Returns and Cumulative Rewards")
        axes1[1].set_ylabel("Cumulative Returns")
        axes1[1].grid(True)
        axes1[1].legend(loc='upper left')
        ax2.legend(loc='upper right')



    if "long_token_proba" in llm_df.columns:
        # Convert log probabilities to percentages for plotting
        log_prob_columns = ['long_token_proba', 'short_token_proba', 'long_token_proba', 'short_token_proba']
        for col in log_prob_columns:
            llm_df[f'{col}_perc'] = np.exp(llm_df[col])


        # Plot 2: Token Probabilities
        if "strat_signal_long" in llm_df.columns:
            axes1[2].plot(llm_df.index, llm_df['trade_signal'], label="Trade Signal", alpha=0.7)
            axes1[2].set_title("LLM Signal Over Time")
            axes1[2].legend(bbox_to_anchor=(1.15, 1))
            axes1[2].grid(True)

        # Plot 3: Long and Short Probabilities (if present)
        if has_probas:
            llm_df['long_perc'] = llm_df['long_conf_score'] / (llm_df['long_conf_score'] + llm_df['short_conf_score']) * 100
            llm_df['short_perc'] = llm_df['short_conf_score'] / (llm_df['long_conf_score'] + llm_df['short_conf_score']) * 100
            axes1[3].plot(llm_df.index, llm_df['long_perc'], label="Long Token Probability", alpha=0.7)
            axes1[3].plot(llm_df.index, llm_df['short_perc'], label="Short Token Probability", alpha=0.7)
            axes1[3].set_title("Long Short Confidence")
            axes1[3].legend(bbox_to_anchor=(1.15, 1))
            axes1[3].grid(True)

        # Perplexity and Entropy
        ax1 = axes1[4 if has_probas else 3]
        ax2 = ax1.twinx()
        ax1.plot(
            llm_df.index,
            llm_df['perplexity'],
            linestyle="--",
            label="Perplexity",
            alpha=0.7,
            color='purple'
        )
        ax2.plot(
            llm_df.index,
            llm_df['entropy'],
            label="Entropy",
            alpha=0.7,
            color='red'
        )
        ax1.set_title("Uncertainty Over Time")
        ax1.set_ylabel("Perplexity")
        ax2.set_ylabel("Entropy")
        ax1.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
        ax2.legend(loc="lower right", bbox_to_anchor=(1.15, 1))
        ax1.grid(True)

        fig1.tight_layout()

        fig3, axes3 = plt.subplots(6 if has_probas else 5, 1, figsize=(16, 10))

        # Plot 1: Distribution of Token Probabilities
        ax = axes3[0]
        ax.set_title("Distribution of Token Probabilities", fontsize=12)
        ax.hist(llm_df['long_token_proba_perc'], bins=50, color='blue', alpha=0.7, label='Long')
        ax.hist(llm_df['short_token_proba_perc'], bins=50, color='red', alpha=0.7, label='Short')
        ax.set_xlabel("Token Probabilities (Long & Short)")
        ax.set_ylabel("Frequency")
        ax.legend(bbox_to_anchor=(1.15, 1))
        ax.grid(True)

        # Plot 3: Distribution of Normalized Entropy
        ax = axes3[1]
        ax.set_title("Distribution of Entropy", fontsize=12)
        ax.hist(
            llm_df['normalized_entropy'] if 'normalized_entropy' in llm_df.columns else llm_df['entropy'],
            bins=50,
            color='red',
            alpha=0.7,
            label='Normalized Entropy'
        )
        ax.set_xlabel("Normalized Entropy")
        ax.set_ylabel("Frequency")
        ax.legend(bbox_to_anchor=(1.15, 1))
        ax.grid(True)

        # Plot 4: Distribution of Perplexity
        ax = axes3[2]
        ax.set_title("Distribution of Perplexity", fontsize=12)
        if 'perplexity' in llm_df.columns:
            ax.hist(
                llm_df['perplexity'],
                bins=50,
                color='purple',
                alpha=0.7,
                label='Perplexity'
            )
            ax.set_xlabel("Perplexity")
            ax.set_ylabel("Frequency")
            ax.legend(bbox_to_anchor=(1.15, 1))
            ax.grid(True)


        # Plot 4: Distribution of Costs
        ax = axes3[3]
        ax.set_title("Distribution of Costs", fontsize=12)
        if 'cost' in llm_df.columns:
            ax.hist(llm_df['cost'], bins=50, color='orange', alpha=0.7, label="Total Cost")
            ax.set_xlabel("Cost")
            ax.set_ylabel("Frequency")
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.15, 1))
        else:
            ax.text(0.5, 0.5, "Costs not available", fontsize=12, ha='center')

        # Plot 5: Distribution of Total Tokens
        ax = axes3[4]
        ax.set_title("Distribution of Total Tokens", fontsize=12)
        if 'total_tokens' in llm_df.columns:
            ax.hist(llm_df['total_tokens'], bins=50, color='cyan', alpha=0.7, label="Total Tokens")
            ax.set_xlabel("Total Tokens")
            ax.set_ylabel("Frequency")
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.15, 1))
        else:
            ax.text(0.5, 0.5, "Total tokens not available", fontsize=12, ha='center')

        # Plot 2: Distribution of Action Probabilities (if available)
        if has_probas:
            ax = axes3[5]
            ax.set_title("Distribution of Action Confidence (1-5)", fontsize=12)
            ax.hist(llm_df['long_conf_score'], bins=50, color='green', alpha=0.7, label='Long Scores')
            ax.hist(llm_df['short_conf_score'], bins=50, color='red', alpha=0.7, label='Short Scores')
            ax.set_xlabel("Confidence Scores")
            ax.set_ylabel("Frequency")
            ax.legend(bbox_to_anchor=(1.15, 1))
            ax.grid(True)

        fig3.tight_layout()

    # Create figure for the second chart: LLM Trading Signals and Related Data
    fig2, axes2 = plt.subplots(8, 1, figsize=(16, 16), sharex=True)

    # Plot 2: Liquidity Metrics (Volume)
    ax = axes2[0]
    ax.set_title("Liquidity Metrics", fontsize=12)
    ax.bar(llm_df.index, llm_df['Volume'], label='Volume', color='gray', alpha=0.6)
    ax.set_ylabel("Volume")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.grid(True)

    # ATR
    ax = axes2[1]
    ax.set_title("Average True Range (ATR)", fontsize=14)
    if 'ATR' in llm_df.columns:
        ax.plot(llm_df.index, llm_df['ATR'], label='ATR', color='blue', linewidth=2.5)
    ax.set_ylabel("ATR")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.grid(True)

    # Relative Strength Index (RSI)
    ax = axes2[2]
    ax.set_title("Relative Strength Index (RSI)", fontsize=14)
    if 'RSI' in llm_df.columns:
        ax.plot(llm_df.index, llm_df['RSI'], label='RSI', color='blue', linewidth=2.5)
        ax.axhline(70, color='red', linestyle='--', linewidth=1, label='Overbought (70)')
        ax.axhline(30, color='green', linestyle='--', linewidth=1, label='Oversold (30)')
    ax.set_ylabel("RSI (0-100)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.grid(True)

    # MACD Indicator
    ax = axes2[3]
    ax.set_title("MACD", fontsize=14)
    if 'MACD' in llm_df.columns and 'Signal_Line' in llm_df.columns:
        ax.plot(llm_df.index, llm_df['MACD'], label='MACD', color='blue', linewidth=2.5)
        ax.plot(llm_df.index, llm_df['Signal_Line'], label='Signal Line', color='red', linewidth=2)
    ax.set_ylabel("MACD Value")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.grid(True)

    # Plot 4: Option Volatility
    ax = axes2[4]
    ax.set_title("Option Put Skew", fontsize=12)
    if 'ATM_Skew' in llm_df.columns and 'OTM_Skew' in llm_df.columns:
        ax.plot(llm_df.index, llm_df['ATM_Skew'], label='ATM Skew', linewidth=2.5, color='blue')
        ax.plot(llm_df.index, llm_df['OTM_Skew'], label='OTM Skew', linewidth=2.5, linestyle='--', color='red')
    ax.set_ylabel("IV Skew)")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=2, alpha=0.8, label="Skew Threshold")

    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.grid(True)

    ax = axes2[5]
    ax.set_title("Market Volatility & Beta", fontsize=12)
    ax2 = ax.twinx()
    if 'Market_Beta' in llm_df.columns:
        ax.plot(llm_df.index, llm_df['Market_Beta'], label='Market Beta', color='orange', linewidth=2.5)
    if 'VIX_Close' in llm_df.columns:
        ax2.plot(llm_df.index, llm_df['VIX_Close'], label='VIX', color='purple', linewidth=2.5, linestyle="--")
    ax.set_ylabel("Market Beta")
    ax2.set_ylabel("VIX Level")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax2.legend(loc="lower right", bbox_to_anchor=(1.15, 1))
    ax.grid(True)

    ax = axes2[6]
    ax.set_title("Valuation & Profitability", fontsize=12)
    valuation_qoq = ['Price_to_Book_Ratio_QoQ_Growth', 'PE_Ratio_QoQ_Growth', 'EPS_QoQ_Growth',
                    'Net_Income_QoQ_Growth', 'Free_Cash_Flow_Per_Share_QoQ_Growth']
    for metric in valuation_qoq:
        if metric in llm_df.columns:
            formatted_label = metric.replace("_QoQ_Growth", "").replace("_", " ")
            ax.plot(llm_df.index, llm_df[metric], label=formatted_label, linewidth=2)
    ax.set_ylabel("QoQ Growth (%)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.grid(True)

    ax = axes2[7]
    ax.set_title("Returns & Cash Flow", fontsize=12)
    returns_qoq = ['Operating_Cash_Flow_Per_Share_QoQ_Growth', 'Return_on_Equity_QoQ_Growth', 'Return_on_Assets_QoQ_Growth']
    for metric in returns_qoq:
        if metric in llm_df.columns:
            formatted_label = metric.replace("_QoQ_Growth", "").replace("_", " ")
            ax.plot(llm_df.index, llm_df[metric], label=formatted_label, linewidth=2)
    ax.set_ylabel("QoQ Growth (%)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.grid(True)
    fig2.tight_layout()
    if plot:
        plt.show()
    else:
        plt.close(fig1)
        if fig3 is not None:
            plt.close(fig3)
        plt.close(fig2)

    return fig1, fig3, fig2



def sanitize_text(text: str) -> str:
    """Convert HTML content into cleaned Markdown-like text."""

    if html2text is None:  # pragma: no cover - guarded by dependency injection
        raise RuntimeError("html2text is required for sanitize_text but is not installed.")

    handler = html2text.HTML2Text()
    handler.ignore_links = True
    handler.ignore_images = True
    handler.body_width = 0
    clean_text = handler.handle(text)
    return clean_text.strip()


def combine_news_by_month(news_df: DataFrame) -> DataFrame:
    """Sanitise article content and aggregate it into month-level batches."""

    tqdm.pandas(desc="Sanitizing text")
    df = news_df.copy()
    df["content"] = df["content"].progress_apply(sanitize_text)
    df["month_year"] = df.index.to_period("M")

    grouped_news: list[dict[str, Any]] = []
    for name, group in tqdm(df.groupby("month_year"), desc="Combining content by month"):
        grouped_news.append({"month_year": str(name), "content": group["content"].tolist()})

    return pd.DataFrame(grouped_news)


def prepare_yaml_with_articles(
    articles: Sequence[str],
    news_template: str,
    ticker: str,
    target_name: str,
    date: str,
) -> str:
    """Inject article bullet points into the news prompt template."""

    sanitized_articles = "\n".join(f"  - {article.strip()}" for article in articles if article and article.strip())
    filled_template = news_template.replace("{articles_list}", f"|\n{sanitized_articles}")
    filled_template = filled_template.replace("{ticker}", ticker).replace("{date}", date).replace("{company_name}", target_name)
    return filled_template


def process_news_with_llm(
    grouped_news: DataFrame,
    ticker: str,
    target_name: str,
    news_yml_file: Path | str,
    llm_client: Any,
    llm_model: str,
    LLM_OUTPUT_PATH: Path | str = "./data",
) -> list[dict[str, Any]]:
    """Run the news-cleaning workflow across pre-grouped monthly article batches."""

    if llm_client is None:
        raise ValueError("llm_client must be provided to process news with the LLM")

    results: list[dict[str, Any]] = []
    for _, row in tqdm(grouped_news.iterrows(), desc="Processing grouped news", total=len(grouped_news)):
        llm_result = call_openai_to_extract_news(
            articles=row["content"],
            news_yml_file=news_yml_file,
            ticker=ticker,
            target_name=target_name,
            date=row["month_year"],
            client=llm_client,
            model=llm_model,
            LLM_OUTPUT_PATH=LLM_OUTPUT_PATH,
        )
        results.append(
            {
                "date": row["month_year"],
                "factors": llm_result["response"],
                "tokens_meta": llm_result["tokens_meta"],
            }
        )

    return results