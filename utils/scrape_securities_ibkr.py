import os
import sys
import re
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from tqdm import tqdm
from ib_insync import *
import logging
import warnings
import argparse
from glob import glob
import yfinance as yf

# Load environment variables
load_dotenv()

# Constants and Environment Configuration
LOGS_PATH = os.getenv("LOGS_PATH", "./logs")
MACRO_PATH = os.getenv("MACRO_PATH", "./macro")
HISTORIC_PATH = os.getenv("HISTORIC_PATH", "./historic")

# Create directories if they do not exist
paths = [MACRO_PATH, LOGS_PATH, HISTORIC_PATH]
for path in paths:
    if path and not os.path.exists(path):
        os.makedirs(path)

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Stock Feature Definitions
class StockFeat:
    DATETIME = "Date"
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    CLOSE = "Close"
    VOLUME = "Volume"
    list = [OPEN, HIGH, LOW, CLOSE, VOLUME]
    SPREAD = "Spread"
    BARCOUNT = "Barcount"
    AVERAGE = "Average"

    # Volatility Features
    IV_FEATURES = [f"IV_{x}" for x in ["Open", "High", "Low", "Close", "Volume"]]
    RV_FEATURES = [f"RV_{x}" for x in ["Open", "High", "Low", "Close", "Volume"]]
    HV_FEATURES = [f"HV_{x}" for x in ["Open", "High", "Low", "Close", "Volume"]]

    ext_list = list + [SPREAD, BARCOUNT, AVERAGE] + IV_FEATURES + RV_FEATURES + HV_FEATURES

# Contract and Position Types
class ContractType:
    OPTION = "OPT"
    STOCK = "STK"
    INDEX = "IND"
    FUTURE = "FUT"
    CONT_FUTURE = "CONTFUT"
    CURRENCY = "FOREX"
    list = [STOCK, INDEX]

### US Markets
VOL_TICKER = "VIX"
MARKET_TICKER = "SPX"
DOW_TICKER = "INDU"
NASDAQ_TICKER = "NDX"
TYIELD_2YR = "IRX"
TYIELD_10YR = "TNX"
US_BOND_TICKER = "BND"

### US Sectors (based on SPDR Sector ETFs)
TECH_TICKER = "XLK"
HEALTH_TICKER = "XLV"
FINANCIAL_TICKER = "XLF"
ENERGY_TICKER = "XLE"
CONSUMER_STAPLES_TICKER = "XLP"
CONSUMER_DISCRETIONARY_TICKER = "XLY"
INDUSTRIAL_TICKER = "XLI"
MATERIALS_TICKER = "XLB"
REAL_ESTATE_TICKER = "XLRE"
UTILITIES_TICKER = "XLU"
SMALLCAP_TICKER = "IWM"

### Global Commodities
GOLD_TICKER = "GLD"
OIL_TICKER = "DBO"
NATURAL_GAS_TICKER = "BOIL"
SILVER_TICKER = "SLV"

### Global Equity Benchmarks
MSCI_WORLD_TICKER = "IWDA"

### Final Ticker List
MACRO_TICKERS = [
    # US Markets
    (VOL_TICKER, ContractType.INDEX, "CBOE", "USD"),
    (MARKET_TICKER, ContractType.INDEX, "CBOE", "USD"),
    (DOW_TICKER, ContractType.INDEX, "CME", "USD"),
    (NASDAQ_TICKER, ContractType.INDEX, "NASDAQ", "USD"),
    (TYIELD_2YR, ContractType.INDEX, "CBOE", "USD"),
    (TYIELD_10YR, ContractType.INDEX, "CBOE", "USD"),

    # US Sectors
    (TECH_TICKER, ContractType.STOCK, "NYSE", "USD"),
    (HEALTH_TICKER, ContractType.STOCK, "NYSE", "USD"),
    (FINANCIAL_TICKER, ContractType.STOCK, "NYSE", "USD"),
    (ENERGY_TICKER, ContractType.STOCK, "NYSE", "USD"),
    (CONSUMER_STAPLES_TICKER, ContractType.STOCK, "NYSE", "USD"),
    (CONSUMER_DISCRETIONARY_TICKER, ContractType.STOCK, "NYSE", "USD"),
    (INDUSTRIAL_TICKER, ContractType.STOCK, "NYSE", "USD"),
    (MATERIALS_TICKER, ContractType.STOCK, "NYSE", "USD"),
    (REAL_ESTATE_TICKER, ContractType.STOCK, "NYSE", "USD"),
    (UTILITIES_TICKER, ContractType.STOCK, "NYSE", "USD"),
    (SMALLCAP_TICKER, ContractType.STOCK, "ARCA", "USD"),

    # Commodities
    (GOLD_TICKER, ContractType.STOCK, "ARCA", "USD"),
    (OIL_TICKER, ContractType.STOCK, "ARCA", "USD"),
    (NATURAL_GAS_TICKER, ContractType.STOCK, "ARCA", "USD"),
    (SILVER_TICKER, ContractType.STOCK, "ARCA", "USD"),

    (US_BOND_TICKER, ContractType.STOCK, "NASDAQ", "USD"),

    (MSCI_WORLD_TICKER, ContractType.STOCK, "AEB", "EUR"),
]

STOCK_TICKERS = [
    # Major Market Indices
    ('DIA', ContractType.STOCK, 'NYSE', 'USD'),  # Dow Jones
    ('SPY', ContractType.STOCK, 'NYSE', 'USD'),  # S&P 500
    ('QQQ', ContractType.STOCK, 'NASDAQ', 'USD'),  # NASDAQ 100
    ('EZU', ContractType.STOCK, 'NYSE', 'USD'),  # FTSE 100
    ('EWJ', ContractType.STOCK, 'NYSE', 'USD'),  # Nikkei 225

    # Major Tech Stocks
    ('GOOGL', ContractType.STOCK, 'NASDAQ', 'USD'),  # Google
    ('AAPL', ContractType.STOCK, 'NASDAQ', 'USD'),  # Apple
    ('META', ContractType.STOCK, 'SMART', 'USD'),  # Meta
    ('AMZN', ContractType.STOCK, 'NASDAQ', 'USD'),  # Amazon
    ('MSFT', ContractType.STOCK, 'NASDAQ', 'USD'),  # Microsoft
    # ('TWTR', ContractType.STOCK, 'VALUE', 'USD'),  # Twitter

    # Telecom and Electronics
    ('NOK', ContractType.STOCK, 'NYSE', 'USD'),  # Nokia
    # ('PHIA.AS', ContractType.STOCK, 'AEB', 'EUR'),  # Philips
    # ('SIE.DE', ContractType.STOCK, 'XETRA', 'EUR'),  # Siemens
    # ('6758.T', ContractType.STOCK, 'TSE', 'JPY'),  # Sony

    # Chinese Tech Giants
    ('BIDU', ContractType.STOCK, 'NASDAQ', 'USD'),  # Baidu
    ('BABA', ContractType.STOCK, 'NYSE', 'USD'),  # Alibaba
    # ('0700.HK', ContractType.STOCK, 'HKEX', 'HKD'),  # Tencent

    # Banking
    ('JPM', ContractType.STOCK, 'NYSE', 'USD'),  # JPMorgan Chase
    ('HSBC', ContractType.STOCK, 'NYSE', 'USD'),  # HSBC
    # ('0939.HK', ContractType.STOCK, 'HKEX', 'HKD'),  # CCB

    # Energy
    ('XOM', ContractType.STOCK, 'NYSE', 'USD'),  # ExxonMobil
    # ('SHELL', ContractType.STOCK, 'AEB', 'EUR'),  # Shell
    # ('0857', ContractType.STOCK, 'SEHK', 'HKD'),  # PetroChina

    # Automotive
    ('TSLA', ContractType.STOCK, 'NASDAQ', 'USD'),  # Tesla
    # ('VOW3.DE', ContractType.STOCK, 'XETRA', 'EUR'),  # Volkswagen
    # ('7203', ContractType.STOCK, 'TSEJ', 'JPY'),  # Toyota

    # Consumer Staples and Beverages
    ('KO', ContractType.STOCK, 'NYSE', 'USD'),  # Coca Cola
    # ('ABI', ContractType.STOCK, 'ENEXT.BE', 'EUR'),  # AB InBev
    # ('2503', ContractType.STOCK, 'TSEJ', 'JPY'),  # Kirin
]


# We pad these dates for analysis and rolling averages
START_DATE = '20120101'
PAD_START_DATE = '20100101'
END_DATE = '20200101'
PAD_END_DATE = '20210101'

start_date = datetime.strptime(PAD_START_DATE, '%Y%m%d')
end_date = datetime.strptime(END_DATE, '%Y%m%d')
MAX_TRADING_DAYS = (end_date - start_date).days
INTERVAL = "1 day"

END_DATE = (datetime.strptime(PAD_END_DATE, '%Y%m%d') + timedelta(days=0)).strftime('%Y%m%d %H:%M:%S')

# Helper decorator to catch IB errors related to a specific contract
def catch_ib_error_on_contract(ib, contract_id_key: str, error_code: int):
    def _raise_ib_error_on_contract(func):
        async def wrapper(*args, **kwargs):
            _errors = []

            def _error_handler(reqId, code, msg, contract):
                _errors.append((reqId, code, msg, contract))
            ib.errorEvent += _error_handler
            try:
                resp = await func(*args, **kwargs)
                for error in _errors:
                    if int(error[1]) == error_code and int(error[3].conId) == int(kwargs[contract_id_key]):
                        raise Exception(f"IB Error: {error[2]} for contract {error[3]}")
                return resp
            finally:
                ib.errorEvent -= _error_handler

        return wrapper
    return _raise_ib_error_on_contract

def get_ib_tickers(ib=None,
                    tickers_info=[],
                    duration_days=1,
                    end=None,
                    interval="1d",
                    datadir="./data",
                    useRTH=True):
    def convert_days_to_yf_period(duration_days):
        if duration_days <= 7:
            return "1d"
        elif duration_days <= 30:
            return "1mo"
        elif duration_days <= 90:
            return "3mo"
        elif duration_days <= 180:
            return "6mo"
        elif duration_days <= 365:
            return "1y"
        elif duration_days <= 730:
            return "2y"
        elif duration_days <= 1825:
            return "5y"
        elif duration_days <= 3650:
            return "10y"
        else:
            return "max"
    def convert_ib_to_yf_interval(ib_interval):
        ib_to_yf_map = {
            "1 min": "1m",
            "2 mins": "2m",
            "5 mins": "5m",
            "15 mins": "15m",
            "30 mins": "30m",
            "1 hour": "1h",
            "1 day": "1d",
            "5 days": "5d",
            "1 week": "1wk",
            "1 month": "1mo",
            "3 months": "3mo"
        }
        return ib_to_yf_map.get(ib_interval.lower(), None)

    def error_handler(reqId, errorCode, errorMsg, contract):
        """
        Handles broker errors and logs the details.
        """
        if reqId > 0:
            logging.error(f"Broker Error - ReqID: {reqId}, Code: {errorCode}, Msg: {errorMsg}, Contract: {contract}")

    assert ib is not None
    ib.errorEvent += error_handler
    if end is None:
        end = datetime.now().strftime('%Y%m%d-%H:%M:%S')
    elif isinstance(end, datetime):
        end = end.strftime('%Y%m%d-%H:%M:%S')

    tickers = {}
    os.makedirs(datadir, exist_ok=True)

    for ticker in tickers_info:
        symbol = ticker[0] if len(ticker) > 0 else None
        ticker_type = ticker[1] if len(ticker) > 1 else None
        exchange = ticker[2] if len(ticker) > 2 else None
        currency = ticker[3] if len(ticker) > 3 else None
        strike = ticker[4] if len(ticker) > 4 else None
        right = ticker[5] if len(ticker) > 5 else None
        lastTradeDateOrContractMonth = ticker[6] if len(ticker) > 6 else None

        if ticker_type == ContractType.OPTION:
            cached_file_path = f"{datadir}/{symbol}-{ticker_type}-{lastTradeDateOrContractMonth}-{strike}{right}-{duration_days}-{end[:8]}-{interval}.csv"
        else:
            cached_file_path = f"{datadir}/{symbol}-{ticker_type}-{duration_days}-{end[:8]}-{interval}.csv"

        try:
            if os.path.exists(cached_file_path):
                df = pd.read_csv(cached_file_path, parse_dates=['Date'])
                df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert('CET')
                assert len(df) > 0, "Empty data"
            elif ib is not None:
                # Initialize the contract
                if ticker_type == ContractType.INDEX:
                    contract = Index(symbol=symbol, exchange=exchange, currency=currency)
                elif ticker_type == ContractType.CONT_FUTURE:
                    contract = ContFuture(symbol=symbol, exchange=exchange, currency=currency)
                elif ticker_type == ContractType.STOCK:
                    contract = Stock(symbol=symbol, exchange=exchange, currency=currency)
                elif ticker_type == ContractType.OPTION:
                    contract = Option(symbol=symbol,
                                      strike=strike,
                                      right=right,
                                      exchange=exchange,
                                      multiplier=100,
                                      currency=currency,
                                      lastTradeDateOrContractMonth=lastTradeDateOrContractMonth,
                                      includeExpired=True)
                else:
                    raise Exception(f"Invalid type: {ticker_type}")

                q_contract = ib.qualifyContracts(contract)
                if len(q_contract) == 0:
                    logging.warning(f"No data qualified for {symbol}")
                    raise Exception("IB Failure to recognise ticker!")
                ib.sleep(2)

                duration_str = f"{duration_days // 365} Y" if duration_days > 365 else f"{duration_days} D"

                df_trades = ib.reqHistoricalData(
                    contract,
                    endDateTime=end,
                    durationStr=duration_str,
                    barSizeSetting=interval,
                    whatToShow='MIDPOINT',
                    useRTH=useRTH,
                    timeout=25,
                )
                ib.sleep(2)
                df_trades = util.df(df_trades)
                if df_trades is None:
                    raise Exception("IB Failure to Retrieve Data!")
                df_trades.rename(columns=lambda x: x.capitalize(), inplace=True)
                try:
                    df_trades['Date'] = pd.to_datetime(df_trades['Date']).dt.tz_convert('CET')
                except:
                    df_trades['Date'] = pd.to_datetime(df_trades['Date']).dt.tz_localize('CET')

                if ticker_type in [ContractType.INDEX, ContractType.STOCK]:
                    df_iv = ib.reqHistoricalData(
                        q_contract[0],
                        endDateTime=end,
                        durationStr=duration_str,
                        barSizeSetting=interval,
                        whatToShow='OPTION_IMPLIED_VOLATILITY',
                        useRTH=useRTH
                    )
                    ib.sleep(2)
                    df_iv = util.df(df_iv)
                    if df_iv is not None:
                        df_iv.rename(columns=lambda x: f"IV_{x.capitalize()}" if x != 'date' else 'Date', inplace=True)
                        try:
                            df_iv['Date'] = pd.to_datetime(df_iv['Date']).dt.tz_convert('CET')
                        except:
                            df_iv['Date'] = pd.to_datetime(df_iv['Date']).dt.tz_localize('CET')
                        df_trades = pd.merge(df_trades, df_iv, on='Date', how='left')

                    df_hv = ib.reqHistoricalData(
                        q_contract[0],
                        endDateTime=end,
                        durationStr=duration_str,
                        barSizeSetting=interval,
                        whatToShow='HISTORICAL_VOLATILITY',
                        useRTH=useRTH
                    )
                    ib.sleep(2)
                    df_hv = util.df(df_hv)

                    if df_hv is not None:
                        df_hv.rename(columns=lambda x: f"HV_{x.capitalize()}" if x != 'date' else 'Date', inplace=True)
                        try:
                            df_hv['Date'] = pd.to_datetime(df_hv['Date']).dt.tz_convert('CET')
                        except:
                            df_hv['Date'] = pd.to_datetime(df_hv['Date']).dt.tz_localize('CET')
                        df_trades = pd.merge(df_trades, df_hv, on='Date', how='left')

                df = df_trades.loc[:, ~df_trades.columns.duplicated()]
            else:
                raise Exception("IB not available and cached file not found.")
        except Exception as e:
            logging.warning(f"IB retrieval failed for {symbol}: {e}. Switching to YFinance.")
            try:
                yf_period = convert_days_to_yf_period(duration_days)
                yf_interval=convert_ib_to_yf_interval(interval)
                yf_data = yf.download(symbol, period=yf_period, interval=yf_interval, timeout=30)

                yf_data.reset_index(inplace=True)
                yf_data['Date'] = pd.to_datetime(yf_data['Date'], utc=True).dt.tz_convert('CET')
                yf_data = yf_data[[StockFeat.DATETIME] + StockFeat.list]
                for col in ['IV_Open', 'IV_Close', 'IV_High', 'IV_Low', 'HV_Open', 'HV_Close', 'HV_High', 'HV_Low']:
                    yf_data[col] = None
                df = yf_data
            except Exception as yf_error:
                logging.error(f"YFinance retrieval failed for {symbol}: {yf_error}.")
                continue

        df.drop_duplicates(subset=['Date'], inplace=True)
        df.set_index('Date', inplace=True)

        if ticker_type == ContractType.OPTION:
            tickers[f"{symbol}-{lastTradeDateOrContractMonth}-{strike}{right}"] = df.ffill().bfill()
        else:
            tickers[symbol] = df.ffill().bfill()

        if not os.path.exists(cached_file_path):
            df.to_csv(cached_file_path)

    ib.errorEvent -= error_handler
    return tickers

def main():
    logging.info(f"Starting the script to pull financial data from {START_DATE} to {END_DATE} of {MAX_TRADING_DAYS} days.")
    util.logToFile(path=f"{LOGS_PATH}/ib.log", level=logging.ERROR)
    ib = IB()
    ib.RequestTimeout=60
    ib.RaiseRequestErrors=True

    try:
        ib.connect(os.getenv("TWS_HOST"), int(os.getenv("TWS_PORT")), clientId=int(os.getenv("TWS_CLIENT_ID")))
        ib.reqMarketDataType(2)
        tickers = get_ib_tickers(
            ib=ib,
            tickers_info=STOCK_TICKERS,
            duration_days=MAX_TRADING_DAYS,
            end=END_DATE,
            interval=INTERVAL,
            datadir=HISTORIC_PATH
        )

        if tickers:
            logging.info("Data fetching completed.")
        else:
            logging.warning("No data fetched.")
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if ib.isConnected():
            ib.disconnect()
            logging.info("Disconnected from Interactive Brokers.")

if __name__ == "__main__":
    main()
