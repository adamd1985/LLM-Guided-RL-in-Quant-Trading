import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
FRED_API_KEY = os.getenv("FRED")
BLS_API_KEY = os.getenv("BLS")
MACRO_DATA_PATH = os.getenv("DATA_PATH")
MACRO_DATA_PATH = f"{MACRO_DATA_PATH}/macro/us"
os.makedirs(MACRO_DATA_PATH, exist_ok=True)

# Set constants
START_YEAR = 2010
END_YEAR = 2024

# Helper functions
def fetch_gdp():
    """Fetch GDP data from FRED."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "GDP",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": f"{START_YEAR}-01-01",
        "observation_end": f"{END_YEAR}-12-31",
    }
    response = requests.get(url, params=params)
    data = response.json()
    return pd.DataFrame(data['observations'])

def fetch_pmi():
    """Fetch Manufacturing PMI data from Investing.com."""
    url = "https://sbcharts.investing.com/events_charts/us/829.json"
    response = requests.get(url)

    # Check for errors
    if response.status_code != 200:
        raise ValueError(f"Error fetching PMI data: {response.status_code}, {response.text}")

    data = response.json()

    # Parse the JSON into a DataFrame
    pmi_data = []
    for entry in data['attr']:
        pmi_data.append({
            "release_date": datetime.utcfromtimestamp(entry['timestamp'] / 1000).strftime('%Y-%m-%d'),
            "actual": entry['actual'],
            "forecast": entry['forecast'] if entry['forecast'] is not None else entry['actual'],
            "previous": entry.get('previous'),
            "actual_state": entry.get('actual_state', 'neutral')
        })

    return pd.DataFrame(pmi_data)

def fetch_employment():
    """Fetch employment data (e.g., non-farm payrolls) from BLS."""
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"  # Example BLS API
    headers = {"Content-Type": "application/json"}
    data = {
        "seriesid": ["CES0000000001"],  # Non-farm payrolls
        "startyear": START_YEAR,
        "endyear": END_YEAR,
        "registrationkey": BLS_API_KEY
    }
    response = requests.post(url, json=data, headers=headers)
    data = response.json()
    observations = data['Results']['series'][0]['data']
    return pd.DataFrame(observations)

def fetch_retail_sales():
    """Fetch retail sales data from FRED."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "RSAFSNA",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": f"{START_YEAR}-01-01",
        "observation_end": f"{END_YEAR}-12-31",
    }
    response = requests.get(url, params=params)
    data = response.json()
    return pd.DataFrame(data['observations'])

def fetch_housing_starts():
    """Fetch housing starts and permits from FRED."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "HOUST",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": f"{START_YEAR}-01-01",
        "observation_end": f"{END_YEAR}-12-31",
    }
    response = requests.get(url, params=params)
    data = response.json()
    return pd.DataFrame(data['observations'])

def fetch_yield_curve():
    """Fetch yield curve data from FRED."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "T10Y2Y",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": f"{START_YEAR}-01-01",
        "observation_end": f"{END_YEAR}-12-31",
    }
    response = requests.get(url, params=params)
    data = response.json()
    return pd.DataFrame(data['observations'])

def fetch_consumer_confidence():
    """Fetch Consumer Confidence Index (CCI) from FRED."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "CSCICP03USM665S",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": f"{START_YEAR}-01-01",
        "observation_end": f"{END_YEAR}-12-31",
    }
    response = requests.get(url, params=params)
    data = response.json()
    return pd.DataFrame(data['observations'])

def fetch_commodity_prices():
    """Fetch commodity prices (e.g., oil, gold) from FRED."""
    commodities = {
        "Gold": "GOLDPMGBD228NLBM",
        "Crude Oil": "DCOILWTICO"
    }
    dfs = {}
    for name, series_id in commodities.items():
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": f"{START_YEAR}-01-01",
            "observation_end": f"{END_YEAR}-12-31",
        }
        response = requests.get(url, params=params)
        data = response.json()
        dfs[name] = pd.DataFrame(data['observations'])
    return dfs

def fetch_treasury_yields():
    """Fetch Treasury yields from FRED."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "GS10",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": f"{START_YEAR}-01-01",
        "observation_end": f"{END_YEAR}-12-31",
    }
    response = requests.get(url, params=params)
    data = response.json()
    return pd.DataFrame(data['observations'])

def main():
    datasets = {
        "GDP": fetch_gdp(),
        "PMI": fetch_pmi(),
        "Employment": fetch_employment(),
        "Retail Sales": fetch_retail_sales(),
        "Housing Starts": fetch_housing_starts(),
        "Yield Curve": fetch_yield_curve(),
        "Consumer Confidence": fetch_consumer_confidence(),
        "Treasury Yields": fetch_treasury_yields(),
    }

    for name, df in datasets.items():
        if isinstance(df, dict):
            for sub_name, sub_df in df.items():
                sub_df.to_csv(f"./{MACRO_DATA_PATH}/{name}_{sub_name}_2010_2021.csv", index=False)
        else:
            df.to_csv(f"./{MACRO_DATA_PATH}/{name}_2010_2021.csv", index=False)
        print(f"Saved {name} data to {MACRO_DATA_PATH}")

if __name__ == "__main__":
    main()
