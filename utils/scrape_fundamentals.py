import re
import json
import numpy as np
import pandas as pd
import requests
import logging
import argparse
from bs4 import BeautifulSoup
import time
import random
import os
from glob import glob
from dotenv import load_dotenv

load_dotenv()

LOGS_PATH = os.getenv("LOGS_PATH", "./logs")
FUNDAMENTALS_PATH = os.getenv("FUNDAMENTALS_PATH", "./fundamentals")

paths = [
    FUNDAMENTALS_PATH,
    LOGS_PATH,
]

for path in paths:
    if path and not os.path.exists(path):
        os.makedirs(path)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    'macrotrends': {
        'financial_ratios': {
            'url_template': 'https://www.macrotrends.net/stocks/charts/{ticker}/{ticker_lower}/financial-ratios?freq=Q',
            'data_variable': 'originalData',
            'fields': [
                'Current Ratio', 'Long-term Debt / Capital', 'Gross Margin',
                'Operating Margin', 'EBIT Margin', 'EBITDA Margin', 'Pre-Tax Profit Margin', 'Net Profit Margin',
                'Asset Turnover', 'Inventory Turnover Ratio', 'Receivable Turnover', 'Days Sales In Receivables',
                'Book Value Per Share', 'Operating Cash Flow Per Share', 'Free Cash Flow Per Share'
            ]
        },
        'pe_ratio': {
            'url_template': 'https://www.macrotrends.net/stocks/charts/{ticker}/{ticker_lower}/pe-ratio?freq=Q',
            'data_variable': 'style-1',
            'fields': ['Date', 'Stock Price', 'TTM Net EPS', 'PE Ratio']
        },
        'ps_ratio': {
            'url_template': 'https://www.macrotrends.net/stocks/charts/{ticker}/{ticker_lower}/price-sales?freq=Q',
            'data_variable': 'style-1',
            'fields': ['Date', 'Stock Price', 'TTM Sales per Share', 'Price to Sales Ratio']
        },
        'price_book': {
            'url_template': 'https://www.macrotrends.net/stocks/charts/{ticker}/{ticker_lower}/price-book?freq=Q',
            'data_variable': 'style-1',
            'fields': ['Date', 'Stock Price', 'Book Value per Share', 'Price to Book Ratio']
        },
        'price_fcf': {
            'url_template': 'https://www.macrotrends.net/stocks/charts/{ticker}/{ticker_lower}/price-fcf?freq=Q',
            'data_variable': 'style-1',
            'fields': ['Date', 'Stock Price', 'TTM FCF per Share', 'Price to FCF Ratio']
        },
        'current_ratio': {
            'url_template': 'https://www.macrotrends.net/stocks/charts/{ticker}/{ticker_lower}/current-ratio?freq=Q',
            'data_variable': 'style-1',
            'fields': ['Date', 'Current Assets', 'Current Liabilities', 'Current Ratio']
        },
        'quick_ratio': {
            'url_template': 'https://www.macrotrends.net/stocks/charts/{ticker}/{ticker_lower}/quick-ratio?freq=Q',
            'data_variable': 'style-1',
            'fields': ['Date', 'Current Assets - Inventory', 'Current Liabilities', 'Quick Ratio']
        },
        'debt_equity_ratio': {
            'url_template': 'https://www.macrotrends.net/stocks/charts/{ticker}/{ticker_lower}/debt-equity-ratio?freq=Q',
            'data_variable': 'style-1',
            'fields': ['Date', 'Long Term Debt', "Shareholder's Equity", 'Debt to Equity Ratio']
        },
        'roe': {
            'url_template': 'https://www.macrotrends.net/stocks/charts/{ticker}/{ticker_lower}/roe?freq=Q',
            'data_variable': 'style-1',
            'fields': ['Date', 'Net Income', "Shareholder's Equity", 'Return on Equity']
        },
        'roa': {
            'url_template': 'https://www.macrotrends.net/stocks/charts/{ticker}/{ticker_lower}/roa?freq=Q',
            'data_variable': 'style-1',
            'fields': ['Date', 'Net Income', 'Total Assets', 'Return on Assets']
        },
        'roi': {
            'url_template': 'https://www.macrotrends.net/stocks/charts/{ticker}/{ticker_lower}/roi?freq=Q',
            'data_variable': 'style-1',
            'fields': ['Date', 'Net Income', 'Invested Capital', 'Return on Investment']
        },
        'eps': {
            'url_template': 'https://www.macrotrends.net/stocks/charts/{ticker}/{ticker_lower}/eps-earnings-per-share-diluted?freq=Q',
            'data_variable': 'style-1',
            'fields': ['Date', 'EPS']
        }
    }
}

def fetch_html_content(url, max_retries=5, backoff_factor=3.5):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.HTTPError as http_err:
            if response.status_code == 429:
                delay = backoff_factor * (2 ** attempt)
                logger.warning(f"Rate limit hit (429). Retrying in {delay:.2f} seconds.")
                time.sleep(delay)
            else:
                logger.error(f"HTTP error occurred: {http_err}")
                return None
        except Exception as err:
            logger.error(f"An error occurred: {err}")
            return None
    logger.error(f"Failed after {max_retries} retries.")
    return None

def parse_financial_data(html_content, data_variable, fields):
    match = re.search(rf"var {data_variable} = (\[.*?\]);", html_content, re.DOTALL)
    if not match:
        logger.error(f"Data variable '{data_variable}' not found in HTML content.")
        return None

    try:
        json_data = json.loads(match.group(1))
        data_dict = {}
        for entry in json_data:
            field_name = re.sub(r'<.*?>', '', entry['field_name']).strip()
            if field_name in fields:
                for date, value in entry.items():
                    if date not in ["field_name", "popup_icon"]:
                        if date not in data_dict:
                            data_dict[date] = {}
                        data_dict[date][field_name] = float(value) if value else np.nan

        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error parsing original data: {e}")
        return None

def parse_generic_table(html_content, fields):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        table_div = soup.find('div', {'id': 'style-1'})
        tables = table_div.find_all('table') if table_div else []
        if not tables:
            logger.error("Table not found in the HTML content.")
            return None
        table = tables[1] if len(tables) > 1 else tables[0]
        rows = table.find('tbody').find_all('tr') if table else []
        if not rows:
            logger.error("No data rows found in the table.")
            return None

        data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) != len(fields):
                continue
            row_data = []
            for idx, col in enumerate(cols):
                text = col.text.strip()
                if idx == 0:  # Handle Date separately
                    try:
                        row_data.append(pd.to_datetime(text, errors='coerce'))
                    except Exception as e:
                        logger.error(f"Error parsing date: {text} -> {e}")
                        row_data.append(None)
                else:
                    if text.endswith('%'):
                        if text.lower() == 'inf%':  # Handle inf%
                            row_data.append(np.inf)
                        else:
                            row_data.append(float(text.strip('%')) / 100)
                    else:
                        text = text.replace('$', '').replace('B', 'e9').replace('M', 'e6').replace(',', '')
                        row_data.append(pd.to_numeric(text, errors='coerce'))
            data.append(row_data)

        df = pd.DataFrame(data, columns=fields)
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        return df

    except Exception as e:
        logger.error(f"Error parsing table data: {e}")
        return None


def save_to_csv(df, output_path):
    if df is not None:
        df.to_csv(output_path, index=True)
        logger.info(f"Data saved to {output_path}")
    else:
        logger.error("No data to save.")

def aggregate_fundamentals(ticker, output_csv_path):
    csv_files = glob(f"{output_csv_path}/{ticker}-*.csv")
    if not csv_files:
        logger.error(f"No CSV files found in {output_csv_path}.")
        return None

    merged_df = None

    for csv_file in csv_files:
        logger.info(f"Processing {csv_file}...")
        # Load CSV with Date parsing
        df = pd.read_csv(csv_file, index_col='Date', parse_dates=['Date'])
        df.sort_index(inplace=True)  # Ensure Date is sorted

        # Handle duplicate columns by renaming
        if merged_df is None:
            merged_df = df
        else:
            # Drop duplicate columns
            overlapping_columns = set(merged_df.columns).intersection(df.columns)
            if overlapping_columns:
                logger.debug(f"Duplicate columns found: {overlapping_columns}. Dropping from new DataFrame.")
                df.drop(columns=overlapping_columns, inplace=True)
            merged_df = pd.merge_asof(merged_df, df, left_index=True, right_index=True, direction='forward')

    if merged_df is not None:
        # Forward-fill missing dates and data
        merged_df.ffill(inplace=True)
        merged_df.sort_index(inplace=True)

        # Remove columns with duplicate base names (e.g., those with suffixes)
        column_names = merged_df.columns
        unique_columns = {}
        for col in column_names:
            base_name = col.rsplit('_', 1)[0]  # Extract base name before any suffix
            if base_name not in unique_columns:
                unique_columns[base_name] = col
        merged_df = merged_df[unique_columns.values()]  # Retain only unique columns

        # Save the aggregated result
        output_file = f"{output_csv_path}/{ticker}-aggregated_fundamentals.csv"
        save_to_csv(merged_df, output_file)
        logger.info(f"Aggregated dataset saved to {output_file}.")
        return merged_df
    else:
        logger.error("No data to aggregate.")
        return None

def main(ticker, ticker_name, output_csv_path, site_key, page_type=None, aggregate=True, scrape=True):
    if site_key not in CONFIG:
        logger.error(f"Configuration for site '{site_key}' not found.")
        return

    page_types = [page_type] if page_type else CONFIG[site_key].keys()

    if scrape:
        for page in page_types:
            config = CONFIG[site_key][page]
            url = config['url_template'].format(ticker=ticker, ticker_lower=ticker_name.lower(), page=page)
            fields = config['fields']

            logger.info(f"Fetching data for ticker: {ticker} from site: {site_key}, page: {page}")
            html_content = fetch_html_content(url)

            if html_content:
                if page == 'financial_ratios':
                    df = parse_financial_data(html_content, config['data_variable'], fields)
                else:
                    df = parse_generic_table(html_content, fields)

                if df is not None:
                    output_file = f"{output_csv_path}/{ticker}-{page}.csv"
                    save_to_csv(df, output_file)
            else:
                logger.error(f"Failed to retrieve HTML content for page: {page}")

            delay = random.uniform(3, 15)
            logger.info(f"Delaying next request by {delay:.2f} seconds.")
            time.sleep(delay)

    if aggregate:
        logger.info("Aggregating all fundamental data into a single dataset.")
        aggregate_fundamentals(ticker, output_csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fetch and save financial data for a specified ticker and site.")

    parser.add_argument('--ticker', type=str, default='META', help="Stock ticker symbol (default: TSLA), e.g. AAPL, TSLA, AMZN, MSFT, GOOGL, META")
    parser.add_argument('--ticker_name', type=str, default='meta', help="Stock ticker symbol (default: tesla), e.g. apple, tesla, amazon, microsoft, google, meta")
    parser.add_argument('--output_csv_path', type=str, default=FUNDAMENTALS_PATH, help="Base path to save the CSV output (default: ./fundamentals)")
    parser.add_argument('--site_key', type=str, choices=CONFIG.keys(), default='macrotrends', help="Site key for data source configuration (default: macrotrends)")
    parser.add_argument('--page_type', type=str, help="Page type (e.g., financial_ratios). If not provided, iterates through all.")
    parser.add_argument('--aggregate', type=bool, default=True, help="Whether to aggregate all CSVs into a single dataset (default: True)")

    args = parser.parse_args()

    main(args.ticker, args.ticker_name,args.output_csv_path, args.site_key, args.page_type, args.aggregate)
