# LLM-Guided Strategy Synthesis for Deep Reinforcement Learning in Trading

<p align="center">
  <img src="./LLM_RL_Banner.png" alt="LLM and RL Agentic Systems Banner" width="600"/>
</p>

This research explores the use of Large Language Models (LLMs) as financial strategists guiding smaller RL agents. 

The approach replicates and extends the Deep Q-Network (DQN) trading system by Thibaut Theate and Damien Ernst (2020) using LLM-generated strategies under different risk profiles.

---

## Thesis Contributions

- A structured prompt design for financial strategy generation.
- An iterative Writer–Judge refinement loop for prompt evaluation and critique.
- LLM-generated signals backtested on historical stock windows with regret-based selection.
- Integration into a reinforcement learning framework (DQN) for end-to-end comparison.

---

## Project Structure

```plaintext
├── data/                       # Historical stock, macro, options, fundamentals
├── utils/                      # Utility modules (scrapers, RL wrappers, etc.)
├── judge_reviews/              # LLM-based critique outputs
├── runs/                       # Execution logs and results
├── .env                        # Environment variable file (API keys, paths)
├── environment.yml             # Conda environment for reproducibility
├── requirements.txt            # Python packages (PyPI only)
├── final_summary.csv           # Final result summary
└── *.ipynb                     # Experiment notebooks
```

# Required Environment

## Install Conda Environment

```bash
conda env create -f environment.yml
conda activate llmtrading
```

If you get a kernel error (TypeAlias), run:

```bash
pip install --upgrade typing_extensions
```

# Setup Your .env File

Create a .env file at the root of the project and configure the following:

```js
LLM_OUTPUT_PATH=./data/llm_data
LOGS_PATH=./logs
LLM_PROMPTS_PATH=./data/prompts
FUNDAMENTALS_PATH=./data/fundamentals
HISTORIC_PATH=./data/historic
MACRO_PATH=./data/macro
OPTIONS_PATH=./data/options
MODELS_PATH=./data/models

OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_key

LOG_LEVEL=INFO

TWS_PORT=7497
TWS_CLIENT_ID=22
TWS_HOST=127.0.0.1
TWS_DATA_PATH=./data
```

# Execution Order

Run the following notebooks in sequence:

1. `DATA_Engineering.ipynb`: Preprocess raw stock, macroeconomic, options, and fundamental data into structured features. Assume you have scraped the data.
2. `EXPERIMENT1_LLM_Baseline_Prompt_EDA.ipynb`: Explore initial prompt templates and run LLMs to produce baseline strategies.
3. `EXPERIMENT1_LLM_Prompt_Refinement.ipynb`: Use the Writer–Judge loop to generate refined prompts and track regret.
4. `EXPERIMENT1_LLM_Test_Strategies.ipynb`: Backtest LLM-generated strategies across rolling time windows and compute returns.
5. `EXPERIMENT2_LLM4RL.ipynb`: Inject LLM strategies into the DRL agent architecture and train on historical data.
6. `EXPERIMENT2_LLM4RL_Test_Bench.ipynb`: Run out-of-sample tests, compare DRL vs LLM+DRL vs baseline DQN, and compute metrics.

If you wish to replicate the benchmark, run BENCHMARK `Test_Bench_Replication.ipynb`

## Utilities for Scraping Data

The project includes several Python utilities to fetch and prepare financial datasets used for LLM-driven strategy generation. These scripts are modular and can be run independently or scheduled for regular updates.

### 1. `scrape_securities_ibkr.py`

This utility connects to the Interactive Brokers API and retrieves metadata and basic information for available US equities. It uses `ib_insync` for efficient interaction with the IBKR TWS gateway. Outputs are saved in CSV format within the `./data/tickers` directory and include information such as sector, industry, and exchange listings.

Use this to refresh the list of eligible stocks for backtesting or LLM strategy generation.

---

### 2. `scrape_us_economic_indicators.py`

This script aggregates macroeconomic indicators from various sources:

- **FRED** (via API): GDP, Retail Sales, Housing Starts, Yield Curve, Consumer Confidence, Treasury Yields
- **Investing.com** (scraped): PMI Index
- **BLS** (via API): Employment (Non-farm Payrolls)

Each dataset is saved to the `./data/macro/us` directory. You must set environment variables `FRED` and `BLS` to valid API keys in your `.env` file for this to work.

Run:
```bash
python scrape_us_economic_indicators.py
```
### 3. scrape_fundamentals.py

This script scrapes detailed quarterly fundamentals for a given ticker from macrotrends.net. It supports downloading and aggregating data across multiple categories:

- Financial Ratios (e.g., Margins, Liquidity)
- Valuation Multiples (PE, PS, PB, FCF)
- Return Ratios (ROE, ROA, ROI)
- Per-share Metrics (EPS, BVPS, FCF/Share)
- Run for a specific ticker:

```bash
python scrape_fundamentals.py --ticker META --ticker_name meta
```

By default, it scrapes all supported pages and aggregates them into a single CSV under ./data/fundamentals.

These scripts are critical for creating the dataset used by the LLM strategy engine.
You are encouraged to get the subscriptions and API keys and run them to keep macro and fundamental data fresh.

# Windows WSL Setup (Optional)
If you're using Windows, enable WSL and run:

```bash
wsl --install
```

Then open the VSCode terminal and run:

```bash
sudo apt-get update
sudo apt-get install wget ca-certificates
```
Launch your VSCode session via code . from WSL. Use the WSL extension for full IDE integration.





