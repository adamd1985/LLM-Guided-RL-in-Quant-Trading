# LLM Agent's Handbook

An agent will read this MD and follow the section `Agent's Tasks`


# Project Layout

| Path | Purpose |
| --- | --- |
| `/` | Root dir for research python notebooks, assume repo path is set in env. |
| `utils/` | Dir for all utilities, algo, signals, and other code. |
| `data/` | Dir for data API or WEB scrapers and DB access. |
| `configs/` | YAML strategy and broker configs. |
| `tests/` | Pytest suite mirroring `utils/`. |

## Environment & Toolchain

- **Python**: 3.12, managed via Poetry.
- **Primary commands**:
  - `poetry install --with dev,types --no-interaction`
  - `poetry run ci`
  - `poetry run pytest`
  - `poetry run ruff check .`
  - `poetry run mypy`
- **Static analysis**: Ruff (rules `E,F,I,B,UP,N,S,C90`), Mypy (strict), Bandit, Pip-audit.

### Type Checking

- We pin Python 3.12: ensure `language_version: python3.12` for the mypy hook to keep feature parity.
- Prefer classic alias assignments (`Foo = dict[str, int]`) over the PEP 695 `type Foo = ...` form for now to avoid requiring experimental flags in isolated hook environments.
- If you introduce PEP 695 syntax, you MUST add `--enable-incomplete-feature=NewGenericSyntax` to the mypy invocation everywhere (CI + pre-commit) and justify in the PR.
- Keep aliases narrow and semantic (e.g., `TradeID = int | str | None`). Avoid aliasing to `Any`.

### Troubleshooting (Mypy / Ruff)

| Symptom | Likely Cause | Fix |
| ------- | ------------ | ---- |
| `PEP 695 type aliases are not yet supported` | Pre-commit mypy lacks feature flag | Revert to classic alias or add `--enable-incomplete-feature=NewGenericSyntax` |
| `Duplicate module named "agents"` | Path confusion / shadowing | Ensure only one `src/agents/__init__.py` and run with `pythonpath=src` (already configured) |
| Plugin import error (`pydantic.mypy`) | Missing dependency in hook env | Add to `additional_dependencies` for the mypy hook |
| Ruff UP035/UP006 warnings | Legacy `typing.Dict/List/Tuple` use | Switch to builtin generics |
| `Cannot find implementation or library stub for module "xyz"` | Third-party package lacks types in hook env | Add published types package (e.g. `types-PyYAML`) or add runtime dep to mypy hook `additional_dependencies`; create a local `typed_stubs/xyz.pyi` if no stubs exist |

## Coding Standards

- Always use type annotations, modern py3.13 (internet search if your're unsure)
- Prefer pure functions and dataclasses unless runtime validation and streaming is required (then use Pydantic models).
- Keep async code idiomatic: prefer `asyncio.create_task`, `async with`, and handle cancellation gracefully.
- Use absolute imports within the `traderlib` tree (e.g., `from traderlib.eda import portfolio_metrics`).
- Adhere to the 120 character line length enforced by Ruff.
- If you're creating a new function, always check others, e.g. `utils` or `eda`, if function already exist and can be reused. For e.g. performance metrics:`evaluate_portfolio_metrics`
- when you create a new public function or non-data class (which is functional, not printing, or datacleaning, or plotting), start with a test.
- All dates must be TZ-aware and UTC, enforce this.

## Workflow for Automation Agents

1. Check the repo status with `git status -sb` before editing to avoid overwriting in-flight code.
2. **Assess context**: open relevant files with `Get-Content`, search using `rg`, inspect configs/tests before editing, or create a new one in `notebooks/` for research, or `traderlib/` for code and utilities.
3. **Plan changes**: outline the approach in your response when non-trivial.
4. **Edit files** using `Set-Content`/`Add-Content` (PowerShell) or helper scripts; ensure ASCII encoding.
5. **Install or uninstall dependencies** only through Poetry `add` or `remove`.
6. **Run checks**: `poetry run ci` whenever feasible; otherwise run targeted commands and mention the rationale if you skip.
7. **Summarise results** with file+line references in the final message.
8. **Never revert user edits** you did not make. If merged changes conflict, stop and ask for guidance.
9. **Document** always document new classes and new files, use `docstr` standards and check it with `pydocstyle`. Explain the function, its use with examples, returns, params, and maths involved. Less documentation for private functions and classes.
10. Your task is ready when the test passes after you create and lint the code.

## Testing Strategy

- Unit tests belong under `tests/` following the module structure (e.g., `tests/agents/strategies/test_pivots.py`).
- Mock broker adapters with `pytest` fixtures and `AsyncMock` or stub connectors.
- Integration tests may use paper-trading configs; store fixtures under `configs/` or `tests/fixtures`.
- Ensure new agents have at least one regression test covering their main execution loop.

## Research Checklist

1. Launch the environment with Poetry: `poetry shell` for an interactive session or prefix commands with `poetry run`.
2. Use `poetry run jupyter lab` (preferred) or `poetry run jupyter notebook` to work with notebooks; shut down kernels from the UI when finished.
3. Capture findings inline with markdown headings, math (LaTeX blocks), and linked visuals so notebooks stay reproducible and shareable.
4. No need to set repo path, assume pypath is set in the environment.

## Signals and Algos

1. When designing or asked to improve, access the internet and find strong literature on the algo or signal we are designing.
2. Every new algorithm needs to have:
   1. signal
   2. backtest
   3. gridsearch
   4. plotting for diagnostics.
3. In backtests, return the signals, performace metrics, and the strategy history. Important info:
   1. "timestamp"
   2. "symbol" - if its a more than 1 instrument.
   3. "action" - BUY or SELL
   4. "side" - LONG or SHORT
   5. "shares" - if viable, the amount of instruments bought, else just 1 for a signal.
   6. "notional" - When leverage is used.
   7. "signal"-  +1 or -1 or depends on the algo implemented.

# Tooling Reference

- **Environment setup**: `poetry install --with dev,types --no-interaction`
- **Notebook server**: `poetry run jupyter lab` (rich UI) or `poetry run jupyter notebook` (classic). Both commands expose keyboard shortcuts like `Shift+Enter` to run a cell and `0,0` to restart the kernel (see the [Jupyter Notebook docs](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html) for the full list).
- **Kernel management**: interrupt long jobs with the Kernel → Interrupt menu or the `i,i` shortcut; restart with Kernel → Restart or `0,0`.
- **Notebook quality**: `poetry run nbqa ruff notebooks/` for linting, `poetry run nbqa black notebooks/` for formatting, `poetry run nbconvert --to html your_notebook.ipynb` to export reports.
- **Data inspection**: prefer `pandas` profiling helpers from `traderlib.ml_utils` and plotting utilities in `traderlib.simulations` to keep analyses consistent.

# Notebook Workflow

- Start from topic folders (`notebooks/factors`, `notebooks/trade_signals`, etc.) to keep history organised.
- Use descriptive filenames (`factor_universe_dm.ipynb`) and document hypotheses in Cell 1 as markdown.
- Pull helpers via `from traderlib import utils, simulations` and `from traderlib.eda import ...` before re-creating common EDA logic.
- Cache intermediate datasets under `data/<domain>/scratch/` and version them with timestamps if reruns are required.
- For collaboration, export HTML summaries (`nbconvert`) and commit both the `.ipynb` and HTML to make peer review easier.

## Validation & Quality

- Run targeted tests when notebooks add reusable code: `poetry run pytest traderlib/tests -k <keyword>`.
- Use `poetry run ruff check traderlib` before committing new utility functions.
- If notebooks depend on fresh data pulls, log the source script and command (e.g., `poetry run python scripts/extract_earnings.py --symbol AAPL`).
- Keep outputs trustworthy by re-running all cells (`Kernel → Restart & Run All`) before finalising a notebook.