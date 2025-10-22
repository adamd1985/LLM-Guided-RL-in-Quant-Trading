from __future__ import annotations

# ruff: noqa: S101

import math
import pickle  # noqa: S403,B403  # nosec: tests load trusted fixtures
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "utils"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from utils.data_utils import (  # noqa: E402
    calc_uncertainty_metrics,
    generate_random_sample_dates,
    get_fundamentals,
    safe_pickle_load,
)


class _FakeTopEntry:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


class _FakeTokenLogProbs:
    def __init__(self, logprob: float, entries: list[_FakeTopEntry]) -> None:
        self.logprob = logprob
        self.top_logprobs = entries


def test_safe_pickle_load_roundtrip(tmp_path: Path) -> None:
    payload = {"alpha": 1, "beta": [1, 2, 3]}
    pickle_file = tmp_path / "payload.pkl"
    with pickle_file.open("wb") as fh:
        pickle.dump(payload, fh)

    loaded = safe_pickle_load(pickle_file)

    if loaded != payload:
        pytest.fail("Loaded payload didn't match original data")


def test_safe_pickle_load_requires_trust_flag(tmp_path: Path) -> None:
    pickle_file = tmp_path / "payload.pkl"
    with pickle_file.open("wb") as fh:
        pickle.dump({"foo": "bar"}, fh)

    with pytest.raises(ValueError):
        safe_pickle_load(pickle_file, trusted_source=False)


def test_calc_uncertainty_metrics_happy_path() -> None:
    entries = [_FakeTopEntry(math.log(0.6)), _FakeTopEntry(math.log(0.3))]
    tokens = [_FakeTokenLogProbs(logprob=math.log(0.6), entries=entries)]

    metrics = calc_uncertainty_metrics(tokens)

    expected_keys = {"entropy", "normalized_entropy", "perplexity", "avg_logprob"}
    if set(metrics.keys()) != expected_keys:
        pytest.fail(f"Unexpected metric keys: {metrics.keys()!r}")
    if metrics["perplexity"] <= 0:
        pytest.fail("Perplexity should be positive")


def test_get_fundamentals_adds_growth_columns(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=5, freq="Q"),
            "Quick Ratio": [1.0, 1.1, 1.1, 1.2, 1.5],
        }
    )
    csv_path = tmp_path / "TEST-aggregated_fundamentals.csv"
    data.to_csv(csv_path, index=False)

    fundamentals = get_fundamentals("TEST", tmp_path)

    if "Quick Ratio_QoQ_Growth" not in fundamentals.columns:
        pytest.fail("QoQ growth column missing for Quick Ratio")
    if "Quick Ratio_YoY_Growth" not in fundamentals.columns:
        pytest.fail("YoY growth column missing for Quick Ratio")


def test_generate_random_sample_dates_validates_requests() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    dataframe = pd.DataFrame({"Close": range(5)}, index=index)

    samples = generate_random_sample_dates(dataframe, num_samples=3)

    if len(samples) != 3:
        pytest.fail("Unexpected number of sampled timestamps")
    if samples.tz is None:
        pytest.fail("Sampled timestamps must be timezone aware")

    with pytest.raises(ValueError):
        generate_random_sample_dates(dataframe, num_samples=0)
    with pytest.raises(ValueError):
        generate_random_sample_dates(dataframe, num_samples=10)
