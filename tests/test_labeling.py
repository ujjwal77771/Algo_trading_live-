"""
tests/test_labeling.py
Unit tests for triple_barrier_label and validate_no_leakage.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.labeling import triple_barrier_label, validate_no_leakage

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def flat_series() -> tuple[pd.Series, pd.Series]:
    """Flat price series: neither barrier should ever be hit."""
    n = 50
    idx = pd.RangeIndex(n)
    close = pd.Series(np.full(n, 30_000.0), index=idx)
    atr   = pd.Series(np.full(n, 500.0),   index=idx)
    return close, atr


@pytest.fixture()
def rising_series() -> tuple[pd.Series, pd.Series]:
    """Strongly rising price series: upper barrier should always be hit."""
    n = 100
    idx = pd.RangeIndex(n)
    close = pd.Series(np.arange(n, dtype=float) * 100 + 30_000, index=idx)
    atr   = pd.Series(np.full(n, 50.0), index=idx)
    return close, atr


@pytest.fixture()
def falling_series() -> tuple[pd.Series, pd.Series]:
    """Strongly falling price series: lower barrier should always be hit."""
    n = 100
    idx = pd.RangeIndex(n)
    close = pd.Series(40_000 - np.arange(n, dtype=float) * 100, index=idx)
    atr   = pd.Series(np.full(n, 50.0), index=idx)
    return close, atr


# ── Basic correctness ─────────────────────────────────────────────────────

def test_label_output_type(flat_series: tuple) -> None:
    close, atr = flat_series
    labels = triple_barrier_label(close, atr, horizon=5)
    assert isinstance(labels, pd.Series)
    assert labels.name == "label"


def test_label_values_in_valid_set(flat_series: tuple) -> None:
    """All non-NaN labels must be in {-1, 0, 1}."""
    close, atr = flat_series
    labels = triple_barrier_label(close, atr, horizon=5)
    valid = {-1.0, 0.0, 1.0}
    non_nan = labels.dropna()
    assert set(non_nan.unique()).issubset(valid), (
        f"Unexpected label values: {set(non_nan.unique()) - valid}"
    )


def test_trailing_rows_are_nan(flat_series: tuple) -> None:
    """Last `horizon` rows must be NaN (no lookahead possible)."""
    close, atr = flat_series
    horizon = 5
    labels = triple_barrier_label(close, atr, horizon=horizon)
    trailing = labels.iloc[-horizon:]
    assert trailing.isna().all(), (
        f"Expected last {horizon} rows to be NaN, got: {trailing.values}"
    )


def test_flat_series_labels_zero(flat_series: tuple) -> None:
    """Flat prices should never touch a barrier — all labels must be 0."""
    close, atr = flat_series
    labels = triple_barrier_label(close, atr, horizon=5)
    non_nan = labels.dropna()
    assert (non_nan == 0).all(), "Expected all 0 labels for flat price series"


def test_rising_series_labels_one(rising_series: tuple) -> None:
    """Strongly rising prices should almost always hit the upper barrier first."""
    close, atr = rising_series
    labels = triple_barrier_label(close, atr, horizon=10)
    non_nan = labels.dropna()
    pct_buy = (non_nan == 1).mean()
    assert pct_buy > 0.8, f"Expected >80% buy labels for rising series, got {pct_buy:.1%}"


def test_falling_series_labels_minus_one(falling_series: tuple) -> None:
    """Strongly falling prices should almost always hit the lower barrier first."""
    close, atr = falling_series
    labels = triple_barrier_label(close, atr, horizon=10)
    non_nan = labels.dropna()
    pct_sell = (non_nan == -1).mean()
    assert pct_sell > 0.8, f"Expected >80% sell labels for falling series, got {pct_sell:.1%}"


def test_label_index_matches_input(flat_series: tuple) -> None:
    """Output label index must match the input close index."""
    close, atr = flat_series
    labels = triple_barrier_label(close, atr, horizon=5)
    pd.testing.assert_index_equal(labels.index, close.index)


# ── Validation guards ──────────────────────────────────────────────────────

def test_mismatched_lengths_raises() -> None:
    close = pd.Series(np.ones(50))
    atr   = pd.Series(np.ones(40))
    with pytest.raises(ValueError, match="same length"):
        triple_barrier_label(close, atr, horizon=5)


def test_zero_horizon_raises() -> None:
    close = pd.Series(np.ones(50))
    atr   = pd.Series(np.ones(50))
    with pytest.raises(ValueError, match="positive integer"):
        triple_barrier_label(close, atr, horizon=0)


# ── Leakage detection ──────────────────────────────────────────────────────

def test_no_leakage_on_uncorrelated_features() -> None:
    """validate_no_leakage must return True when features are uncorrelated with label."""
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.RangeIndex(n)
    features = pd.DataFrame(
        {"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)},
        index=idx,
    )
    labels = pd.Series(rng.choice([-1, 0, 1], n), index=idx, dtype=float)
    assert validate_no_leakage(features, labels) is True


def test_leakage_detected_on_perfect_correlation() -> None:
    """validate_no_leakage must return False when a feature equals the label."""
    n = 200
    idx = pd.RangeIndex(n)
    labels = pd.Series(
        np.random.default_rng(1).choice([-1, 0, 1], n).astype(float), index=idx
    )
    # 'leaked' feature is identical to the label — perfect correlation
    features = pd.DataFrame({"leaked": labels.values, "noise": np.ones(n)}, index=idx)
    assert validate_no_leakage(features, labels) is False
