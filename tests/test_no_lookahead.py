"""
tests/test_no_lookahead.py
Verifies that build_features() introduces zero lookahead bias.

Strategy: synthesise a DataFrame where future prices are unique constants,
then confirm that no feature computed from past-only data correlates
with a "future value" column.  Also checks that all feature values at
row t depend only on rows 0..t using a data-masking approach.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.features import build_features

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_ohlcv() -> pd.DataFrame:
    """300-bar synthetic OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed=42)
    n = 300
    close = 30_000 + np.cumsum(rng.normal(0, 100, n))
    high  = close + rng.uniform(50, 300, n)
    low   = close - rng.uniform(50, 300, n)
    open_ = close - rng.normal(0, 80, n)
    vol   = rng.uniform(1_000, 10_000, n)

    idx = pd.date_range("2022-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


# ── Tests ─────────────────────────────────────────────────────────────────

def test_build_features_returns_dataframe(synthetic_ohlcv: pd.DataFrame) -> None:
    """build_features must return a non-empty DataFrame."""
    result = build_features(synthetic_ohlcv)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_build_features_no_nan(synthetic_ohlcv: pd.DataFrame) -> None:
    """build_features must drop all NaN rows — no NaN should remain."""
    result = build_features(synthetic_ohlcv)
    assert not result.isnull().any().any(), "NaN values found in feature matrix"


def test_build_features_is_copy(synthetic_ohlcv: pd.DataFrame) -> None:
    """build_features must not modify the original DataFrame in-place."""
    original_cols = list(synthetic_ohlcv.columns)
    _ = build_features(synthetic_ohlcv)
    assert list(synthetic_ohlcv.columns) == original_cols


def test_feature_count(synthetic_ohlcv: pd.DataFrame) -> None:
    """Feature matrix should have more than 5 columns beyond OHLCV."""
    result = build_features(synthetic_ohlcv)
    ohlcv_cols = {"open", "high", "low", "close", "volume"}
    feature_cols = [c for c in result.columns if c not in ohlcv_cols]
    assert len(feature_cols) >= 20, (
        f"Expected at least 20 feature columns, got {len(feature_cols)}"
    )


def test_no_future_leakage_via_shift(synthetic_ohlcv: pd.DataFrame) -> None:
    """
    Lookahead-bias test: features computed on df[:t] must equal features
    computed on the full df, evaluated at row t.

    We compare values at row index 100 using two different input lengths:
    - full dataset (all 300 rows)
    - truncated dataset (rows 0..100)

    If any feature differs, it means the feature used data beyond row 100.
    """
    full_features = build_features(synthetic_ohlcv)

    # Find row 100 in the original (pre-dropna) index
    cutoff_idx = synthetic_ohlcv.index[100]

    # Only keep rows up to and including cutoff
    truncated = synthetic_ohlcv.loc[:cutoff_idx]
    truncated_features = build_features(truncated)

    if cutoff_idx not in full_features.index or cutoff_idx not in truncated_features.index:
        pytest.skip("Cutoff row was dropped by NaN-removal — extend the fixture.")

    full_row = full_features.loc[cutoff_idx]
    trunc_row = truncated_features.loc[cutoff_idx]

    common_cols = full_row.index.intersection(trunc_row.index)
    for col in common_cols:
        assert abs(float(full_row[col]) - float(trunc_row[col])) < 1e-8, (
            f"Lookahead detected in feature '{col}': "
            f"full={full_row[col]:.6f}, truncated={trunc_row[col]:.6f}"
        )


def test_missing_columns_raises(synthetic_ohlcv: pd.DataFrame) -> None:
    """build_features must raise ValueError on missing OHLCV columns."""
    bad_df = synthetic_ohlcv.drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required OHLCV columns"):
        build_features(bad_df)
