"""
tests/test_ensemble_no_trade_on_missing_data.py
Verifies that EnsembleSignal returns HOLD (0) when insufficient bars
are provided to the underlying signal generators.

This is the "never trade on missing data" acceptance criterion from Step 3.
"""

from typing import Dict, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.signals.ensemble import EnsembleSignal
from src.signals.base import SignalBase
from src.ml.regime import RegimeDetector


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with n bars."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h")
    close = 30_000 + np.cumsum(rng.normal(0, 100, n))
    return pd.DataFrame({
        "open":   close - 10,
        "high":   close + 50,
        "low":    close - 50,
        "close":  close,
        "volume": rng.uniform(100, 1000, n),
    }, index=idx)


class _FixedSignal(SignalBase):
    """Signal generator that always returns a fixed value."""
    def __init__(self, value: int) -> None:
        self._value = value

    def generate_signal(self, data: pd.DataFrame) -> int:
        return self._value


class _RaisesSignal(SignalBase):
    """Signal generator that raises an exception — simulates broken feed."""
    def generate_signal(self, data: pd.DataFrame) -> int:
        raise RuntimeError("Feed unavailable")


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def config() -> Dict[str, Any]:
    return {
        "signals": {
            "ema_fast": 12, "ema_slow": 26,
            "rsi_window": 14, "rsi_overbought": 70, "rsi_oversold": 30,
        }
    }


@pytest.fixture()
def regime_sideways() -> RegimeDetector:
    """A RegimeDetector that always returns 'sideways'."""
    det = MagicMock(spec=RegimeDetector)
    det.detect.return_value = "sideways"
    return det


@pytest.fixture()
def regime_trending() -> RegimeDetector:
    """A RegimeDetector that always returns 'trending'."""
    det = MagicMock(spec=RegimeDetector)
    det.detect.return_value = "trending"
    return det


# ── Tests ──────────────────────────────────────────────────────────────────

def test_hold_on_empty_dataframe(config, regime_sideways) -> None:
    """EnsembleSignal must return 0 (HOLD) when given an empty DataFrame."""
    ensemble = EnsembleSignal(
        config=config,
        rule_signal=_FixedSignal(1),
        ml_signal=_FixedSignal(1),
        regime_detector=regime_sideways,
    )
    empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert ensemble.generate_signal(empty_df) == 0


def test_hold_on_single_bar(config, regime_sideways) -> None:
    """EnsembleSignal must return 0 (HOLD) on a single-bar DataFrame."""
    ensemble = EnsembleSignal(
        config=config,
        rule_signal=_FixedSignal(1),
        ml_signal=_FixedSignal(1),
        regime_detector=regime_sideways,
    )
    assert ensemble.generate_signal(_make_ohlcv(1)) == 0


def test_hold_when_rule_signal_raises(config, regime_sideways) -> None:
    """EnsembleSignal must return 0 if rule_signal raises an exception."""
    ensemble = EnsembleSignal(
        config=config,
        rule_signal=_RaisesSignal(),
        ml_signal=_FixedSignal(1),
        regime_detector=regime_sideways,
    )
    assert ensemble.generate_signal(_make_ohlcv(100)) == 0


def test_hold_when_ml_signal_raises(config, regime_sideways) -> None:
    """EnsembleSignal must return 0 if ml_signal raises an exception."""
    ensemble = EnsembleSignal(
        config=config,
        rule_signal=_FixedSignal(1),
        ml_signal=_RaisesSignal(),
        regime_detector=regime_sideways,
    )
    assert ensemble.generate_signal(_make_ohlcv(100)) == 0


def test_unanimous_buy_returns_one(config, regime_sideways) -> None:
    """When both signals agree on BUY, ensemble must return 1."""
    ensemble = EnsembleSignal(
        config=config,
        rule_signal=_FixedSignal(1),
        ml_signal=_FixedSignal(1),
        regime_detector=regime_sideways,
    )
    result = ensemble.generate_signal(_make_ohlcv(100))
    assert result == 1


def test_unanimous_sell_returns_minus_one(config, regime_sideways) -> None:
    """When both signals agree on SELL, ensemble must return -1."""
    ensemble = EnsembleSignal(
        config=config,
        rule_signal=_FixedSignal(-1),
        ml_signal=_FixedSignal(-1),
        regime_detector=regime_sideways,
    )
    result = ensemble.generate_signal(_make_ohlcv(100))
    assert result == -1


def test_conflicting_signals_return_hold(config, regime_sideways) -> None:
    """When rule=BUY and ml=SELL with equal weights, the sum is 0 → HOLD."""
    ensemble = EnsembleSignal(
        config=config,
        rule_signal=_FixedSignal(1),
        ml_signal=_FixedSignal(-1),
        regime_detector=regime_sideways,
    )
    result = ensemble.generate_signal(_make_ohlcv(100))
    assert result == 0


def test_trending_regime_weights_rule_higher(config, regime_trending) -> None:
    """In a trending regime rule weight (0.6) > ml weight (0.4).
    rule=1, ml=0 → weighted sum = 0.6 → buy."""
    ensemble = EnsembleSignal(
        config=config,
        rule_signal=_FixedSignal(1),
        ml_signal=_FixedSignal(0),
        regime_detector=regime_trending,
    )
    result = ensemble.generate_signal(_make_ohlcv(100))
    assert result == 1


def test_sideways_regime_weights_ml_higher(config, regime_sideways) -> None:
    """In a sideways regime ml weight (0.6) > rule weight (0.4).
    rule=0, ml=-1 → weighted sum = -0.6 → sell."""
    ensemble = EnsembleSignal(
        config=config,
        rule_signal=_FixedSignal(0),
        ml_signal=_FixedSignal(-1),
        regime_detector=regime_sideways,
    )
    result = ensemble.generate_signal(_make_ohlcv(100))
    assert result == -1
