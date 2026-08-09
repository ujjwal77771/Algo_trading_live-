"""
tests/test_risk_manager.py
Unit tests for RiskManager — position sizing, SL/TP, halt logic.
"""

import pytest

from src.risk.risk_manager import RiskManager

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def base_config() -> dict:
    return {
        "risk_per_trade_pct": 0.01,
        "atr_multiplier": 2.0,
        "max_drawdown": 0.15,
        "max_daily_loss": 0.05,
        "reward_risk_ratio": 1.5,
    }


@pytest.fixture()
def rm(base_config: dict) -> RiskManager:
    return RiskManager(base_config)


# ── Init / config validation ───────────────────────────────────────────────

def test_init_missing_key_raises() -> None:
    """RiskManager must raise KeyError if any required config key is absent."""
    bad_config = {
        "risk_per_trade_pct": 0.01,
        "atr_multiplier": 2.0,
        "max_drawdown": 0.15,
        # max_daily_loss missing
        # reward_risk_ratio missing
    }
    with pytest.raises(KeyError, match="Missing required risk config keys"):
        RiskManager(bad_config)


def test_init_values(rm: RiskManager, base_config: dict) -> None:
    """RiskManager must correctly store all config values."""
    assert rm.risk_per_trade == base_config["risk_per_trade_pct"]
    assert rm.atr_multiplier == base_config["atr_multiplier"]
    assert rm.max_drawdown == base_config["max_drawdown"]
    assert rm.max_daily_loss == base_config["max_daily_loss"]
    assert rm.reward_risk_ratio == base_config["reward_risk_ratio"]
    assert rm.halted is False


# ── Position sizing ────────────────────────────────────────────────────────

def test_position_size_basic(rm: RiskManager) -> None:
    """Position size = (capital * risk_pct) / (atr * multiplier)."""
    capital = 10_000.0
    price = 30_000.0
    atr = 500.0
    size = rm.calculate_position_size(capital, price, atr)
    expected = (capital * 0.01) / (atr * 2.0)  # = 0.1
    assert abs(size - expected) < 1e-9


def test_position_size_capped_by_capital(rm: RiskManager) -> None:
    """Position size must never exceed capital / price."""
    capital = 100.0
    price = 50_000.0
    atr = 1.0          # tiny ATR → huge uncapped size
    size = rm.calculate_position_size(capital, price, atr)
    assert size <= capital / price


def test_position_size_zero_on_invalid_atr(rm: RiskManager) -> None:
    """Position size must be 0 when ATR is zero or negative."""
    assert rm.calculate_position_size(10_000, 30_000, 0.0) == 0.0
    assert rm.calculate_position_size(10_000, 30_000, -1.0) == 0.0


def test_position_size_zero_when_halted(rm: RiskManager) -> None:
    """Position size must be 0 when the RiskManager is halted."""
    rm.halted = True
    assert rm.calculate_position_size(10_000, 30_000, 500) == 0.0


# ── SL / TP ───────────────────────────────────────────────────────────────

def test_sl_tp_long(rm: RiskManager) -> None:
    """For a long position, SL < entry < TP."""
    entry = 30_000.0
    atr = 500.0
    sl, tp = rm.calculate_sl_tp(entry, direction=1, atr=atr)
    assert sl is not None and tp is not None
    assert sl < entry < tp


def test_sl_tp_short(rm: RiskManager) -> None:
    """For a short position, TP < entry < SL."""
    entry = 30_000.0
    atr = 500.0
    sl, tp = rm.calculate_sl_tp(entry, direction=-1, atr=atr)
    assert sl is not None and tp is not None
    assert tp < entry < sl


def test_sl_tp_distance(rm: RiskManager) -> None:
    """TP distance must equal SL distance × reward_risk_ratio."""
    entry = 30_000.0
    atr = 500.0
    sl, tp = rm.calculate_sl_tp(entry, direction=1, atr=atr)
    sl_dist = entry - sl
    tp_dist = tp - entry
    assert abs(tp_dist / sl_dist - rm.reward_risk_ratio) < 1e-9


def test_sl_tp_zero_atr_returns_none(rm: RiskManager) -> None:
    """SL/TP must be (None, None) when ATR is zero."""
    sl, tp = rm.calculate_sl_tp(30_000, direction=1, atr=0.0)
    assert sl is None and tp is None


# ── Drawdown halt ──────────────────────────────────────────────────────────

def test_max_drawdown_triggers_halt(rm: RiskManager) -> None:
    """RiskManager must halt when drawdown exceeds max_drawdown threshold."""
    rm.update_equity(10_000.0)   # peak
    rm.update_equity(8_000.0)    # 20 % drawdown > 15 % threshold
    assert rm.halted is True


def test_drawdown_below_threshold_no_halt(rm: RiskManager) -> None:
    """No halt if drawdown is within bounds and daily loss is not exceeded."""
    rm.update_equity(10_000.0)              # sets peak + start_of_day_equity
    rm.update_equity(9_500.0, is_new_day=True)  # new day resets daily baseline to 9500
    rm.update_equity(9_050.0)              # 4.7% daily loss (< 5%) and ~9.5% drawdown (< 15%)
    assert rm.halted is False



def test_daily_loss_triggers_halt(rm: RiskManager) -> None:
    """RiskManager must halt when daily loss exceeds max_daily_loss."""
    rm.update_equity(10_000.0, is_new_day=True)   # start of day
    rm.update_equity(9_400.0)                      # 6 % daily loss > 5 %
    assert rm.halted is True


def test_new_day_resets_start_of_day_equity(rm: RiskManager) -> None:
    """start_of_day_equity must update when is_new_day=True."""
    rm.update_equity(10_000.0)
    rm.update_equity(9_500.0, is_new_day=True)   # new day at lower equity
    # Daily loss relative to 9_500, not original 10_000
    rm.update_equity(9_010.0)   # ~5.2 % drop from 9_500 → should halt
    assert rm.halted is True
