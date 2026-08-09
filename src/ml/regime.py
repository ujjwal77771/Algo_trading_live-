"""
Market regime detection module.
Classifies the current market as 'trending' or 'sideways' using ADX and
rolling return volatility.
"""

import numpy as np
import pandas as pd

from src.features.indicators import calculate_adx
from src.utils.logger import logger


class RegimeDetector:
    """Detects whether the market is in a trending or sideways regime.

    Decision rule (applied to the *latest* bar):
        * **Trending** — ADX > ``adx_threshold`` **and** current rolling
          volatility is above its historical median.
        * **Sideways** — everything else.

    Attributes:
        adx_threshold: ADX value above which a trend is considered present.
        vol_lookback: Window (in bars) used for the rolling standard-deviation
            of log returns.
    """

    # ADX uses a 14-bar internal window by default; together with the
    # volatility lookback we need at least this many bars to produce a
    # non-NaN result.
    _ADX_INTERNAL_WINDOW: int = 14

    def __init__(
        self,
        adx_threshold: float = 25.0,
        vol_lookback: int = 20,
    ) -> None:
        """Initialise the regime detector.

        Args:
            adx_threshold: ADX reading above which a trend may be declared.
            vol_lookback: Rolling window for return-volatility calculation.
        """
        self.adx_threshold: float = adx_threshold
        self.vol_lookback: int = vol_lookback

        logger.info(
            "RegimeDetector initialised — adx_threshold=%.1f, vol_lookback=%d",
            self.adx_threshold,
            self.vol_lookback,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, data: pd.DataFrame) -> str:
        """Classify the current market regime.

        Args:
            data: OHLCV DataFrame with at least ``high``, ``low``, and
                ``close`` columns.  Must contain enough rows for both the
                ADX and volatility calculations.

        Returns:
            ``'trending'`` or ``'sideways'``.
        """
        min_bars: int = max(
            2 * self._ADX_INTERNAL_WINDOW,  # ADX needs ~2× its window
            self.vol_lookback,
        ) + 1

        if len(data) < min_bars:
            logger.warning(
                "RegimeDetector.detect: insufficient bars (%d < %d). "
                "Defaulting to 'sideways'.",
                len(data),
                min_bars,
            )
            return "sideways"

        # --- ADX ----------------------------------------------------------
        adx: pd.Series = calculate_adx(
            data["high"],
            data["low"],
            data["close"],
            window=self._ADX_INTERNAL_WINDOW,
        )
        current_adx: float = float(adx.iloc[-1])

        # --- Rolling volatility -------------------------------------------
        log_returns: pd.Series = np.log(data["close"] / data["close"].shift(1))
        rolling_vol: pd.Series = log_returns.rolling(window=self.vol_lookback).std()
        current_vol: float = float(rolling_vol.iloc[-1])
        median_vol: float = float(rolling_vol.median())

        is_trending: bool = (
            current_adx > self.adx_threshold and current_vol > median_vol
        )
        regime: str = "trending" if is_trending else "sideways"

        logger.debug(
            "Regime detection — ADX=%.2f (threshold=%.1f), "
            "vol=%.6f (median=%.6f) → %s",
            current_adx,
            self.adx_threshold,
            current_vol,
            median_vol,
            regime,
        )

        return regime
