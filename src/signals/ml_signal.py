"""
ML-based signal generator.
Wraps a trained LightGBM model behind the standard ``SignalBase`` interface.
"""

from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ml.features import build_features
from src.signals.base import SignalBase
from src.utils.logger import logger

# LightGBM multiclass predictions produce class indices 0, 1, 2.
# We map them back to the trading-signal domain: -1 (sell), 0 (hold), 1 (buy).
_CLASS_TO_SIGNAL: Dict[int, int] = {
    0: -1,  # class 0 → Sell
    1: 0,   # class 1 → Hold
    2: 1,   # class 2 → Buy
}


class MLSignal(SignalBase):
    """Generates trading signals from a trained LightGBM model.

    The pipeline is:
      1. ``build_features`` creates the feature matrix from raw OHLCV data.
      2. The model predicts class probabilities for the latest bar.
      3. The argmax class is mapped through ``_CLASS_TO_SIGNAL``.

    If insufficient data is available (fewer bars than the configured
    lookback), the signal defaults to 0 (Hold).

    Attributes:
        config: Application configuration dict.
        model: Pre-trained LightGBM Booster.
        lookback: Minimum bars required before a prediction can be made.
    """

    def __init__(self, config: Dict[str, Any], model: lgb.Booster) -> None:
        """Initialise the ML signal generator.

        Args:
            config: Application configuration dict (must contain ``ml`` block).
            model: A trained ``lgb.Booster`` instance.

        Raises:
            KeyError: If the ``ml`` config block is missing.
        """
        ml_config: Dict[str, Any] = config.get("ml", {})
        if not ml_config:
            raise KeyError(
                "Missing required 'ml' block in config for MLSignal."
            )

        self.config: Dict[str, Any] = config
        self.model: lgb.Booster = model
        self.lookback: int = int(ml_config["lookback"])

        logger.info("MLSignal initialised — lookback=%d", self.lookback)

    # ------------------------------------------------------------------
    # SignalBase implementation
    # ------------------------------------------------------------------

    def generate_signal(self, data: pd.DataFrame) -> int:
        """Produce a trading signal from the latest OHLCV data.

        Args:
            data: OHLCV DataFrame.  Must have at least ``self.lookback``
                rows for feature construction to succeed.

        Returns:
            ``1`` (Buy), ``-1`` (Sell), or ``0`` (Hold).
        """
        if len(data) < self.lookback:
            logger.debug(
                "MLSignal: insufficient bars (%d < %d). Returning HOLD.",
                len(data),
                self.lookback,
            )
            return 0

        # --- Build features -----------------------------------------------
        try:
            features: pd.DataFrame = build_features(data)
        except (KeyError, ValueError) as exc:
            logger.warning(
                "MLSignal: build_features failed (%s). Returning HOLD.", exc
            )
            return 0

        if features.empty or len(features) == 0:
            logger.debug(
                "MLSignal: build_features returned empty DataFrame. Returning HOLD."
            )
            return 0

        # --- Predict on the most recent bar --------------------------------
        latest_row: pd.DataFrame = features.iloc[[-1]]

        try:
            probabilities: np.ndarray = self.model.predict(latest_row)
        except (lgb.basic.LightGBMError, ValueError) as exc:
            logger.warning(
                "MLSignal: model.predict failed (%s). Returning HOLD.", exc
            )
            return 0

        predicted_class: int = int(np.argmax(probabilities, axis=1)[0])
        signal: int = _CLASS_TO_SIGNAL.get(predicted_class, 0)

        logger.debug(
            "MLSignal: probabilities=%s, class=%d, signal=%d",
            probabilities,
            predicted_class,
            signal,
        )

        return signal
