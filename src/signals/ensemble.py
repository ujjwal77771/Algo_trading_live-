"""
Ensemble signal module.
Combines a rule-based signal, an ML signal, and a regime detector into a
single adaptive trading signal with regime-dependent weighting and a
unanimous-agreement override.
"""

from typing import Any, Dict

import pandas as pd

from src.ml.regime import RegimeDetector
from src.signals.base import SignalBase
from src.utils.logger import logger

# Regime-dependent signal weights.
# In a trending market the rule-based (trend-following) signal is weighted
# higher; in a sideways market the ML model gets more weight.
_TRENDING_RULE_WEIGHT: float = 0.6
_TRENDING_ML_WEIGHT: float = 0.4
_SIDEWAYS_RULE_WEIGHT: float = 0.4
_SIDEWAYS_ML_WEIGHT: float = 0.6

# Threshold for converting the weighted sum to a discrete signal.
_SIGNAL_THRESHOLD: float = 0.5


class EnsembleSignal(SignalBase):
    """Adaptive ensemble that blends rule-based and ML signals.

    Decision logic (per bar):
      1. Detect the current market regime (``'trending'`` / ``'sideways'``).
      2. Query both child signal generators.
      3. If **either** child returns ``None`` or raises, emit ``0`` (Hold) —
         we never trade on missing data.
      4. **Unanimous-agreement override**: if both signals agree (both ``1``
         or both ``-1``) return that signal regardless of weighting.
      5. Otherwise compute the regime-weighted sum and apply a ±0.5 threshold
         to produce Buy / Sell / Hold.

    Attributes:
        config: Application configuration dict.
        rule_signal: Rule-based signal generator.
        ml_signal: ML-based signal generator.
        regime_detector: Market regime classifier.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        rule_signal: SignalBase,
        ml_signal: SignalBase,
        regime_detector: RegimeDetector,
    ) -> None:
        """Initialise the ensemble.

        Args:
            config: Application configuration dict.
            rule_signal: A ``SignalBase``-conforming rule-based generator.
            ml_signal: A ``SignalBase``-conforming ML generator.
            regime_detector: A :class:`RegimeDetector` instance.
        """
        self.config: Dict[str, Any] = config
        self.rule_signal: SignalBase = rule_signal
        self.ml_signal: SignalBase = ml_signal
        self.regime_detector: RegimeDetector = regime_detector

        logger.info("EnsembleSignal initialised.")

    # ------------------------------------------------------------------
    # SignalBase implementation
    # ------------------------------------------------------------------

    def generate_signal(self, data: pd.DataFrame) -> int:
        """Generate the final blended trading signal.

        Args:
            data: OHLCV DataFrame passed to both child signals and the
                regime detector.

        Returns:
            ``1`` (Buy), ``-1`` (Sell), or ``0`` (Hold).
        """
        # 1. Detect regime ------------------------------------------------
        regime: str = self.regime_detector.detect(data)

        # 2. Obtain child signals -----------------------------------------
        rule_sig = self._safe_signal(self.rule_signal, data, label="rule")
        ml_sig = self._safe_signal(self.ml_signal, data, label="ml")

        # 3. Missing-data safety gate -------------------------------------
        if rule_sig is None or ml_sig is None:
            logger.warning(
                "EnsembleSignal: a child signal returned None. "
                "Emitting HOLD to avoid trading on missing data."
            )
            return 0

        # 4. Unanimous-agreement override ----------------------------------
        if rule_sig == ml_sig and rule_sig != 0:
            logger.debug(
                "EnsembleSignal: unanimous agreement (%d). "
                "Overriding weighted logic.",
                rule_sig,
            )
            return rule_sig

        # 5. Regime-weighted blending --------------------------------------
        if regime == "trending":
            rule_weight = _TRENDING_RULE_WEIGHT
            ml_weight = _TRENDING_ML_WEIGHT
        else:
            rule_weight = _SIDEWAYS_RULE_WEIGHT
            ml_weight = _SIDEWAYS_ML_WEIGHT

        weighted_sum: float = rule_weight * rule_sig + ml_weight * ml_sig

        if weighted_sum > _SIGNAL_THRESHOLD:
            signal = 1
        elif weighted_sum < -_SIGNAL_THRESHOLD:
            signal = -1
        else:
            signal = 0

        logger.debug(
            "EnsembleSignal: regime=%s, rule=%d, ml=%d, "
            "weighted_sum=%.2f → signal=%d",
            regime,
            rule_sig,
            ml_sig,
            weighted_sum,
            signal,
        )

        return signal

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_signal(
        signal_source: SignalBase,
        data: pd.DataFrame,
        label: str,
    ) -> int | None:
        """Invoke a child signal, catching errors.

        Args:
            signal_source: A ``SignalBase`` instance.
            data: OHLCV DataFrame to pass to
                ``signal_source.generate_signal``.
            label: Human-readable name for logging.

        Returns:
            The integer signal, or ``None`` if the call failed.
        """
        try:
            result: int = signal_source.generate_signal(data)
            if result is None:
                logger.warning(
                    "EnsembleSignal: %s signal returned None.", label
                )
                return None
            return result
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            logger.warning(
                "EnsembleSignal: %s signal raised %s: %s",
                label,
                type(exc).__name__,
                exc,
            )
            return None
