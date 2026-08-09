"""
ML labeling module.
Implements triple-barrier labeling from Marcos López de Prado's
*Advances in Financial Machine Learning*.

Labels are inherently forward-looking (they encode future price action)
and must NEVER be used as features.  The ``validate_no_leakage`` helper
provides a simple correlation-based sanity check.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import logger

# Correlation threshold above which a feature is flagged as leaked
_LEAKAGE_CORR_THRESHOLD: float = 0.99


def triple_barrier_label(
    close: pd.Series,
    atr: pd.Series,
    horizon: int,
    atr_multiplier: float = 2.0,
) -> pd.Series:
    """Compute triple-barrier labels for supervised learning.

    For each bar *t* three barriers are defined:

    * **Upper barrier** — ``close[t] + atr[t] * atr_multiplier``
    * **Lower barrier** — ``close[t] - atr[t] * atr_multiplier``
    * **Time barrier**  — ``t + horizon`` bars

    The label records which barrier is touched first in the window
    ``[t+1 … t+horizon]``:

    * ``1``  — upper barrier hit first (buy signal)
    * ``-1`` — lower barrier hit first (sell signal)
    * ``0``  — neither barrier hit within the horizon (hold)
    * ``NaN`` — insufficient forward data (last *horizon* bars)

    Args:
        close: Series of closing prices, indexed consistently with *atr*.
        atr: Series of Average True Range values, same index as *close*.
        horizon: Number of forward bars to evaluate.
        atr_multiplier: Scalar applied to ATR for barrier width.

    Returns:
        A ``pd.Series`` of integer labels (``{-1, 0, 1}``) aligned to
        the input index, with ``NaN`` for the trailing *horizon* bars.

    Raises:
        ValueError: If *close* and *atr* have different lengths or if
            *horizon* is not a positive integer.
    """
    _validate_label_inputs(close, atr, horizon)

    n = len(close)
    close_arr: np.ndarray = close.values.astype(np.float64)
    atr_arr: np.ndarray = atr.values.astype(np.float64)

    labels = np.full(n, np.nan, dtype=np.float64)

    # Vectorised barrier boundaries
    upper = close_arr + atr_arr * atr_multiplier
    lower = close_arr - atr_arr * atr_multiplier

    labelable_end = n - horizon  # exclusive upper bound

    logger.info(
        "Computing triple-barrier labels: horizon=%d, atr_mult=%.2f, "
        "labelable bars=%d/%d",
        horizon,
        atr_multiplier,
        max(labelable_end, 0),
        n,
    )

    for t in range(labelable_end):
        upper_barrier = upper[t]
        lower_barrier = lower[t]

        # Scan forward from t+1 to t+horizon (inclusive)
        window_end = t + horizon + 1  # exclusive for slicing
        future_prices = close_arr[t + 1 : window_end]

        upper_hits = np.where(future_prices >= upper_barrier)[0]
        lower_hits = np.where(future_prices <= lower_barrier)[0]

        first_upper: Optional[int] = (
            int(upper_hits[0]) if len(upper_hits) > 0 else None
        )
        first_lower: Optional[int] = (
            int(lower_hits[0]) if len(lower_hits) > 0 else None
        )

        if first_upper is not None and first_lower is not None:
            # Both barriers touched — the earlier one wins
            labels[t] = 1.0 if first_upper <= first_lower else -1.0
        elif first_upper is not None:
            labels[t] = 1.0
        elif first_lower is not None:
            labels[t] = -1.0
        else:
            labels[t] = 0.0

    result = pd.Series(labels, index=close.index, name="label")

    # Log class distribution for quick diagnostics
    counts = result.dropna().value_counts().sort_index()
    logger.info("Label distribution:\n%s", counts.to_string())

    return result


def validate_no_leakage(
    features_df: pd.DataFrame,
    labels: pd.Series,
) -> bool:
    """Check that no feature column is suspiciously correlated with labels.

    A feature with Pearson |correlation| > ``_LEAKAGE_CORR_THRESHOLD``
    against the label is almost certainly leaked.

    Args:
        features_df: DataFrame of feature columns (X).
        labels: Series of target labels (y), aligned to *features_df*.

    Returns:
        ``True`` if no leakage is detected, ``False`` otherwise.
    """
    aligned_labels = labels.reindex(features_df.index)
    leaked_cols = []

    for col in features_df.columns:
        try:
            corr = features_df[col].corr(aligned_labels)
        except (TypeError, ValueError):
            # Non-numeric or constant columns cannot be correlated
            continue

        if abs(corr) > _LEAKAGE_CORR_THRESHOLD:
            leaked_cols.append((col, corr))

    if leaked_cols:
        for col, corr in leaked_cols:
            logger.warning(
                "LEAKAGE DETECTED: feature '%s' has %.4f correlation with label",
                col,
                corr,
            )
        return False

    logger.info(
        "Leakage check passed — no feature exceeds %.2f |correlation| with label",
        _LEAKAGE_CORR_THRESHOLD,
    )
    return True


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_label_inputs(
    close: pd.Series,
    atr: pd.Series,
    horizon: int,
) -> None:
    """Validate inputs for ``triple_barrier_label``.

    Args:
        close: Closing price series.
        atr: ATR series.
        horizon: Forward look-ahead window.

    Raises:
        ValueError: On mismatched lengths or non-positive horizon.
    """
    if len(close) != len(atr):
        raise ValueError(
            f"close and atr must have the same length, "
            f"got {len(close)} vs {len(atr)}"
        )
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError(
            f"horizon must be a positive integer, got {horizon!r}"
        )
