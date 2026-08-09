"""
ML dataset module.
Handles walk-forward cross-validation splits with purging and provides
a convenience function to build aligned (X, y) datasets.

Walk-forward CV respects temporal ordering — training always precedes
testing in time, and a configurable *purge gap* between train and test
prevents label leakage from overlapping forward-looking windows.
"""

from typing import Dict, Generator, List, Tuple

import numpy as np
import pandas as pd

from src.features.indicators import calculate_atr
from src.ml.features import build_features
from src.ml.labeling import triple_barrier_label, validate_no_leakage
from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Named constants — defaults for WalkForwardCV
# ---------------------------------------------------------------------------
DEFAULT_N_SPLITS: int = 5
DEFAULT_TRAIN_PCT: float = 0.6
DEFAULT_PURGE_BARS: int = 5
ATR_WINDOW_FOR_LABELS: int = 14
ATR_MULTIPLIER_KEY: str = "atr_multiplier"
ATR_MULTIPLIER_DEFAULT: float = 2.0


class WalkForwardCV:
    """Walk-forward cross-validator with purge gap for time-series data.

    Each fold advances forward in time:

    * **Train set** — a contiguous block of bars at the start of the
      fold's window.
    * **Purge gap** — ``purge_bars`` bars immediately after training that
      are excluded from both train and test to prevent label leakage.
    * **Test set** — bars after the purge gap up to the next fold boundary.

    No random shuffling is applied — temporal order is always preserved.

    Args:
        n_splits: Number of train/test folds to generate.
        train_pct: Fraction of each fold's window used for training
            (before purging). Must be in ``(0, 1)``.
        purge_bars: Number of bars to drop between train and test sets
            to avoid leakage from forward-looking labels.

    Raises:
        ValueError: If ``n_splits < 2``, ``train_pct`` is outside
            ``(0, 1)``, or ``purge_bars`` is negative.
    """

    def __init__(
        self,
        n_splits: int = DEFAULT_N_SPLITS,
        train_pct: float = DEFAULT_TRAIN_PCT,
        purge_bars: int = DEFAULT_PURGE_BARS,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if not (0.0 < train_pct < 1.0):
            raise ValueError(
                f"train_pct must be in (0, 1), got {train_pct}"
            )
        if purge_bars < 0:
            raise ValueError(
                f"purge_bars must be >= 0, got {purge_bars}"
            )

        self.n_splits = n_splits
        self.train_pct = train_pct
        self.purge_bars = purge_bars

    def split(
        self, X: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate walk-forward train/test index arrays.

        The dataset is divided into ``n_splits + 1`` anchor points.
        For fold *k* (0-indexed):

        * Train indices: ``[anchor[0], split_point)``
        * Purge gap:     ``[split_point, split_point + purge_bars)``
        * Test indices:  ``[split_point + purge_bars, anchor[k+2])``

        where ``split_point`` is determined by ``train_pct`` of the
        fold window ``[anchor[k], anchor[k+2])``.

        Args:
            X: DataFrame whose integer index positions define the
                available bars. Only ``len(X)`` matters; column values
                are not inspected.

        Yields:
            Tuples of ``(train_indices, test_indices)`` as 1-D
            ``np.ndarray`` of integer positions.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # n_splits + 1 equally-spaced anchor points
        anchors: List[int] = [
            int(round(i * n_samples / (self.n_splits + 1)))
            for i in range(self.n_splits + 2)
        ]
        # Ensure last anchor equals n_samples
        anchors[-1] = n_samples

        logger.info(
            "WalkForwardCV: n_splits=%d, train_pct=%.2f, purge=%d, "
            "n_samples=%d",
            self.n_splits,
            self.train_pct,
            self.purge_bars,
            n_samples,
        )

        for fold in range(self.n_splits):
            fold_start = anchors[0]  # always start from beginning
            fold_end = anchors[fold + 2]

            fold_size = fold_end - fold_start
            split_point = fold_start + int(round(fold_size * self.train_pct))

            purge_end = min(split_point + self.purge_bars, fold_end)

            train_idx = indices[fold_start:split_point]
            test_idx = indices[purge_end:fold_end]

            if len(train_idx) == 0 or len(test_idx) == 0:
                logger.warning(
                    "Fold %d skipped: train=%d, test=%d bars",
                    fold,
                    len(train_idx),
                    len(test_idx),
                )
                continue

            logger.info(
                "Fold %d: train [%d–%d] (%d bars), purge %d bars, "
                "test [%d–%d] (%d bars)",
                fold,
                fold_start,
                split_point - 1,
                len(train_idx),
                purge_end - split_point,
                purge_end,
                fold_end - 1,
                len(test_idx),
            )
            yield train_idx, test_idx

    def __repr__(self) -> str:
        return (
            f"WalkForwardCV(n_splits={self.n_splits}, "
            f"train_pct={self.train_pct}, purge_bars={self.purge_bars})"
        )


def build_dataset(
    df: pd.DataFrame,
    config: Dict,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build an aligned (X, y) dataset from raw OHLCV bars.

    Pipeline steps:

    1. ``build_features`` — adds ~35 lagged feature columns and drops
       warm-up NaN rows.
    2. ``triple_barrier_label`` — generates forward-looking labels using
       ATR-based barriers.
    3. Align features and labels on their common index, drop any
       remaining NaN.
    4. ``validate_no_leakage`` — correlation-based sanity check.

    Args:
        df: Raw OHLCV DataFrame with columns
            ``['open', 'high', 'low', 'close', 'volume']``.
        config: Configuration dictionary (from ``load_config``). Expected
            keys: ``config['ml']['horizon']``,
            ``config.get('atr_multiplier', 2.0)``.

    Returns:
        A tuple ``(X, y)`` where *X* is the feature DataFrame and *y*
        is the label Series, both sharing the same index with no NaN.

    Raises:
        ValueError: If the resulting dataset is empty.
    """
    horizon: int = config["ml"]["horizon"]
    atr_mult: float = config.get(ATR_MULTIPLIER_KEY, ATR_MULTIPLIER_DEFAULT)

    logger.info(
        "build_dataset: horizon=%d, atr_multiplier=%.2f", horizon, atr_mult
    )

    # Step 1 — features (backward-looking only, NaN-safe)
    featured_df = build_features(df)

    # Step 2 — labels (forward-looking, computed on the featured index)
    atr = calculate_atr(
        featured_df["high"],
        featured_df["low"],
        featured_df["close"],
        window=ATR_WINDOW_FOR_LABELS,
    )
    labels = triple_barrier_label(
        close=featured_df["close"],
        atr=atr,
        horizon=horizon,
        atr_multiplier=atr_mult,
    )

    # Step 3 — align and drop NaN
    common_idx = featured_df.index.intersection(labels.dropna().index)
    feature_cols = [
        c
        for c in featured_df.columns
        if c not in {"open", "high", "low", "close", "volume"}
    ]
    X = featured_df.loc[common_idx, feature_cols]
    y = labels.loc[common_idx]

    # Final NaN sweep (paranoia)
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    if len(X) == 0:
        raise ValueError(
            "build_dataset produced an empty dataset — check input data "
            "length vs. rolling window warm-up and label horizon."
        )

    # Step 4 — leakage check
    validate_no_leakage(X, y)

    logger.info(
        "Dataset ready: X shape=%s, y shape=%s, label counts:\n%s",
        X.shape,
        y.shape,
        y.value_counts().sort_index().to_string(),
    )
    return X, y
