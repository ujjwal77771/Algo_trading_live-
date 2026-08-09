"""
Optuna-based walk-forward hyperparameter tuning for LightGBM.
Each trial samples a parameter set, trains on every walk-forward fold
provided by ``WalkForwardCV``, and reports mean test-fold accuracy.
"""

from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score

from src.ml.dataset import WalkForwardCV
from src.utils.logger import logger

# Suppress Optuna's internal INFO chatter; we log summaries ourselves.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------- Search-space bounds (named constants, not magic numbers) ----------
_MIN_NUM_LEAVES: int = 15
_MAX_NUM_LEAVES: int = 63
_MIN_LEARNING_RATE: float = 0.01
_MAX_LEARNING_RATE: float = 0.1
_MIN_N_ESTIMATORS: int = 100
_MAX_N_ESTIMATORS: int = 500
_MIN_CHILD_SAMPLES: int = 10
_MAX_CHILD_SAMPLES: int = 50
_MIN_SUBSAMPLE: float = 0.6
_MAX_SUBSAMPLE: float = 1.0
_MIN_COLSAMPLE: float = 0.6
_MAX_COLSAMPLE: float = 1.0

# Fixed LightGBM params that are not tuned.
_FIXED_PARAMS: Dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "verbose": -1,
}


class HyperparamTuner:
    """Walk-forward Optuna tuner for LightGBM multiclass models.

    For each Optuna trial the tuner:
      1. Samples a LightGBM parameter set from the search space.
      2. Iterates over every fold produced by :class:`WalkForwardCV`.
      3. Trains on the train indices and predicts on the test indices.
      4. Computes per-fold accuracy and returns the mean across folds.

    After all trials the best parameter dictionary is returned.

    Attributes:
        config: Full application config dict.
        n_trials: Number of Optuna trials to run.
    """

    def __init__(self, config: Dict[str, Any], n_trials: int = 50) -> None:
        """Initialise the tuner.

        Args:
            config: Application configuration dict (must contain ``ml`` block).
            n_trials: Number of Optuna optimisation trials.
        """
        self.config: Dict[str, Any] = config
        self.n_trials: int = n_trials

        ml_config: Dict[str, Any] = config.get("ml", {})
        if not ml_config:
            raise KeyError(
                "Missing required 'ml' block in config for HyperparamTuner."
            )

        logger.info("HyperparamTuner initialised — n_trials=%d", self.n_trials)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Run the Optuna study and return the best parameters.

        Args:
            X: Feature matrix.
            y: Integer-encoded target labels (0 / 1 / 2).

        Returns:
            Dict with the best LightGBM parameters found.
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
        )

        best_params: Dict[str, Any] = {**_FIXED_PARAMS, **study.best_params}
        logger.info(
            "Optuna tuning complete — best accuracy=%.4f, params=%s",
            study.best_value,
            best_params,
        )
        return best_params

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """Optuna objective: walk-forward CV mean accuracy.

        Args:
            trial: Current Optuna trial.
            X: Feature matrix.
            y: Labels.

        Returns:
            Mean accuracy across walk-forward folds.
        """
        params: Dict[str, Any] = {
            **_FIXED_PARAMS,
            "num_leaves": trial.suggest_int(
                "num_leaves", _MIN_NUM_LEAVES, _MAX_NUM_LEAVES
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", _MIN_LEARNING_RATE, _MAX_LEARNING_RATE, log=True
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", _MIN_CHILD_SAMPLES, _MAX_CHILD_SAMPLES
            ),
            "subsample": trial.suggest_float(
                "subsample", _MIN_SUBSAMPLE, _MAX_SUBSAMPLE
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", _MIN_COLSAMPLE, _MAX_COLSAMPLE
            ),
        }
        n_estimators: int = trial.suggest_int(
            "n_estimators", _MIN_N_ESTIMATORS, _MAX_N_ESTIMATORS, step=50
        )

        ml_config: Dict[str, Any] = self.config["ml"]
        cv = WalkForwardCV(
            n_splits=ml_config.get("n_splits", 5),
            train_pct=ml_config.get("train_pct", 0.8),
            purge_bars=ml_config.get("purge_bars", self.config["ml"]["horizon"]),
        )

        fold_accuracies: list[float] = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            train_ds = lgb.Dataset(X_train, label=y_train)
            model: lgb.Booster = lgb.train(params, train_ds, num_boost_round=n_estimators)

            y_prob: np.ndarray = model.predict(X_test)
            y_pred: np.ndarray = np.argmax(y_prob, axis=1)

            fold_acc: float = float(accuracy_score(y_test, y_pred))
            fold_accuracies.append(fold_acc)

        mean_acc: float = float(np.mean(fold_accuracies))
        return mean_acc
