"""
ML training module.
Trains LightGBM multiclass classifiers with a feature-importance sanity gate.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import lightgbm as lgb
import pandas as pd

from src.utils.logger import logger

# Very short-lag return features that are predominantly noise.
# If ALL top-k features by gain come from this set the model is likely
# overfitting to random walks rather than learning real structure.
NOISE_FEATURES: Set[str] = {"ret_1", "ret_2"}

# How many top features (by gain) to inspect in the sanity gate.
TOP_K_FEATURES: int = 5

# ----- LightGBM default hyper-parameters -----
_DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "verbose": -1,
}

_DEFAULT_MODEL_FILENAME: str = "lgb_model.txt"


class ModelTrainer:
    """Trains, evaluates, saves and loads LightGBM multiclass models.

    The trainer includes a **feature-importance sanity gate**: after training
    it checks whether all of the ``TOP_K_FEATURES`` highest-gain features
    belong to the ``NOISE_FEATURES`` set.  If so, a warning is logged and
    ``self.sanity_passed`` is set to ``False``.

    Attributes:
        config: Full application config dict (must contain an ``ml`` block).
        model_dir: Directory for persisting trained models.
        sanity_passed: ``True`` after a successful sanity check, ``False``
            if the gate tripped, ``None`` before any training has run.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the trainer from the application config.

        Args:
            config: Application configuration dict.  Must contain an ``ml``
                sub-dict with at least ``lookback``, ``horizon``, and
                ``model_dir`` keys.

        Raises:
            KeyError: If the required ``ml`` config block is missing.
        """
        ml_config: Dict[str, Any] = config.get("ml", {})
        if not ml_config:
            raise KeyError(
                "Missing required 'ml' block in config. "
                "Ensure settings.yaml defines ml.lookback, ml.horizon, ml.model_dir."
            )

        self.lookback: int = int(ml_config["lookback"])
        self.horizon: int = int(ml_config["horizon"])
        self.model_dir: Path = Path(str(ml_config["model_dir"]))
        self.sanity_passed: Optional[bool] = None

        logger.info(
            "ModelTrainer initialised — lookback=%d, horizon=%d, model_dir=%s",
            self.lookback,
            self.horizon,
            self.model_dir,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ) -> lgb.Booster:
        """Train a LightGBM multiclass classifier.

        Args:
            X: Feature matrix (rows = samples, columns = feature names).
            y: Integer-encoded labels.  Expected values are ``{0, 1, 2}``
               which map to sell / hold / buy respectively.
            params: Optional dict of LightGBM parameters.  Merged on top of
                ``_DEFAULT_PARAMS`` so callers only need to specify overrides.

        Returns:
            The trained ``lgb.Booster`` instance.
        """
        merged_params: Dict[str, Any] = {**_DEFAULT_PARAMS, **(params or {})}
        n_estimators: int = int(merged_params.pop("n_estimators", _DEFAULT_PARAMS["n_estimators"]))

        logger.info(
            "Starting LightGBM training — %d samples, %d features, n_estimators=%d",
            len(X),
            X.shape[1],
            n_estimators,
        )

        train_dataset = lgb.Dataset(X, label=y)
        model: lgb.Booster = lgb.train(
            merged_params,
            train_dataset,
            num_boost_round=n_estimators,
        )

        logger.info("LightGBM training complete.")
        self._run_sanity_gate(model, list(X.columns))
        return model

    # ------------------------------------------------------------------
    # Sanity gate
    # ------------------------------------------------------------------

    def _run_sanity_gate(self, model: lgb.Booster, feature_names: List[str]) -> None:
        """Check that the model is not over-relying on noise features.

        Inspects the ``TOP_K_FEATURES`` most important features by *gain*.
        If every one of them belongs to ``NOISE_FEATURES`` the gate fails.

        Args:
            model: A trained LightGBM Booster.
            feature_names: Ordered list of feature names matching the
                training matrix columns.
        """
        importance: List[float] = model.feature_importance(importance_type="gain").tolist()
        name_importance = list(zip(feature_names, importance))
        name_importance.sort(key=lambda pair: pair[1], reverse=True)

        top_k = name_importance[:TOP_K_FEATURES]
        top_names = [name for name, _ in top_k]

        logger.info(
            "Feature importance sanity gate — top-%d by gain: %s",
            TOP_K_FEATURES,
            top_k,
        )

        if all(name in NOISE_FEATURES for name in top_names):
            logger.warning(
                "SANITY GATE FAILED: all top-%d features (%s) are noise features. "
                "The model is likely overfitting to random short-lag returns.",
                TOP_K_FEATURES,
                top_names,
            )
            self.sanity_passed = False
        else:
            logger.info("Feature importance sanity gate PASSED.")
            self.sanity_passed = True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, model: lgb.Booster, path: Optional[str] = None) -> None:
        """Persist a trained Booster to disk.

        Args:
            model: The trained LightGBM Booster to save.
            path: Optional file path.  Defaults to
                ``<model_dir>/<_DEFAULT_MODEL_FILENAME>``.
        """
        save_path = Path(path) if path else self.model_dir / _DEFAULT_MODEL_FILENAME
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(save_path))
        logger.info("Model saved to %s", save_path)

    def load_model(self, path: Optional[str] = None) -> lgb.Booster:
        """Load a previously saved Booster from disk.

        Args:
            path: Optional file path.  Defaults to
                ``<model_dir>/<_DEFAULT_MODEL_FILENAME>``.

        Returns:
            The loaded ``lgb.Booster``.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        load_path = Path(path) if path else self.model_dir / _DEFAULT_MODEL_FILENAME
        if not load_path.exists():
            raise FileNotFoundError(f"No model file found at {load_path}")

        model = lgb.Booster(model_file=str(load_path))
        logger.info("Model loaded from %s", load_path)
        return model
