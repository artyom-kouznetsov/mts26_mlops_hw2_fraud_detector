import logging
import joblib
import numpy as np
import pandas as pd

from pathlib import Path


logger = logging.getLogger(__name__)


def load_model_artifact(model_path: Path) -> dict:
    """
    Load trained model artifact.
    """
    logger.info("Loading model artifact: %s", model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    model_artifact = joblib.load(model_path)

    logger.info("Model artifact loaded successfully")

    return model_artifact


def make_pred(
    processed_df: pd.DataFrame,
    model_artifact: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score processed data with trained model artifact.
    """
    model = model_artifact["model"]
    threshold = model_artifact["threshold"]
    selected_features = model_artifact["selected_features"]

    missing_features = [
        col for col in selected_features
        if col not in processed_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Selected features missing after preprocessing: "
            f"{missing_features}"
        )

    x_model = processed_df[selected_features].copy()

    probabilities = model.predict_proba(x_model)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    logger.info("Prediction completed. Shape: %s", x_model.shape)
    logger.info("Threshold: %.6f", threshold)
    logger.info("Positive prediction share: %.4f", predictions.mean())

    return predictions, probabilities


def get_top_feature_importances(
    model_artifact: dict,
    top_n: int = 5,
) -> dict:
    """
    Extract top feature importances from fitted LightGBM classifier.
    """
    model = model_artifact["model"]
    selected_features = model_artifact["selected_features"]

    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_

    feature_importances = (
        pd.DataFrame(
            {
                "feature": selected_features,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    return dict(
        zip(
            feature_importances["feature"],
            feature_importances["importance"].astype(float),
        )
    )
