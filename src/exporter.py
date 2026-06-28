import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


logger = logging.getLogger(__name__)


def save_submission(
    input_df: pd.DataFrame,
    predictions,
    output_path: Path,
) -> pd.DataFrame:
    """
    Save predictions in sample submission format.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "id" in input_df.columns:
        submission = pd.DataFrame(
            {
                "id": input_df["id"],
                "target": predictions,
            }
        )
    else:
        submission = pd.DataFrame(
            {
                "target": predictions,
            }
        )

    submission.to_csv(output_path, index=False)

    logger.info("Submission saved to: %s", output_path)
    logger.info("Submission shape: %s", submission.shape)

    return submission


def save_feature_importances(
    feature_importances: dict,
    output_path: Path,
) -> None:
    """
    Save top feature importances to json.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(feature_importances, file, indent=4)

    logger.info("Feature importances saved to: %s", output_path)


def save_score_distribution(
    probabilities,
    output_path: Path,
) -> None:
    """
    Save the predicted score distribution plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(probabilities, bins=50, density=True)
    plt.title("Predicted score distribution")
    plt.xlabel("Predicted fraud probability")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info("Score distribution plot saved to: %s", output_path)
