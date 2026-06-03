import logging
import sys
import time
import pandas as pd

from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from exporter import (
    save_feature_importances,
    save_score_distribution,
    save_submission,
)
from preprocessing import prepare_features
from scorer import (
    get_top_feature_importances,
    load_model_artifact,
    make_pred,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/logs/service.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info("Initializing ProcessingService...")

        self.input_dir = Path("/app/input")
        self.output_dir = Path("/app/output")
        self.model_path = Path("/app/models/fraud_lgbm_pipeline.joblib")

        self.model_artifact = load_model_artifact(self.model_path)

        logger.info("Service initialized")

    def process_single_file(self, file_path):
        try:
            file_path = Path(file_path)

            logger.info("Processing file: %s", file_path)

            input_df = pd.read_csv(file_path)
            logger.info("Input shape: %s", input_df.shape)

            processed_df = prepare_features(input_df)

            predictions, probabilities = make_pred(
                processed_df,
                self.model_artifact,
            )

            input_name = file_path.stem

            submission_path = (
                self.output_dir / f"sample_submission_{input_name}.csv"
            )
            importances_path = (
                self.output_dir / f"feature_importances_top5_{input_name}.json"
            )
            plot_path = (
                self.output_dir / f"score_distribution_{input_name}.png"
            )

            save_submission(
                input_df=input_df,
                predictions=predictions,
                output_path=submission_path,
            )

            top_importances = get_top_feature_importances(
                self.model_artifact,
                top_n=5,
            )

            save_feature_importances(
                feature_importances=top_importances,
                output_path=importances_path,
            )

            save_score_distribution(
                probabilities=probabilities,
                output_path=plot_path,
            )

            logger.info("Processing finished for: %s", file_path)

        except Exception as exc:
            logger.error(
                "Error processing file %s: %s",
                file_path,
                exc,
                exc_info=True,
            )


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if event.is_directory:
            return

        if event.src_path.endswith(".csv"):
            logger.info("New .csv file detected: %s", event.src_path)
            time.sleep(1)
            self.service.process_single_file(event.src_path)


if __name__ == "__main__":
    logger.info("Starting ML scoring service...")

    service = ProcessingService()

    existing_csv_files = sorted(service.input_dir.glob("*.csv"))

    if existing_csv_files:
        logger.info(
            "Found %d existing .csv file(s). Processing them first.",
            len(existing_csv_files),
        )

        for csv_file in existing_csv_files:
            service.process_single_file(csv_file)

    observer = Observer()
    observer.schedule(
        FileHandler(service),
        path=str(service.input_dir),
        recursive=False,
    )
    observer.start()

    logger.info("File observer started")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
        observer.stop()

    observer.join()
