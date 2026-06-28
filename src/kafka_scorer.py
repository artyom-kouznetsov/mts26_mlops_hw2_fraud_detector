import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from confluent_kafka import Consumer, KafkaError, Producer, TopicPartition

from preprocessing import prepare_features
from scorer import load_model_artifact, make_pred


# --- Kafka-based fraud scoring service ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("kafka-scorer")

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "transactions")
OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC", "scores")
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", "/app/models/fraud_lgbm_pipeline.joblib")
)
BATCH_SIZE = int(os.getenv("SCORER_BATCH_SIZE", "256"))
BATCH_TIMEOUT_SECONDS = float(os.getenv("SCORER_BATCH_TIMEOUT_SECONDS", "1"))


def _clean_merchant(value: object) -> str | None:
    if value is None:
        return None
    return str(value).replace("fraud_", "", 1)


def score_transactions(messages: list[dict], model_artifact: dict) -> list[dict]:
    """Preprocess and score a microbatch of kafka transaction messages."""
    transaction_ids = [str(message["transaction_id"]) for message in messages]
    raw_transactions = [
        {key: value for key, value in message.items() if key != "transaction_id"}
        for message in messages
    ]

    processed = prepare_features(pd.DataFrame(raw_transactions))
    fraud_flags, scores = make_pred(processed, model_artifact)
    processed_at = datetime.now(timezone.utc).isoformat()

    return [
        {
            "transaction_id": transaction_id,
            "score": float(score),
            "fraud_flag": int(fraud_flag),
            # keep dashboard dimensions outside the strict three-field payload
            "us_state": raw_transaction.get("us_state"),
            "merch": _clean_merchant(raw_transaction.get("merch")),
            "cat_id": raw_transaction.get("cat_id"),
            "processed_at": processed_at,
        }
        for transaction_id, raw_transaction, score, fraud_flag in zip(
            transaction_ids,
            raw_transactions,
            scores,
            fraud_flags,
        )
    ]


def rewind_batch(consumer: Consumer, messages: list) -> None:
    """Rewind consumed partitions so a failed microbatch can be retried."""
    earliest_offsets: dict[tuple[str, int], int] = {}
    for message in messages:
        if message.error():
            continue
        key = (message.topic(), message.partition())
        earliest_offsets[key] = min(
            earliest_offsets.get(key, message.offset()),
            message.offset(),
        )

    for (topic, partition), offset in earliest_offsets.items():
        consumer.seek(TopicPartition(topic, partition, offset))


def main() -> None:
    model_artifact = load_model_artifact(MODEL_PATH)
    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP_SERVERS,
            "group.id": os.getenv("SCORER_CONSUMER_GROUP", "fraud-scorer-v1"),
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})
    running = True

    def stop(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    consumer.subscribe([INPUT_TOPIC])
    logger.info(
        "Listening to Kafka topic %s with batch_size=%d",
        INPUT_TOPIC,
        BATCH_SIZE,
    )

    try:
        while running:
            kafka_messages = consumer.consume(
                num_messages=BATCH_SIZE,
                timeout=BATCH_TIMEOUT_SECONDS,
            )
            if not kafka_messages:
                continue

            try:
                transactions = []
                for kafka_message in kafka_messages:
                    if kafka_message.error():
                        if kafka_message.error().code() != KafkaError._PARTITION_EOF:
                            logger.error(
                                "Kafka consumer error: %s",
                                kafka_message.error(),
                            )
                        continue
                    try:
                        transactions.append(
                            json.loads(kafka_message.value().decode("utf-8"))
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        logger.exception("Skipping malformed transaction message")

                results = (
                    score_transactions(transactions, model_artifact)
                    if transactions
                    else []
                )
                delivery_errors: list[str] = []

                def delivered(error, _message) -> None:
                    if error is not None:
                        delivery_errors.append(str(error))

                for result in results:
                    score_payload = {
                        "transaction_id": result["transaction_id"],
                        "score": result["score"],
                        "fraud_flag": result["fraud_flag"],
                    }
                    context_headers = [
                        (
                            field,
                            str(result[field]).encode("utf-8")
                            if result[field] is not None
                            else None,
                        )
                        for field in (
                            "us_state",
                            "merch",
                            "cat_id",
                            "processed_at",
                        )
                    ]
                    producer.produce(
                        OUTPUT_TOPIC,
                        key=result["transaction_id"].encode("utf-8"),
                        value=json.dumps(score_payload).encode("utf-8"),
                        headers=context_headers,
                        callback=delivered,
                    )
                undelivered = producer.flush(10)
                if undelivered or delivery_errors:
                    details = (
                        delivery_errors[0]
                        if delivery_errors
                        else f"{undelivered} message(s) remain queued"
                    )
                    raise RuntimeError(
                        f"Kafka did not deliver scoring result: {details}"
                    )
                # commit only after every result in the batch reaches kafka
                consumer.commit(asynchronous=False)
                logger.info(
                    "Scored batch: transactions=%d fraud=%d",
                    len(results),
                    sum(result["fraud_flag"] for result in results),
                )
            except Exception:
                logger.exception("Failed to score Kafka microbatch; rewinding")
                rewind_batch(consumer, kafka_messages)
                time.sleep(1)
    finally:
        producer.flush(10)
        consumer.close()


if __name__ == "__main__":
    main()
