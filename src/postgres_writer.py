import json
import logging
import os
import signal
import time
from datetime import datetime, timezone

import psycopg
from confluent_kafka import Consumer, KafkaError, TopicPartition


# --- persist scoring messages from kafka in the postgresql data mart ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("postgres-writer")

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC", "scores")
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://fraud:fraud@postgres:5432/fraud"
)
WRITER_BATCH_SIZE = int(os.getenv("WRITER_BATCH_SIZE", "500"))
WRITER_BATCH_TIMEOUT_SECONDS = float(
    os.getenv("WRITER_BATCH_TIMEOUT_SECONDS", "1")
)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transaction_scores (
    id BIGSERIAL PRIMARY KEY,
    transaction_id TEXT NOT NULL UNIQUE,
    score DOUBLE PRECISION NOT NULL CHECK (score >= 0 AND score <= 1),
    fraud_flag SMALLINT NOT NULL CHECK (fraud_flag IN (0, 1)),
    us_state TEXT,
    merch TEXT,
    cat_id TEXT,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

UPSERT_SQL = """
INSERT INTO transaction_scores (
    transaction_id, score, fraud_flag, us_state, merch, cat_id, processed_at
) VALUES (
    %(transaction_id)s, %(score)s, %(fraud_flag)s, %(us_state)s,
    %(merch)s, %(cat_id)s, %(processed_at)s
)
ON CONFLICT (transaction_id) DO UPDATE SET
    score = EXCLUDED.score,
    fraud_flag = EXCLUDED.fraud_flag,
    us_state = EXCLUDED.us_state,
    merch = EXCLUDED.merch,
    cat_id = EXCLUDED.cat_id,
    processed_at = EXCLUDED.processed_at
"""


def connect_with_retry() -> psycopg.Connection:
    while True:
        try:
            connection = psycopg.connect(DATABASE_URL)
            connection.execute(CREATE_TABLE_SQL)
            connection.commit()
            return connection
        except psycopg.OperationalError as exc:
            logger.warning("PostgreSQL is not ready: %s", exc)
            time.sleep(3)


def rewind_batch(consumer: Consumer, messages: list) -> None:
    """Rewind consumed partitions so a failed database batch can be retried."""
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


def enrich_result_from_headers(result: dict, kafka_message) -> dict:
    """
    Attach dashboard dimensions carried outside the three-field json value.
    """
    decoded_headers = {
        key: value.decode("utf-8") if value is not None else None
        for key, value in (kafka_message.headers() or [])
    }
    for field in ("us_state", "merch", "cat_id", "processed_at"):
        if field in decoded_headers:
            result[field] = decoded_headers[field]
        else:
            # support enriched json messages produced by earlier versions
            result.setdefault(field, None)

    if result["processed_at"] is None:
        result["processed_at"] = datetime.now(timezone.utc).isoformat()
    return result


def main() -> None:
    connection = connect_with_retry()
    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP_SERVERS,
            "group.id": os.getenv("WRITER_CONSUMER_GROUP", "postgres-writer-v1"),
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    running = True

    def stop(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    consumer.subscribe([OUTPUT_TOPIC])
    logger.info(
        "Listening to Kafka topic %s with batch_size=%d",
        OUTPUT_TOPIC,
        WRITER_BATCH_SIZE,
    )

    try:
        while running:
            kafka_messages = consumer.consume(
                num_messages=WRITER_BATCH_SIZE,
                timeout=WRITER_BATCH_TIMEOUT_SECONDS,
            )
            if not kafka_messages:
                continue

            try:
                results = []
                for kafka_message in kafka_messages:
                    if kafka_message.error():
                        if kafka_message.error().code() != KafkaError._PARTITION_EOF:
                            logger.error(
                                "Kafka consumer error: %s",
                                kafka_message.error(),
                            )
                        continue
                    try:
                        results.append(
                            enrich_result_from_headers(
                                json.loads(
                                    kafka_message.value().decode("utf-8")
                                ),
                                kafka_message,
                            )
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        logger.exception("Skipping malformed scoring message")

                if results:
                    with connection.cursor() as cursor:
                        cursor.executemany(UPSERT_SQL, results)
                connection.commit()
                # acknowledge kafka only after the database transaction commits
                consumer.commit(asynchronous=False)
                logger.info("Saved scoring batch: transactions=%d", len(results))
            except (psycopg.InterfaceError, psycopg.OperationalError):
                logger.exception("PostgreSQL connection lost; reconnecting")
                try:
                    connection.close()
                except Exception:
                    pass
                connection = connect_with_retry()
                rewind_batch(consumer, kafka_messages)
            except Exception:
                try:
                    connection.rollback()
                except psycopg.Error:
                    pass
                logger.exception("Failed to persist scoring batch; rewinding")
                rewind_batch(consumer, kafka_messages)
                time.sleep(1)
    finally:
        consumer.close()
        connection.close()


if __name__ == "__main__":
    main()
