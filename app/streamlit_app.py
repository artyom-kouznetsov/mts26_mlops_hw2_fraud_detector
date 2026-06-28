import json
import os
import time
import uuid

import numpy as np
import pandas as pd
import psycopg
import streamlit as st
from confluent_kafka import Producer


# --- Streamlit producer and scoring results ui ---
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "transactions")
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://fraud:fraud@postgres:5432/fraud"
)
GRAFANA_PORT = os.getenv("GRAFANA_PORT", "3000")
KAFKA_UI_PORT = os.getenv("KAFKA_UI_PORT", "8080")

# minimum input schema required by feature engineering and model inference
REQUIRED_COLUMNS = {
    "transaction_time",
    "merch",
    "cat_id",
    "amount",
    "gender",
    "one_city",
    "us_state",
    "post_code",
    "lat",
    "lon",
    "population_city",
    "jobs",
    "merchant_lat",
    "merchant_lon",
}


@st.cache_resource
def kafka_producer() -> Producer:
    """Reuse one producer across streamlit reruns."""
    return Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})


def json_safe(value: object) -> object:
    """Convert pandas and numpy scalars into json compatible values."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def row_to_message(row: pd.Series) -> dict:
    """Convert one csv row and guarantee a stable transaction identifier."""
    message = {key: json_safe(value) for key, value in row.items()}

    # preserve a source identifier when possible and generate one otherwise
    source_id = message.get("transaction_id")
    if source_id in (None, ""):
        source_id = message.get("id")
    if source_id in (None, ""):
        source_id = uuid.uuid4()
    message["transaction_id"] = str(source_id)
    return message


def publish_dataframe(data: pd.DataFrame) -> int:
    """Publish every dataframe row to the input kafka topic."""
    producer = kafka_producer()
    delivery_errors: list[str] = []

    # kafka delivery happens asynchronously, so callbacks collect failures
    def delivered(error, _message) -> None:
        if error is not None:
            delivery_errors.append(str(error))

    for _, row in data.iterrows():
        message = row_to_message(row)
        while True:
            try:
                producer.produce(
                    INPUT_TOPIC,
                    key=message["transaction_id"].encode("utf-8"),
                    value=json.dumps(message).encode("utf-8"),
                    callback=delivered,
                )
                break
            except BufferError:
                # poll delivery callbacks until the local 
                # producer queue has room
                producer.poll(0.5)
        producer.poll(0)

    # ensure that every queued row is acknowledged before success
    remaining = producer.flush(30)
    if remaining or delivery_errors:
        details = delivery_errors[0] if delivery_errors else "delivery timeout"
        raise RuntimeError(f"Kafka delivery failed: {details}")
    return len(data)


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two result sets displayed by the ui."""
    with psycopg.connect(DATABASE_URL) as connection:
        fraud = pd.read_sql_query(
            """
            SELECT transaction_id, score, fraud_flag, us_state, merch, cat_id,
                   processed_at
            FROM transaction_scores
            WHERE fraud_flag = 1
            ORDER BY processed_at DESC
            LIMIT 10
            """,
            connection,
        )
        scores = pd.read_sql_query(
            """
            SELECT score
            FROM transaction_scores
            ORDER BY processed_at DESC
            LIMIT 100
            """,
            connection,
        )
    return fraud, scores


st.set_page_config(page_title="Fraud detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Streaming fraud detector")
st.caption("CSV → Kafka → ML inference → PostgreSQL → Grafana")

with st.sidebar:
    st.header("Services")
    st.markdown(f"[Grafana](http://localhost:{GRAFANA_PORT})")
    st.markdown(f"[Kafka UI](http://localhost:{KAFKA_UI_PORT})")
    st.info("The model runs inference on CPU only.")

upload_tab, results_tab = st.tabs(["Send transactions", "Results"])

# validate the uploaded csv before publishing its rows to kafka
with upload_tab:
    uploaded_file = st.file_uploader("Upload test.csv", type=["csv"])
    if uploaded_file is not None:
        try:
            dataframe = pd.read_csv(uploaded_file)
            missing = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                st.write(f"Rows: {len(dataframe):,}")
                st.dataframe(dataframe.head(10), use_container_width=True)
                if st.button("Send to Kafka", type="primary"):
                    with st.spinner("Publishing transactions..."):
                        count = publish_dataframe(dataframe)
                    st.success(f"Published to `{INPUT_TOPIC}`: {count:,}")
        except Exception as exc:
            st.error(f"Could not process CSV: {exc}")

# query postgresql only when the user explicitly requests fresh results
with results_tab:
    if st.button("View results", type="primary"):
        # allow a just published batch to reach the database before querying
        time.sleep(0.5)
        try:
            fraud_rows, recent_scores = load_results()
            st.subheader("Latest 10 transactions with fraud_flag = 1")
            if fraud_rows.empty:
                st.info("No fraudulent transactions yet.")
            else:
                st.dataframe(fraud_rows, use_container_width=True)

            st.subheader("Score distribution for the latest 100 transactions")
            if recent_scores.empty:
                st.info("No scoring results yet.")
            else:
                counts, edges = np.histogram(
                    recent_scores["score"], bins=np.linspace(0, 1, 21)
                )
                histogram = pd.DataFrame(
                    {
                        "score": [
                            f"{left:.2f}–{right:.2f}"
                            for left, right in zip(edges[:-1], edges[1:])
                        ],
                        "transactions": counts,
                    }
                ).set_index("score")
                st.bar_chart(histogram)
        except Exception as exc:
            st.error(f"Could not load results: {exc}")
