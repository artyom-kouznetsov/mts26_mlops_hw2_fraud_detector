# Realtime fraud detector

This educational MLOps project implements streaming fraud scoring. A user
uploads `test.csv` through Streamlit, each row travels through Kafka and a
pretrained LightGBM model, and the result is published to a second Kafka topic
and stored in PostgreSQL. Streamlit displays recent results, while a provisioned
Grafana dashboard visualizes model and processing metrics.

The containers perform inference only. No model training or GPU is required.

The project extends the first homework service and reuses its model and feature
engineering. 

## Architecture

```text
                         ┌──────────────┐
test.csv ─► Streamlit ─► │ transactions │
                         │    Kafka     │
                         └──────┬───────┘
                                ▼
                  preprocessing + LightGBM
                                │
                         ┌──────▼───────┐
                         │    scores    │
                         │    Kafka     │
                         └──────┬───────┘
                                ▼
                     PostgreSQL writer
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
            Streamlit results        Grafana dashboard
```

| Container | Purpose |
|---|---|
| `kafka` | Kafka 3.9 in KRaft mode without ZooKeeper |
| `kafka-init` | Creates the `transactions` and `scores` topics |
| `scorer` | Reads transactions, preprocesses them, and runs inference |
| `postgres-writer` | Reads `scores` and upserts results into PostgreSQL |
| `postgres` | Stores the `transaction_scores` data mart and indexes |
| `interface` | Uploads CSV files and displays results in Streamlit |
| `kafka-ui` | Displays Kafka topics and messages |
| `grafana` | Displays the provisioned PostgreSQL dashboard |

## Message contracts

The `transactions` topic receives one JSON record per CSV row. If the source
file has neither `transaction_id` nor `id`, the UI generates a UUID.

The JSON value in `scores` contains exactly three required fields:

```json
{
  "transaction_id": "de67cdda-5c65-4c42-881c-e02f20d09272",
  "score": 0.912341,
  "fraud_flag": 1
}
```

The `us_state`, `merch`, and `cat_id` fields, along with the `processed_at`
timestamp, are passed in Kafka headers. The writer stores them in PostgreSQL for
Grafana filters and throughput calculations.

The scorer processes microbatches of 256 messages, and the writer persists
batches of up to 500 rows. Batch sizes and timeouts can be changed in `.env`
through `SCORER_BATCH_SIZE`, `SCORER_BATCH_TIMEOUT_SECONDS`,
`WRITER_BATCH_SIZE`, and `WRITER_BATCH_TIMEOUT_SECONDS`.

## Quick start

Requirements:

- Docker Engine;
- Docker Compose v2;
- at least 4 GB of free RAM.

Host ports are configured in `.env`. The default PostgreSQL host port is
`55432` to avoid conflicts with locally installed PostgreSQL. Containers still
use port `5432` inside the Compose network.

1. Create the local configuration file:

   ```bash
   cp .env.example .env
   ```

2. Build and start the complete stack:

   ```bash
   docker compose up --build -d
   ```

3. Check container status:

   ```bash
   docker compose ps -a
   ```

   The one-time `kafka-init` service should report `Exited (0)`. The remaining
   services should be `Up` or `healthy`.

4. Open [Streamlit](http://localhost:8501), upload a competition-format
   `test.csv`, and select **Send to Kafka**. For a quick local check, use
   `input/test.csv` if it exists in the working copy.

5. Open the **Results** tab and select **View results**. The interface displays:

   - up to 10 latest transactions with `fraud_flag = 1`;
   - a histogram of scores from the latest 100 transactions.

6. Open [Grafana](http://localhost:3000). The default credentials are
   `admin` / `admin`. Open **Fraud detector / Realtime fraud scoring** to view:

   - score distribution density;
   - processing throughput (transactions per second);
   - average fraud share by `cat_id` over the latest 1000 transactions;
   - multi-select state and merchant filters.

Kafka messages are available in [Kafka UI](http://localhost:8080) under the
`local` cluster and the `transactions` or `scores` topic.

## Command line verification

Count persisted results:

```bash
docker compose exec postgres psql -U fraud -d fraud -c \
  "SELECT COUNT(*) FROM transaction_scores;"
```

Follow streaming-service logs:

```bash
docker compose logs -f scorer postgres-writer interface
```

## PostgreSQL data mart

The `transaction_scores` table is created automatically from `db/init.sql`.

| Field | Type | Purpose |
|---|---|---|
| `transaction_id` | `TEXT UNIQUE` | Transaction identifier |
| `score` | `DOUBLE PRECISION` | Fraud probability from 0 to 1 |
| `fraud_flag` | `SMALLINT` | Model decision: 0 or 1 |
| `us_state` | `TEXT` | State field for Grafana |
| `merch` | `TEXT` | Merchant field for Grafana |
| `cat_id` | `TEXT` | Product category |
| `processed_at` | `TIMESTAMPTZ` | Inference completion time |

The writer uses `ON CONFLICT (transaction_id) DO UPDATE`, so replaying a Kafka
message does not create a duplicate row.

## Repository structure

```text
├── app/
│   ├── app.py                    # legacy batch service from part one
│   └── streamlit_app.py          # kafka producer and results ui
├── db/init.sql                   # postgresql data-mart initialization
├── grafana/
│   ├── dashboards/              # dashboard json
│   └── provisioning/            # datasource and dashboard provider
├── models/fraud_lgbm_pipeline.joblib
├── src/
│   ├── kafka_scorer.py           # transactions to scores
│   ├── postgres_writer.py        # scores to postgresql
│   ├── preprocessing.py          # feature engineering
│   └── scorer.py                 # model loading and predict_proba
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Stop

Stop the project:

```bash
docker compose down
```

Use `docker compose down -v` to also remove all stored data.
