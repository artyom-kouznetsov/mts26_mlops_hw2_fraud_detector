# ML fraud detection service

Project for MTS/Teta ML 2026 MLOps course.

Dataset source:
https://www.kaggle.com/competitions/teta-ml-1-2025

This service performs automatic fraud detection for transaction data in batch scoring mode. It processes `.csv` files from the mounted input directory using a pretrained LightGBM pipeline.

The service performs inference only. Model training is not performed inside the container.

## Solution architecture

```
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
│
├── app/
│   └── app.py                      # service core and file watcher
│
├── src/
│   ├── preprocessing.py            # data preprocessing pipeline
│   ├── scorer.py                   # model loading, prediction, feature importance
│   └── exporter.py                 # submission, json and plot export
│
├── models/
│   └── fraud_lgbm_pipeline.joblib  # serialized LightGBM sklearn pipeline
│
├── input/                          # directory for input csv files
├── output/                         # directory for scoring results
└── logs/                           # service logs, if mounted
```

## Key features

### Logging

- Console logging during container runtime
- File logging to `/app/logs/service.log`
- Current logging level: `INFO`

To save logs locally, mount `./logs` to `/app/logs`:

```bash
-v "$(pwd)/logs:/app/logs"
```

### Data preprocessing pipeline (`preprocessing.py`)

1. **Unused columns removal**
   - Drops personal / high-cardinality raw columns not used by the model:
     - `name_1`
     - `name_2`
     - `street`

2. **Categorical cleaning**
   - Cleans merchant names
   - Converts `post_code` to string type

3. **Time features**
   - Extracts hour, year, month, day of month, and day of week
   - Creates night/weekend indicators
   - Creates cyclic features for hour and month
   - Removes the original `transaction_time`

4. **Geospatial features**
   - Calculates customer–merchant distance
   - Creates latitude and longitude difference features
   - Creates distance transformations and buckets

5. **Numerical transformations**
   - Log-transformations for amount, population, and distance
   - Amount-to-distance and amount-to-population ratios

6. **Interaction features**
   - Category-hour interaction
   - Category-period interaction
   - Category-weekend interaction
   - State-category interaction

### Model layer (`scorer.py`)

- Automatic model loading during service initialization
- Batch scoring through `predict_proba`
- Classification threshold is loaded from the saved model artifact (~0.875)
- Top-5 feature importances are exported to `.json`

### Export layer (`exporter.py`)

- Exports prediction file in sample submission format
- Exports top-5 feature importances as `.json`
- Exports predicted score distribution plot as `.png`

## Quick start

### Requirements

- Docker 20.10+
- 2 GB free disk space
- No GPU required
- Ports: filesystem only

### Run the service

1. Build Docker image:

```bash
docker build -t fraud-detector-service .
```

2. Run the container with mounted volumes:

```bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  fraud-detector-service
```

3. After the message `File observer started`, add a competition-format `.csv` file to `./input`.


4. Wait for preprocessing and scoring to finish. The generated files will be saved to `./output`.

5. Stop the service after scoring (`Ctrl + C`).

## Output

For `input/test.csv`, the service creates:

    output/sample_submission_test.csv
    output/feature_importances_top5_test.json
    output/score_distribution_test.png


## Notes

- The input file must have the same format as `test.csv` from the competition.
- Any `.csv` filename can be used in the `input/` directory.
- `train.csv` is not required inside the container.
- The model artifact must already exist at `models/fraud_lgbm_pipeline.joblib`.
