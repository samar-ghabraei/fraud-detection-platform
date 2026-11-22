# Credit Card Fraud Detection Platform

A machine learning project for detecting fraudulent credit card transactions using imbalanced classification, model comparison, cost-based threshold optimization, and a REST API built with FastAPI.

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem. Fraudulent transactions represent a very small proportion of all transactions, so accuracy alone is not a reliable evaluation metric.

This project aims to build a fraud detection pipeline that:

- Analyzes and preprocesses transaction data.
- Trains and compares multiple machine learning models.
- Optimizes classification thresholds using a cost-based approach.
- Evaluates model performance using appropriate classification metrics.
- Exposes predictions through a REST API.
- Validates API inputs and outputs using Pydantic.
- Tests API endpoints using Pytest.

## Dataset

The project uses the Credit Card Fraud Detection dataset.

**Dataset characteristics:**
- 284,807 transactions
- 492 fraudulent transactions
- 284,315 normal transactions
- 30 input features
- Binary target variable: `Class`

The dataset contains anonymized PCA-transformed features (`V1` to `V28`), as well as `Time` and `Amount`.

The dataset is highly imbalanced, with fraudulent transactions representing approximately 0.173% of all transactions.

**Note:** The dataset file is not included in this repository. Place `creditcard.csv` inside the `data/` directory before running the training scripts.

## Machine Learning Models

Three classification models were trained and compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

The classification threshold was optimized using a cost-based approach.

### Cost Function

The project uses illustrative business cost assumptions:

- False Negative (FN): 1,000,000 Toman
- False Positive (FP): 20,000 Toman

The total cost is calculated as:

`Total Cost = (FN × Fraud Loss) + (FP × Review Cost)`

These values are hypothetical assumptions used for model comparison, not actual banking costs.

## Model Evaluation

The models were evaluated using Precision, Recall, F1-score, and total estimated cost.

| Model | Threshold | Precision | Recall | F1-score | Total Cost (Toman) |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.15 | 0.7213 | 0.8980 | 0.8000 | 5,340,000 |
| XGBoost | 0.08 | 0.3409 | 0.9184 | 0.4972 | 5,740,000 |
| Logistic Regression | 0.98 | 0.5455 | 0.8571 | 0.6667 | 7,700,000 |

**Selected Model: Random Forest**

Random Forest achieved the lowest estimated cost and the highest F1-score among the evaluated models.

It was selected as the model used by the prediction API.

**Evaluation limitation:** Model comparison was performed on the test split. Therefore, the test set was used during model selection and should not be considered a completely untouched final holdout set.

## API Development

A REST API was developed using FastAPI.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Predict whether a transaction is fraudulent |

### Prediction Response Example

```json
{
  "prediction": 1,
  "result": "Fraud",
  "fraud_probability": 1.0,
  "threshold": 0.15
}
```

The API uses Pydantic models to validate transaction inputs and prediction responses.

Interactive API documentation is available through Swagger UI at:

`http://localhost:8000/docs`

## Testing

API tests are implemented using Pytest and FastAPI TestClient.

The current test suite covers:

- Home endpoint
- Health endpoint
- Valid transaction prediction
- Missing input fields
- Negative transaction amount

Run the tests using Docker:

```bash
docker run --rm -v "$PWD:/app" -w /app fraud-detection-ml bash -c "pip install pytest httpx && python -m pytest -v"
```

## Technologies

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Pydantic
- Uvicorn
- Pytest
- Docker
- Git & GitHub

## Project Structure

```text
fraud-detection-platform/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── config/
├── dags/
├── data/
├── docker/
├── models/
├── notebooks/
├── reports/
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
│   └── monitoring/
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
├── requirements-dev.txt
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

## Running the API

### 1. Build the Docker image

```bash
docker build -f docker/Dockerfile.ml -t fraud-detection-ml .
```

### 2. Prepare the dataset and train the model

Place the dataset at:

`data/creditcard.csv`

Then run:

```bash
docker run --rm -v "$PWD:/app" -w /app fraud-detection-ml python src/models/train_random_forest.py
```

This step should generate the selected model artifact at:

`models/fraud_detection_random_forest.joblib`

### 3. Start the API

```bash
docker run --rm -p 8000:8000 -v "$PWD:/app" -w /app fraud-detection-ml uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 4. Open API documentation

Visit:

`http://localhost:8000/docs`

## Limitations

- The dataset contains only two days of transactions from 2013.
- Features V1–V28 are anonymized PCA-transformed variables.
- The dataset does not provide customer or merchant identifiers.
- The cost assumptions are illustrative and require real business data for practical deployment.
- The test split was used during model comparison, limiting its independence as a final evaluation set.
- The project is a portfolio and learning project, not a production-ready banking fraud detection system.

## Future Improvements

- Add database integration for storing predictions.
- Implement model monitoring and data drift detection.
- Add automated model retraining.
- Improve API integration and deployment.
- Add more comprehensive integration tests.
