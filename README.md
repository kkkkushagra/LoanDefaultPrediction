# Loan Default Prediction

A prototype for scoring Home Credit loan applications with a fitted machine-learning model.
It includes a FastAPI service and a React dashboard. It is an academic demonstration, not
a production credit-decision system.

## Status

| Area | Status | What is present |
|---|---|---|
| Data preprocessing and feature pipeline | Done | Source code and fitted transformers are included. Running the data pipeline still requires the source dataset. |
| Model scoring API | Done | `/health`, `/predict`, `/explain`, `/simulate`, `/application-options`, `/model-info`, and `/portfolio/summary`. |
| Web dashboard | Done | Application scoring, what-if comparisons, model information, and a portfolio summary view. |
| Model explainability and threshold | Prototype | Model associations and a cost-based cutoff are exposed; they are not causal or validated operating policies. |
| Portfolio reporting | Prototype | A static aggregate in `docs/portfolio_summary.json`; it is not refreshed from a live portfolio. |
| Authentication, database, case management, and human review workflow | Not implemented | The API has no authentication or persistence, and it does not route cases to reviewers. |
| Early-warning alerts, production monitoring, and live credit-bureau integration | Not implemented | No live data feeds, alerting, or production drift monitoring are connected. |
| Production readiness or regulatory certification | Not implemented | The model is not approved for real lending decisions. |

## Model results and limitations

The existing calibration report records held-out test AUC **0.781905**, KS **0.432412**,
and Brier score **0.067716** for the calibrated XGBoost model. These are results from one
Home Credit train/validation/test split, not a guarantee of future performance. See
[the calibration report](docs/calibration_report.md) and
[the model comparison](docs/model_comparison.md) for the recorded evaluation.

The current cost-based cutoff is **0.242685**, selected using validation data with assumed
false-negative and false-positive costs. On the held-out test split it produced **11.28%
recall** and **34.40% precision**; it is not a 70%-recall operating point. See the
[review-threshold evaluation](docs/review_threshold_evaluation.md). Costs are placeholders,
not observed lender costs. The portfolio screen shows precomputed aggregate test results
from `docs/portfolio_summary.json`; it does not read borrower-level test data at runtime.

Scores and explanations are decision-support demonstrations only. Model associations are
not causes, and the threshold must not be interpreted as a recommendation to approve,
decline, or manually review an application.

## Run locally

Use Python 3.12. Install the pinned dependencies and start the API:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

In another terminal, start the frontend:

```powershell
cd frontend
npm ci
npm run dev
```

The Vite development server proxies `/api` requests to `http://127.0.0.1:8000`. Open the
URL printed by Vite. Model serving uses the committed model and transformer artifacts;
it does not require `data/raw` or processed parquet files.

Check the API and submit the seven-field sample:

```powershell
curl.exe http://localhost:8000/health
$payload = @'
{"AMT_CREDIT":300000,"AMT_INCOME_TOTAL":100000,"AMT_ANNUITY":25000,"AMT_GOODS_PRICE":350000,"DAYS_EMPLOYED":-3653,"NAME_CONTRACT_TYPE":"Cash loans","NAME_INCOME_TYPE":"Working"}
'@
$payload | curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data-binary "@-"
```

`DAYS_EMPLOYED` is elapsed days and is normally negative for employed applicants.
`/application-options` lists accepted loan and income categories. Optional request fields
may be omitted.

## Run with Docker Compose

Docker Desktop with Compose is required. Optionally copy `.env.example` to `.env` to
override the host ports; the example contains no credentials.

```powershell
docker compose up --build
```

The backend is available at `http://localhost:8000` and the dashboard at
`http://localhost:8080`. Nginx serves the frontend and proxies `/api/` to the backend.
The backend image runs as a non-root user and has a `/health` container health check.

## Tests and CI

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests use synthetic fixtures and the committed model artifacts; they do not require
raw or processed data files. GitHub Actions runs the Python tests and builds the frontend
with `npm ci` and `npm run build`.

## Optional data pipeline

To run preprocessing and feature generation, place the Home Credit
`application_train.csv` at `data/application_train.csv`, install the requirements above,
and run:

```powershell
.\.venv\Scripts\python.exe -m src.data.run
```

This writes borrower-level intermediate data under the ignored `data/` directory. Do not
commit raw datasets, processed parquet files, or other borrower-level records. Only the
fitted `preprocessor.joblib` and `feature_pipeline.joblib` are tracked for serving.
Optional external-table aggregation scripts require their corresponding source files.

## Repository map

- `backend/`: FastAPI application and scoring service
- `frontend/`: React/Vite dashboard and production nginx configuration
- `src/`: data, feature, and model code
- `models/`: saved scoring models
- `data/processed/`: only the two fitted transformer artifacts are tracked
- `docs/`: reports and the static portfolio summary
- `tests/`: API, model, data, and feature tests

See the [project synopsis](Team36_ProjectSynopsis.pdf), [EDA report](docs/EDA_report.md),
and [feature report](docs/feature_report.md) for additional project context.
