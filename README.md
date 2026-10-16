# AI-Powered Loan Default Prediction System

A project focused on building an explainable machine-learning solution for predicting loan default risk in banking and NBFC workflows. The system is designed to help lenders assess borrower risk early, reduce credit losses, and provide transparent reasoning for each decision.

## Project Overview

Banks and lenders often rely on static scorecards and manual judgment, which can miss non-linear risk patterns and delay intervention. This project introduces an AI-driven risk scoring workflow that evaluates applicant data, transforms it into risk-aware features, and predicts the probability of default using machine learning models.

The project emphasizes:
- early-risk detection
- transparency through explainability
- operational usability for loan officers and risk managers
- API-driven integration with loan origination systems
- monitoring and retraining support for future production use

## Key Objectives

- reduce default-related financial losses
- detect risky borrowers before disbursal or in early stages of the loan lifecycle
- provide explainable outputs using SHAP-style feature importance
- support governance, auditability, and decision transparency
- create a working demo and technical design suitable for academic evaluation

## Project Scope

### In Scope
- dataset ingestion and validation
- preprocessing and feature engineering
- model training and comparison
- explainability and reason codes
- real-time scoring API
- loan officer dashboard
- historical trend analysis and monitoring

### Out of Scope
- real bank production deployment
- live credit-bureau integration
- fully automated approval or rejection decisions
- production regulatory certification in v1

## Proposed Solution

The solution combines ML and explainability to produce a risk score and a risk band for each applicant. Each prediction includes a probability of default and the top contributing features behind the result, enabling loan officers to understand why a borrower was flagged.

## Technologies and Tools

### Data and Machine Learning
- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- SHAP
- Optuna
- MLflow

### Backend and API
- FastAPI
- Pydantic
- SQLAlchemy
- Alembic
- PostgreSQL

### Frontend and Visualization
- React
- Recharts

### DevOps and Deployment
- Docker
- GitHub Actions
- Render / AWS / Vercel-style deployment targets

## Quality and Evaluation Goals

The project targets the following demonstration quality benchmarks:
- AUC-ROC >= 0.78
- recall on defaulters >= 70%
- p95 scoring latency <= 300 ms
- top-5 SHAP-based reasons returned for each prediction
- dashboard and API workflows suitable for loan decision support

## Architecture Overview

The system is planned as a layered architecture:
1. clients and user interfaces
2. API layer for authentication and score requests
3. model/service layer for preprocessing and prediction
4. explainability layer for risk reason generation
5. storage and monitoring layer for scores, outcomes, and drift tracking

## User Roles

- Loan Officer: reviews applicant risk and explains decisions
- Risk Manager: monitors portfolio risk and trends
- Integration Engineer: connects the scoring service to existing systems

## Documentation

- Project synopsis PDF: [Team36_ProjectSynopsis.pdf](./Team36_ProjectSynopsis.pdf)
- This README provides the project overview and setup context

## Repository Structure

```text
LoanDefaultPrediction/
|-- README.md
|-- Team36_ProjectSynopsis.pdf
|-- project assets and source code (to be added as the solution develops)
```

## Current Status

This repository currently contains the project synopsis and project documentation foundation. The implementation phase, source modules, tests, and deployment files will be added as the project evolves.

## Team

- Team ID: 36
- Institute: GLA University, Mathura
- Course: B.Tech CSE (AIML and IoT)
- Semester: 7
- Project Lead: Kushagra Trivedi

## References

The synopsis references RBI financial stability insights, credit-risk literature, and explainable AI methods for credit scoring.

## License

This project is intended for academic and educational use as part of the Major Project submission. Please check the institution's guidance and repository policies before reuse or redistribution.
