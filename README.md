# Opioid Use Disorder (OUD) Risk Prediction Pipeline

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![MLOps Standard](https://img.shields.io/badge/MLOps-Modular-success)

**Author:** MLOps Instructor (Ivan Diaz)  
**Course:** MLOps: Master in Business Analytics and Data Science  
**Status:** Production-Ready (Modularized Pipeline Phase 0.5)

---

## 1. Business Objective

Healthcare systems and insurers currently rely on lagging indicators (e.g., overdoses) to identify Opioid Use Disorder (OUD). This project shifts the paradigm from reactive treatment to proactive intervention.

* **The Goal:** Predict a patient's risk of developing OUD based on a 2-year medical history, prescription supply, and socioeconomic indicators.
* **The User:** Clinical Care Managers who use the model's probability scores to prioritize high-risk patients for alternative pain management or addiction prevention resources.
* **In Scope:** A repeatable, auditable MLOps pipeline generating binary classifications and operational probability scores.
* **Out of Scope:** Automated clinical diagnosis, causal inference, or real-world prescribing limits.

---

## 2. ML Pipeline Architecture

This repository transitions from a fragile Jupyter Notebook into a testable software engineering architecture. The pipeline enforces strict **Separation of Concerns** to prevent data leakage and ensure reproducibility.

* **1. Ingestion (`load_data.py`):** Deterministic raw data loading.
* **2. Preprocessing (`clean_data.py`):** Stateless formatting and NA handling.
* **3. Quality Gates (`validate.py`):** "Fail-fast" schema and domain boundary checks.
* **4. Feature Engineering (`features.py`):** Unfitted `ColumnTransformer` recipes.
* **5. Training & Artifacts (`train.py`):** Bundling features and algorithms into a single deployable `.joblib` Pipeline.
* **6. Evaluation & Inference (`evaluate.py`, `infer.py`):** Isolated metric computation and probability generation.

---

## 3. Success Metrics & Fairness

* **Business KPI:** Increase the early-intervention outreach yield, reducing the 12-month incidence rate of severe OUD-related emergency room admissions.
* **Technical Metric:** **F1-Score** on the Validation set. We must balance catching true cases (Recall) without overwhelming care managers with false alarms (Precision).
* **Acceptance Criteria (Fairness):** The model must maintain **Equalized Odds**. The True Positive Rate (TPR) and False Positive Rate (FPR) difference between low-income (`Low_inc = 1`) and non-low-income cohorts must remain below strict operational thresholds.

---

## 4. The Data

**Data Sensitivity Warning:** This teaching dataset contains synthetic, de-identified claims. In real healthcare contexts, this is Protected Health Information (PHI) and must **never** be committed to public version control.

* **Snapshot:** 1000 rows, 22 columns. Unit of analysis is a member record over a 2-year lookback.
* **Prevalence:** Positive class (`OD=1`) is 18.1%.
* **Target (`OD`):** Opioid abuse disorder, operationalized as ICD-10 F11 related claims within 2 years.

<details>
<summary><b>Click to expand Data Dictionary</b></summary>

> *Note: `rx ds` is standardized to `rx_ds` during cleaning to stabilize downstream contracts.*

| Feature | Description |
|---|---|
| OD | Target: Opioid abuse disorder indicator (F11 claims) |
| Low_inc | Low income flag (1 = low income) |
| SURG | Major surgery in the last 2 years |
| rx_ds | Days supply filled for opioid drugs over 2 years |
| A - V | Various ICD-10 chapter flags (Infectious diseases, Neoplasms, etc.) |
</details>

---

## 5. Getting Started (How to Run)

### Step 1: Environment Setup
Ensure complete dependency reproducibility by building the Conda environment:
```bash
conda env create -f environment.yml
conda activate mlops-modul
```

### Step 2: Ensure Pipeline Quality (Testing)
Run the unit test suite to verify mathematically sound logic and unbroken data contracts before executing heavy compute:
```bash
pytest -q
```
*(Expected output: 100% passing tests)*

### Step 3: Execute the Orchestrator
The orchestrator (`src/main.py`) is the automated "factory". It provides deterministic execution and is the *only* entry point authorized to write canonical production artifacts to your hard drive.
```bash
python -m src.main
```

**Outputs Generated:**
1. `data/processed/clean.csv`: Deterministically cleaned input data.
2. `models/model.joblib`: Deployable Pipeline artifact (preventing training-serving skew).
3. `reports/predictions.csv`: Inference log containing predictions and probabilities.

### Optional: Exploratory Sandbox
Before running the automated pipeline, interactively explore data using the provided notebooks (`notebooks/`). These run entirely in memory and do not corrupt production artifacts.
```bash
jupyter notebook notebooks/01_opioid_analysis_vExp.ipynb
```

---

## 6. Repository Structure

```text
.
├── config.yaml               # Centralized configuration (hyperparameters, paths)
├── environment.yml           # Conda dependency manager
├── pytest.ini                # Pytest configuration
├── data/
│   ├── raw/                  # Immutable source data
│   └── processed/            # Cleaned data for training/inference
├── models/                   # Serialized model artifacts (.joblib)
├── notebooks/                # Exploratory Data Analysis (EDA) sandboxes
├── reports/                  # Generated predictions, metrics, and logs
├── src/                      # Core MLOps Python modules
│   ├── clean_data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── infer.py
│   ├── load_data.py
│   ├── main.py               # Pipeline orchestrator
│   ├── train.py
│   ├── utils.py
│   └── validate.py
└── tests/                    # Unit tests for CI/CD pipelines
    └── test_*.py             # Tests matching src/ modules
```

---

## 7. Educational Roadmap

**Current Session Achievements:**
- Translated a notebook workflow into a `src/` layout with explicit module contracts.
- Implemented baseline testing (`pytest`) and quality gates.
- Enforced leakage prevention by design through strict split/fit boundaries.

**Upcoming Sessions:**
- Migrate all hardcoded `SETTINGS` into `config.yaml` and use `.env` for secrets
- Replace `print()` statements with Python standard library **logging** (consistent, structured, reusable across modules)
- Add **Weights & Biases (W&B)** for experiment tracking, including basic data fingerprinting for provenance
- Use **W&B artifacts and model registry** to register and retrieve a pinned model version for reproducible inference
- Serve predictions via a **FastAPI** application with a clear request and response contract and a `/health` endpoint
- Containerize the inference service with **Docker** and validate it runs locally end to end
- Implement Continuous Integration (CI/CD) quality checks using **GitHub Actions**
- Deploy the live service to **Render** with automated deploys