# Opioid Use Disorder (OUD) Risk Prediction

**Author:** MLOps Instructor (Ivan Diaz)
**Course:** MLOps: Master in Business Analytics and Data Science  
**Status:** Production-Ready (Modularized Pipeline)

---

## 1. Business Objective

Healthcare systems and insurers currently rely on lagging indicators (e.g., overdoses) to identify Opioid Use Disorder (OUD). This project shifts the paradigm from reactive treatment to proactive intervention.

* **The Goal:** Predict a patient's risk of developing OUD based on a 2-year medical history, prescription supply, and socioeconomic indicators.
* **The User:** Clinical Care Managers who use the model's probability scores to prioritize high-risk patients for alternative pain management or addiction prevention resources.
* **In Scope:** A repeatable, auditable MLOps pipeline generating binary classifications and operational probability scores.
* **Out of Scope:** Automated clinical diagnosis, causal inference, or real-world prescribing limits.

---

## 2. Success Metrics

* **Business KPI:** Increase the early-intervention outreach yield, reducing the 12-month incidence rate of severe OUD-related emergency room admissions.
* **Technical Metric:** **F1-Score** on the Validation set. We must balance catching true cases (Recall) without overwhelming care managers with false alarms (Precision).
* **Acceptance Criteria (Fairness):** The model must maintain **Equalized Odds**. The True Positive Rate (TPR) and False Positive Rate (FPR) difference between low-income (`Low_inc = 1`) and non-low-income cohorts must remain below strict operational thresholds.

---

## 3. The Data

### Source and unit of analysis
- Synthetic, de identified claims style dataset for teaching purposes
- Unit of analysis is an individual member record summarised over a 2 year lookback period

### Dataset snapshot
- Rows: 1000
- Columns: 22 (including `ID` and target)
- Positive class prevalence (`OD=1`): 18.1% (181 of 1000)
- `rx_ds` range: 1 to 1699 days supplied over 2 years

### Target definition
- `OD`: whether someone had opioid abuse disorder, operationalised as having International Classification of Diseases, Tenth Revision (ICD-10) F11 related claims within 2 years

### Data sensitivity
- This teaching dataset contains no direct personal identifiers
- In real healthcare contexts, this type of data is sensitive and must be treated as protected health information, never committed to public version control

### Data Dictionary
> Notes  
> `rx ds` is standardised to `rx_ds` in `clean_data.py` to keep downstream contracts stable  

| Feature | Description |
|---|---|
| OD | Opioid abuse disorder indicator based on F11 claims in 2 years |
| Low_inc | Low income flag, 1 means low income |
| SURG | Major surgery in the last 2 years |
| rx_ds | Days supply filled for opioid drugs over 2 years |
| A | Infectious diseases group A ICD-10 chapter flag |
| B | Infectious diseases group B ICD-10 chapter flag |
| C | Malignant neoplasm flag |
| D | Benign neoplasm flag |
| E | Endocrine conditions flag |
| F | Mental and behavioural health conditions excluding F11 related |
| H | Ear conditions flag |
| I | Circulatory system conditions flag |
| J | Respiratory system conditions flag |
| K | Digestive system conditions flag |
| L | Skin conditions flag |
| M | Musculoskeletal system conditions flag |
| N | Urinary system conditions flag |
| R | Other signs and symptoms flag |
| S | Injuries flag |
| T | Burns and toxic conditions flag |
| V | External trauma conditions flag |

---

## 4. Academic Purpose & ML Approach

This repository is a teaching scaffold for **Machine Learning Operations (MLOps)**. We transition from a fragile Jupyter Notebook into a testable software engineering architecture.

* **Separation of Concerns:** Every step (Loading, Cleaning, Splitting, Training) has a dedicated, single-purpose Python module.
* **Fail-Fast Security Gates:** `validate.py` blocks missing values and invalid data types before expensive compute begins.
* **Leakage Prevention:** We split data *before* fitting feature recipes.
* **Deployable Artifacts:** The orchestrator bundles preprocessing and the algorithm into a single `.joblib` file, preventing training-serving skew.

### Future Roadmap (Upcoming Sessions)
* Move `SETTINGS` into `config.yaml`.
* Replace `print()` statements with structured logging.
* Add MLflow for experiment tracking and model registry.
* Containerize and serve predictions via a FastAPI application.

---

## 5. Repository Structure

```text
.
├── LICENSE
├── README.md
├── config.yaml
├── data
│   ├── inference
│   ├── processed
│   │   └── clean.csv
│   └── raw
│       └── opiod_raw_data.csv
├── environment.yml
├── models
│   └── model.joblib
├── notebooks
│   ├── 00_opioid_analysis_vLegacy.ipynb
│   ├── 01_opioid_analysis_vExp.ipynb
│   └── exmpak.py
├── pytest.ini
├── reports
│   └── predictions.csv
├── src
│   ├── __init__.py
│   ├── clean_data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── infer.py
│   ├── load_data.py
│   ├── main.py
│   ├── train.py
│   ├── utils.py
│   └── validate.py
└── tests
    ├── test_clean_data.py
    ├── test_evaluate.py
    ├── test_features.py
    ├── test_infer.py
    ├── test_load_data.py
    ├── test_main.py
    ├── test_train.py
    └── test_validate.py
```
## 6. How to Run & Test

### Step 1: Environment Setup
Build and activate the Conda environment:
```
conda env create -f environment.yml
conda activate mlops-modulddfdf
```
### Step 2: Run the Test Suite
Ensure the codebase is mathematically sound and pipeline contracts are unbroken:
```
python -m pytest -q tests/
```
> (You should see 100% passing tests!)

### Step 3: Execute the Orchestrator
Run the end-to-end machine learning pipeline to clean data, train the model, and generate artifacts:
```
python -m src.main
```
## 7. Outputs Generated:
1. data/processed/clean.csv: The deterministically cleaned input data
2. models/model.joblib: The deployable pipeline artifact
3. reports/predictions.csv: The inference log containing predictions and probabilities

## 8. Academic Purpose

This repository is a teaching scaffold for Machine Learning Operations (MLOps)

**Learning outcomes**

- Translate a notebook workflow into a src/ layout with explicit module contracts
- Implement quality gates before training to prevent silent failure modes
- Enforce leakage prevention by design through split and fit boundaries
- Produce model and data artifacts for auditability and reproducibility
- Add tests that validate behaviour, not just that code runs

**Roadmap for later sessions**
- Move SETTINGS into config.yaml and add environment based secrets via .env
- Replace prints with standard library logging and structured logs
- Add experiment tracking, model registry, and continuous integration
- Containerise and serve predictions via an application programming interface, FastAPI