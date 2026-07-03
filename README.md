# CreditGuard ML Pipeline
### Production-Grade Credit Card Fraud Detection & Transaction Anomaly Classification

> An end-to-end machine learning system engineered to detect fraudulent financial transactions at scale — built on the [ULB/Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), with leakage-free validation, imbalanced-learning safeguards, multi-model benchmarking, and automated enterprise reporting.

**Author:** Vedant Mandavale  
**Stack:** Python · scikit-learn · imbalanced-learn · Keras 3 · ReportLab · pandas · matplotlib · seaborn

---

## Executive Summary

This repository implements a complete fraud detection pipeline designed for real-world financial security constraints: extreme class imbalance, strict holdout integrity, reproducible cross-validation, and deployment-ready artifact generation. The system trains and evaluates four distinct model families, selects the optimal production candidate by **F1-score** (balancing precision and recall), and compiles a structured PDF analytical report alongside serialized models and diagnostic visualizations.

**Production recommendation:** **Random Forest** — F1 = **0.8276**, Precision = **91.1%**, Recall = **75.8%**, ROC-AUC = **0.9543**

---

## System Architecture — Engineering Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CREDITGUARD ML PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐
  │  creditcard.csv  │  284,807 transactions · 31 features · 0 missing values
  │  (Raw Ingestion) │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │       EDA        │  Shape · dtypes · class distribution · correlation heatmap
  │  & Visualization │  Amount/Time distributions · fraud skew analysis
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Data Cleaning   │  Drop 1,081 duplicate rows · cast Class to int
  │ & Deduplication  │  StandardScaler on Amount + Time (V1–V28 pre-PCA)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Train/Test Split │  80/20 · stratified · random_state=42
  │     FIRST        │  Train: 226,980 │ Test: 56,746 (imbalanced holdout)
  └────────┬─────────┘
           │
           ├──────────────────────────────────────┐
           │                                      │
           ▼                                      ▼
  ┌──────────────────┐                 ┌──────────────────┐
  │ SMOTE (Train     │                 │  Test Set Kept   │
  │  Data ONLY)      │                 │  Pure & Realistic│
  │  453,204 samples │                 │  No resampling   │
  └────────┬─────────┘                 └────────┬─────────┘
           │                                     │
           ▼                                     │
  ┌──────────────────┐                           │
  │ Feature Import.  │  Random Forest importance │  (held out until
  │  (Top-15 Plot)   │  V14 · V10 · V12 · V4…    | final evaluation)
  └────────┬─────────┘                           │
           │                                     │
           ▼                                     │
  ┌──────────────────┐                           │
  │ Stratified 5-Fold│  ImbPipeline: SMOTE inside│
  │ Cross-Validation │  each train fold only     │
  │ (LR + RF)        │  n_jobs=1 · fold progress │
  └────────┬─────────┘                           │
           │                                     │
           ▼                                     ▼
  ┌──────────────────────────────────────────────────────────┐
  │              Multi-Model Training & Evaluation           │
  │  ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐    |
  │  │ Logistic    │ │ Decision    │ │ Random Forest    │    |
  │  │ Regression  │ │ Tree        │ │ (100 estimators) │  │
  │  └─────────────┘ └─────────────┘ └──────────────────┘  │
  │  ┌──────────────────────────────────────────────────┐  │
  │  │ Keras Neural Network (Dense→Dropout→Sigmoid)     │  │
  │  │ Optimized on PR-AUC · Precision · Recall         │  │
  │  └──────────────────────────────────────────────────┘  │
  └──────────────────────────┬───────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────┐
  │           Artifact Generation & Report Compilation       │
  │  · Confusion matrices & ROC curves (per model)           │
  │  · Model comparison bar chart                            │
  │  · best_model.pkl · scaler.pkl · results.json            │
  │  · fraud_detection_report_Vedant_Mandavale.pdf(ReportLab)│
  └──────────────────────────────────────────────────────────┘
```

---

## Data Characteristics & Class Skew Breakdown

| Attribute | Value |
|-----------|-------|
| **Source** | Kaggle — ULB Machine Learning Group |
| **Total transactions** | 284,807 rows |
| **Feature count** | 31 (`Time`, `Amount`, `V1`–`V28`, `Class`) |
| **Legitimate (Class = 0)** | 284,315 (99.827%) |
| **Fraudulent (Class = 1)** | 492 (0.173%) |
| **Missing values** | 0 |
| **Duplicates removed** | 1,081 |

### Why Standard Accuracy Fails Here

A naive classifier that predicts **"not fraud" for every transaction** would achieve **~99.83% accuracy** while catching **zero fraud cases**. In production, that model is worthless — it would pass every fraudulent transaction undetected.

For imbalanced fraud detection, the pipeline prioritizes:

| Metric | Business Meaning |
|--------|------------------|
| **Recall** | What fraction of actual fraud is intercepted |
| **Precision** | What fraction of flagged transactions are truly fraud |
| **F1-Score** | Harmonic balance between precision and recall |
| **ROC-AUC** | Overall ranking quality across thresholds |
| **PR-AUC** | Ranking quality on the minority class (Neural Network) |

---

## Leakage-Free Preprocessing Methodology

### 1. Duplicate Removal (1,081 rows)
Duplicate transaction records inflate training signal and can cause the model to memorize repeated patterns rather than learn generalizable fraud indicators. Removing duplicates ensures each training observation represents a distinct event.

### 2. Train/Test Split Before Any Balancing
The dataset is split **80/20 with stratification** (`random_state=42`) **before** SMOTE is applied. This guarantees:

- The test set reflects the **true production fraud rate** (~0.17%)
- Synthetic fraud samples **never contaminate** evaluation data
- Reported metrics represent **real-world deployment performance**

### 3. SMOTE Applied to Training Data Only
SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic fraud examples exclusively on the **226,980-sample training split**, expanding it to **453,204 balanced samples**. The **56,746-sample test set remains untouched**.

### 4. Cross-Validation Without Synthetic Leakage
During 5-fold stratified cross-validation, SMOTE is wrapped inside an `ImbPipeline` and applied **only within each training fold**. Validation folds always contain **original, imbalanced data** — preventing the artificially inflated CV scores (~0.9999) that occur when SMOTE is applied globally before splitting folds.

### 5. Feature Scaling Scope
`StandardScaler` is fit on training data and applied to `Amount` and `Time`. Features `V1`–`V28` are already PCA-transformed and anonymized by the dataset provider.

---

## Validation & Resolution Engineering (Windows)

Training on Windows with nested parallel processing (`n_jobs=-1` inside `cross_val_score` combined with parallel Random Forest estimators) causes **CPU deadlocks and indefinite hangs** — a well-documented issue with joblib/loky on Windows.

### Mitigations Implemented

| Layer | Configuration | Rationale |
|-------|--------------|-----------|
| **Cross-validation loop** | Manual fold iteration with `n_jobs=1` | Sequential fold execution, visible `Fold 1/5…` progress |
| **CV Random Forest** | `n_jobs=1` inside `ImbPipeline` | No nested parallelism during validation |
| **CV classifiers** | `sklearn.base.clone()` per fold | Prevents state carry-over between folds |
| **BLAS threads** | `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` | Prevents thread contention at the linear algebra layer |
| **Final RF training** | `warm_start=True` with 10-tree progress batches | User-visible training progress on 453K SMOTE rows |
| **Pipeline checkpoint** | Saved after CV completes | Enables `finish_pipeline_Vedant_Mandavale.py` resume on interruption |

### Cross-Validation Results (Leakage-Free)

| Model | Mean ROC-AUC | Std Dev |
|-------|-------------|---------|
| Logistic Regression | **0.9807** | ± 0.0079 |
| Random Forest | **0.9685** | ± 0.0111 |

These scores are **realistic and trustworthy** — unlike pre-fix CV runs that reported ~0.9999 due to synthetic data leaking into validation folds.

---

## Model Portfolio Performance

Four model architectures were trained on SMOTE-balanced training data and evaluated on the **original imbalanced test set**.

### Test Set Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.9737 | 0.0531 | **0.8737** | 0.1002 | 0.9619 |
| Decision Tree | 0.9979 | 0.4250 | 0.7158 | 0.5333 | 0.8571 |
| **Random Forest** ★ | **0.9995** | **0.9114** | **0.7579** | **0.8276** | **0.9543** |
| Neural Network (Keras) | 0.9990 | 0.6579 | 0.7895 | 0.7177 | 0.9568 |

★ **Production selection criterion:** highest **F1-score** → Random Forest

### Architecture Matrix

#### Logistic Regression
- **Role:** Fast, interpretable baseline
- **Strength:** Highest recall (87.4%) — aggressive fraud capture
- **Weakness:** Precision of 5.3% — excessive false alarms in production
- **Use case:** High-recall screening tier when cost of missed fraud dominates

#### Decision Tree Classifier
- **Role:** Non-linear rule extraction
- **Strength:** Simple, inspectable decision boundaries
- **Weakness:** Overfitting tendency; moderate F1 (0.5333)
- **Use case:** Exploratory rule mining, not primary deployment

#### Random Forest Classifier (100 estimators)
- **Role:** **Production-recommended ensemble model**
- **Strength:** Best F1 balance — 91.1% precision with 75.8% recall
- **Architecture:** Bagged decision trees with feature subsampling
- **Use case:** Primary fraud scoring engine

#### Deep Learning — Keras Neural Network
- **Architecture:** `Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)`
- **Optimizer:** Adam · **Loss:** Binary Cross-Entropy
- **Training metrics:** **PR-AUC**, Precision, Recall (optimized for imbalanced classification)
- **Backend:** Keras 3 with PyTorch backend (Python 3.14 compatibility)
- **Use case:** Complex non-linear pattern capture; competitive ROC-AUC (0.9568)

### Top Predictive Features (Random Forest Importance)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | V14 | 0.189 |
| 2 | V10 | 0.118 |
| 3 | V12 | 0.102 |
| 4 | V17 | 0.097 |
| 5 | V4 | 0.093 |

> `V1`–`V28` are PCA-derived components; original transaction attributes are anonymized per dataset design.

---

## Production Artifacts Directory

### Core Pipeline Files

| File | Type | Description |
|------|------|-------------|
| `fraud_detection_Vedant_Mandavale.py` | Python | Main end-to-end pipeline orchestrator |
| `fraud_detection_Vedant_Mandavale.ipynb` | Jupyter Notebook | Interactive, documented walkthrough of all pipeline stages |
| `finish_pipeline_Vedant_Mandavale.py` | Python | Resume script — skips EDA/CV, trains models from checkpoint or cached results |
| `regenerate_outputs_Vedant_Mandavale.py` | Python | Rebuilds PDF report and comparison charts from existing `results_Vedant_Mandavale.json` |
| `requirements.txt` | Config | Pinned Python dependencies |

### Serialized Models

| File | Description |
|------|-------------|
| `models/best_model.pkl` | Production-ready **Random Forest** classifier (F1-selected) |
| `models/best_model_Vedant_Mandavale.pkl` | Named copy of the best model artifact |
| `models/best_model_meta_Vedant_Mandavale.json` | Model metadata — type, path, selection name |
| `models/scaler_Vedant_Mandavale.pkl` | Fitted `StandardScaler` for `Amount` and `Time` |
| `models/pipeline_checkpoint.pkl` | Mid-pipeline checkpoint (post-CV) for resume capability |

### Structured Logs & Metrics

| File | Description |
|------|-------------|
| `outputs/results_Vedant_Mandavale.json` | Complete run manifest — EDA stats, CV scores, test metrics, best model |
| `outputs/model_comparison_Vedant_Mandavale.csv` | Tabular model comparison (Accuracy, Precision, Recall, F1, ROC-AUC) |
| `outputs/pipeline_log.txt` | Full stdout capture from the most recent pipeline execution |

### Visualization Layouts (14 PNG Artifacts)

#### Layout 1 — Exploratory Data Analysis (`outputs/plots/eda_*.png`)

| File | Content |
|------|---------|
| `eda_class_distribution.png` | Fraud vs. legitimate transaction count — visualizes 0.173% skew |
| `eda_correlation_heatmap.png` | Feature correlation matrix across V1–V28, Amount, Time, Class |
| `eda_amount_time_distribution.png` | Histogram + KDE for Amount and Time feature distributions |
| `eda_amount_by_class.png` | Boxplot — Amount distribution segmented by fraud label |

#### Layout 2 — Model Evaluation Diagnostics (`outputs/plots/cm_*.png`, `roc_*.png`)

| File | Content |
|------|---------|
| `cm_logistic_regression.png` | Confusion matrix — Logistic Regression |
| `cm_decision_tree.png` | Confusion matrix — Decision Tree |
| `cm_random_forest.png` | Confusion matrix — Random Forest |
| `cm_neural_network.png` | Confusion matrix — Neural Network |
| `roc_logistic_regression.png` | ROC curve with AUC annotation |
| `roc_decision_tree.png` | ROC curve with AUC annotation |
| `roc_random_forest.png` | ROC curve with AUC annotation |
| `roc_neural_network.png` | ROC curve with AUC annotation |

#### Layout 3 — Comparative Analytics & Feature Engineering (`outputs/plots/feature_*.png`, `model_*.png`)

| File | Content |
|------|---------|
| `feature_importance_top15.png` | Horizontal bar chart — Random Forest top-15 feature importances |
| `model_comparison_chart.png` | Grouped bar chart — Precision, Recall, F1, ROC-AUC across all models |

### Final Report

| File | Description |
|------|-------------|
| `fraud_detection_report_Vedant_Mandavale.pdf` | **ReportLab-compiled** analytical report — problem statement, methodology, EDA findings, embedded visualizations, model comparison table, challenges, conclusion, and references |

---

## Quick Start

### Prerequisites

```bash
py -m pip install -r requirements.txt
```

> **Note:** On Python 3.14, Keras 3 uses the PyTorch backend (`KERAS_BACKEND=torch`). TensorFlow is not required.

### Run Full Pipeline

```bash
py fraud_detection_Vedant_Mandavale.py
```

### Resume After Interruption (skips EDA + CV)

```bash
py finish_pipeline_Vedant_Mandavale.py
```

### Regenerate PDF & Charts Only

```bash
py regenerate_outputs_Vedant_Mandavale.py
```

---

## Project Structure

```
Fraud Detection Analysis/
├── data/
│   └── creditcard.csv                  # Source dataset (not committed — download from Kaggle)
├── models/
│   ├── best_model.pkl                  # Production model
│   ├── scaler_Vedant_Mandavale.pkl                # Feature scaler
│   └── pipeline_checkpoint.pkl         # Resume checkpoint
├── outputs/
│   ├── plots/                          # 14 diagnostic PNG visualizations
│   ├── results_Vedant_Mandavale.json              # Run metrics manifest
│   └── model_comparison_Vedant_Mandavale.csv      # Model comparison table
├── fraud_detection_Vedant_Mandavale.py            # Main pipeline
├── fraud_detection_Vedant_Mandavale.ipynb         # Interactive notebook
├── finish_pipeline_Vedant_Mandavale.py            # Resume utility
├── fraud_detection_report_Vedant_Mandavale.pdf    # Compiled analytical report
├── requirements.txt
└── README.md
```

---

## Key Engineering Decisions

1. **F1 over ROC-AUC for model selection** — Logistic Regression achieves the highest ROC-AUC (0.9619) but only 5.3% precision, generating massive false alarm volume. Random Forest's F1 of 0.8276 reflects a deployable precision-recall tradeoff.

2. **SMOTE inside CV folds only** — Prevents synthetic sample leakage that inflates validation scores to near-perfect levels.

3. **Test set integrity** — All four models are evaluated on the original 56,746-sample imbalanced holdout, simulating production conditions.

4. **PR-AUC for neural network training** — Standard ROC-AUC is insensitive to class imbalance; Precision-Recall AUC provides a more honest training signal for the minority fraud class.

---

## References

- [Kaggle — Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [GeeksforGeeks — ML Credit Card Fraud Detection](https://www.geeksforgeeks.org/ml-credit-card-fraud-detection/)
- Dal Pozzolo, A. et al. (2015) — *Calibrating Probability with Undersampling*
- [scikit-learn — Imbalanced Learning Documentation](https://scikit-learn.org/)
- [imbalanced-learn — SMOTE](https://imbalanced-learn.org/)

---

*Built as a portfolio-grade fraud detection system demonstrating production ML engineering practices — leakage prevention, imbalanced classification, multi-model benchmarking, and automated reporting.*
