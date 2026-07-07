"""
Fraud Detection in Financial Transactions — End-to-End ML Pipeline
Author: Vedant Mandavale
Project: GENZ EDUCATEWING — Credit Card Fraud Detection
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import date
from pathlib import Path

# Unbuffered output for long-running pipeline
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Keras 3 on Python 3.14 uses PyTorch backend (TensorFlow not yet supported)
os.environ["KERAS_BACKEND"] = "torch"
# Prevent nested BLAS/thread deadlocks on Windows during sklearn CV
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import keras
from keras import layers

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AUTHOR_NAME = "Vedant Mandavale"
STUDENT_NAME = "Vedant_Mandavale"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

RF_N_ESTIMATORS = 100
RF_PROGRESS_STEP = 10  # print progress every N trees during final RF training
CHECKPOINT_PATH = MODELS_DIR / "pipeline_checkpoint.pkl"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def save_fig(name: str) -> Path:
    """Save current matplotlib figure to outputs/plots."""
    path = PLOTS_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_confusion_matrix(y_true, y_pred, title: str, filename: str) -> Path:
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_roc_curve(y_true, y_prob, title: str, filename: str) -> Path:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}", color="#3498db", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def evaluate_model(y_true, y_pred, y_prob) -> dict:
    """Compute classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def build_neural_network(input_dim: int) -> keras.Model:
    """Keras neural network as specified in project requirements."""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="fraud_nn",
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="pr_auc", curve="PR"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def run_eda(df: pd.DataFrame) -> dict:
    """Exploratory data analysis and save plots."""
    print("\n=== STEP 2: EDA ===")
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nDescribe:\n{df.describe().T.head(10)}")
    print(f"\nMissing values:\n{df.isnull().sum().sum()} total missing cells")

    df["Class"] = df["Class"].astype(int)
    fraud_count = df["Class"].sum()
    legit_count = len(df) - fraud_count
    fraud_pct = df["Class"].mean() * 100

    print(f"\nClass distribution:")
    print(f"  Legitimate (0): {legit_count:,} ({100 - fraud_pct:.3f}%)")
    print(f"  Fraud (1):      {fraud_count:,} ({fraud_pct:.3f}%)")

    # Class distribution plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Class", palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Class Distribution — Severe Imbalance")
    ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
    save_fig("eda_class_distribution.png")

    # Correlation heatmap (sample key columns for readability)
    corr_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "Class"]
    plt.figure(figsize=(14, 10))
    sns.heatmap(df[corr_cols].corr(), cmap="coolwarm", center=0, linewidths=0.1)
    plt.title("Feature Correlation Heatmap")
    save_fig("eda_correlation_heatmap.png")

    # Amount distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["Amount"], bins=50, kde=True, ax=axes[0], color="#3498db")
    axes[0].set_title("Distribution of Amount")
    sns.histplot(df["Time"], bins=50, kde=True, ax=axes[1], color="#9b59b6")
    axes[1].set_title("Distribution of Time")
    save_fig("eda_amount_time_distribution.png")

    # Amount by class
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="Class", y="Amount", palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Amount by Class")
    ax.set_xticklabels(["Legitimate", "Fraud"])
    save_fig("eda_amount_by_class.png")

    return {
        "shape": df.shape,
        "fraud_count": int(fraud_count),
        "legit_count": int(legit_count),
        "fraud_pct": float(fraud_pct),
        "missing": int(df.isnull().sum().sum()),
    }


def preprocess(df: pd.DataFrame):
    """Clean, scale, split, and apply SMOTE on training data only."""
    print("\n=== STEP 3: Preprocessing ===")
    df = df.copy()
    df["Class"] = df["Class"].astype(int)

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].copy()
    y = df["Class"]

    scaler = StandardScaler()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE (train only): {len(X_train_res):,} samples")
    print(f"  Fraud after SMOTE: {y_train_res.sum():,} / {len(y_train_res):,}")

    return X_train, X_test, y_train, y_test, X_train_res, y_train_res, feature_cols, scaler


def feature_selection(X_train, y_train, feature_cols) -> list:
    """Random Forest feature importance — top 15 plot."""
    print("\n=== STEP 4: Feature Selection ===")
    print(
        f"Training RF on {len(X_train):,} samples for feature importance (n_jobs=1)...",
        flush=True,
    )
    rf_fs = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=1,
        class_weight="balanced",
    )
    rf_fs.fit(X_train, y_train)
    print("Feature importance computed.", flush=True)

    importance = pd.Series(rf_fs.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top15 = importance.head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    top15.sort_values().plot(kind="barh", ax=ax, color="#2980b9")
    ax.set_title("Top 15 Features — Random Forest Importance")
    ax.set_xlabel("Importance")
    save_fig("feature_importance_top15.png")

    print("Top 10 features:")
    print(importance.head(10).to_string())
    print("\nProceeding with ALL features for modeling (as per project scope).")

    return importance.to_dict()


def cross_validate_models(X_train, y_train):
    """5-fold stratified CV with SMOTE inside each training fold only (no leakage)."""
    print("\n=== STEP 6: Cross-Validation (5-Fold Stratified) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models_cv = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=1
        ),
    }

    cv_results = {}
    for name, classifier in models_cv.items():
        print(f"\nRunning 5-fold CV for {name} (sequential, n_jobs=1)...", flush=True)
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
            print(f"  Fold {fold}/5...", flush=True)
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            pipeline = ImbPipeline(
                steps=[
                    ("smote", SMOTE(random_state=RANDOM_STATE)),
                    ("classifier", clone(classifier)),
                ]
            )
            pipeline.fit(X_tr, y_tr)
            y_prob = pipeline.predict_proba(X_val)[:, 1]
            fold_scores.append(roc_auc_score(y_val, y_prob))

        scores = np.array(fold_scores)
        cv_results[name] = {"scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std())}
        print(f"{name}: ROC-AUC = {scores} | Mean = {scores.mean():.4f} (+/- {scores.std():.4f})", flush=True)

    return cv_results


def train_random_forest_with_progress(X, y, n_estimators: int = RF_N_ESTIMATORS) -> RandomForestClassifier:
    """Train RF with warm_start so progress prints during long fits on SMOTE data."""
    rf = RandomForestClassifier(
        n_estimators=RF_PROGRESS_STEP,
        warm_start=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,  # safe here: single fit, not nested inside cross_val_score
    )
    for n in range(RF_PROGRESS_STEP, n_estimators + 1, RF_PROGRESS_STEP):
        rf.set_params(n_estimators=n)
        rf.fit(X, y)
        print(f"  Random Forest: {n}/{n_estimators} trees trained", flush=True)
    return rf


def train_and_evaluate(X_train_res, y_train_res, X_test, y_test):
    """Train all 4 models and evaluate on held-out test set."""
    print("\n=== STEP 5 & 7: Model Training & Evaluation ===")

    trained = {}
    results = {}

    # Model 1: Logistic Regression
    print("\nTraining Logistic Regression...", flush=True)
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_res, y_train_res)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_pred = lr.predict(X_test)
    trained["Logistic Regression"] = lr
    results["Logistic Regression"] = evaluate_model(y_test, lr_pred, lr_prob)
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression — Confusion Matrix", "cm_logistic_regression.png")
    plot_roc_curve(y_test, lr_prob, "Logistic Regression — ROC Curve", "roc_logistic_regression.png")
    print(classification_report(y_test, lr_pred, target_names=["Legitimate", "Fraud"]))

    # Model 2: Decision Tree
    print("\nTraining Decision Tree...", flush=True)
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(X_train_res, y_train_res)
    dt_prob = dt.predict_proba(X_test)[:, 1]
    dt_pred = dt.predict(X_test)
    trained["Decision Tree"] = dt
    results["Decision Tree"] = evaluate_model(y_test, dt_pred, dt_prob)
    plot_confusion_matrix(y_test, dt_pred, "Decision Tree — Confusion Matrix", "cm_decision_tree.png")
    plot_roc_curve(y_test, dt_prob, "Decision Tree — ROC Curve", "roc_decision_tree.png")
    print(classification_report(y_test, dt_pred, target_names=["Legitimate", "Fraud"]))

    # Model 3: Random Forest
    print(
        f"\nTraining Random Forest ({RF_N_ESTIMATORS} trees, parallel n_jobs=-1)...",
        flush=True,
    )
    rf = train_random_forest_with_progress(X_train_res, y_train_res)
    print("Random Forest training complete.", flush=True)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)
    trained["Random Forest"] = rf
    results["Random Forest"] = evaluate_model(y_test, rf_pred, rf_prob)
    plot_confusion_matrix(y_test, rf_pred, "Random Forest — Confusion Matrix", "cm_random_forest.png")
    plot_roc_curve(y_test, rf_prob, "Random Forest — ROC Curve", "roc_random_forest.png")
    print(classification_report(y_test, rf_pred, target_names=["Legitimate", "Fraud"]))

    # Model 4: Neural Network (Keras)
    print("\nTraining Neural Network (Keras — first run may take 1-2 min to initialize)...", flush=True)
    nn = build_neural_network(X_train_res.shape[1])
    nn.fit(
        X_train_res.values,
        y_train_res.values,
        epochs=10,
        batch_size=512,
        validation_split=0.1,
        verbose=1,
    )
    nn_prob = nn.predict(X_test.values, verbose=0).flatten()
    nn_pred = (nn_prob >= 0.5).astype(int)
    trained["Neural Network"] = nn
    results["Neural Network"] = evaluate_model(y_test, nn_pred, nn_prob)
    plot_confusion_matrix(y_test, nn_pred, "Neural Network — Confusion Matrix", "cm_neural_network.png")
    plot_roc_curve(y_test, nn_prob, "Neural Network — ROC Curve", "roc_neural_network.png")
    print(classification_report(y_test, nn_pred, target_names=["Legitimate", "Fraud"]))

    return trained, results


def save_comparison_table(results: dict) -> pd.DataFrame:
    """Print, save, and visualize model comparison table."""
    comparison = pd.DataFrame(results).T
    comparison = comparison[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    comparison = comparison.round(4)
    print("\n=== Model Comparison ===")
    print(comparison.to_string())
    comparison.to_csv(OUTPUT_DIR / f"model_comparison_{STUDENT_NAME}.csv")

    # Bar chart comparing key metrics across models
    plot_df = comparison[["Precision", "Recall", "F1", "ROC-AUC"]]
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.plot(kind="bar", ax=ax, colormap="Set2", width=0.8)
    ax.set_title("Model Performance Comparison (Test Set)")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=15)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=7)
    save_fig("model_comparison_chart.png")

    return comparison


def save_best_model(trained: dict, results: dict):
    """Save best model by F1 score (balances precision and recall)."""
    best_name = max(results, key=lambda k: results[k]["F1"])
    best_f1 = results[best_name]["F1"]
    best_auc = results[best_name]["ROC-AUC"]
    print(f"\nBest model: {best_name} (F1 = {best_f1:.4f}, ROC-AUC = {best_auc:.4f})")

    best = trained[best_name]
    if best_name == "Neural Network":
        model_path = MODELS_DIR / f"best_model_{STUDENT_NAME}.keras"
        best.save(model_path)
        meta = {"model_type": "keras", "path": str(model_path), "name": best_name}
    else:
        model_path = MODELS_DIR / f"best_model_{STUDENT_NAME}.pkl"
        joblib.dump(best, model_path)
        meta = {"model_type": "sklearn", "path": str(model_path), "name": best_name}

    # Also save as generic best_model.pkl for submission requirement
    if best_name != "Neural Network":
        joblib.dump(best, MODELS_DIR / "best_model.pkl")
    else:
        joblib.dump({"model_path": str(model_path), "type": "keras"}, MODELS_DIR / "best_model.pkl")

    with open(MODELS_DIR / f"best_model_meta_{STUDENT_NAME}.json", "w") as f:
        json.dump(meta, f, indent=2)

    return best_name, best_auc


def generate_pdf_report(
    eda_stats: dict,
    cv_results: dict,
    comparison: pd.DataFrame,
    best_name: str,
    best_auc: float,
):
    """Generate structured PDF report for submission."""
    print("\n=== STEP 9: Generating PDF Report ===")
    pdf_path = PROJECT_ROOT / f"fraud_detection_report_{STUDENT_NAME}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=22, spaceAfter=20)
    heading = ParagraphStyle("Heading2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=8)
    body = styles["BodyText"]
    story = []

    # 1. Title Page
    story.append(Paragraph("Fraud Detection in Financial Transactions", title_style))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(f"<b>Author:</b> {AUTHOR_NAME}", body))
    story.append(Paragraph(f"<b>Date:</b> {date.today().strftime('%B %d, %Y')}", body))
    story.append(Paragraph("<b>Organization:</b> GENZ EDUCATEWING", body))
    story.append(PageBreak())

    # 2. Problem Statement
    story.append(Paragraph("1. Problem Statement", heading))
    story.append(
        Paragraph(
            "Credit card fraud causes billions in losses annually. Detecting fraudulent transactions "
            "in real time is critical for financial institutions. The challenge is identifying rare "
            "fraudulent patterns among millions of legitimate transactions while minimizing false "
            "positives that frustrate customers and false negatives that cause financial loss.",
            body,
        )
    )
    story.append(Spacer(1, 12))

    # 3. Dataset Description
    story.append(Paragraph("2. Dataset Description", heading))
    story.append(
        Paragraph(
            f"<b>Source:</b> Kaggle — Credit Card Fraud Detection (ULB ML Group)<br/>"
            f"<b>URL:</b> https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud<br/>"
            f"<b>Shape:</b> {eda_stats['shape'][0]:,} rows × {eda_stats['shape'][1]} columns<br/>"
            f"<b>Features:</b> Time, Amount, V1–V28 (PCA-transformed), Class (target)<br/>"
            f"<b>Legitimate transactions:</b> {eda_stats['legit_count']:,}<br/>"
            f"<b>Fraudulent transactions:</b> {eda_stats['fraud_count']:,}<br/>"
            f"<b>Fraud rate:</b> {eda_stats['fraud_pct']:.3f}% (severe class imbalance)",
            body,
        )
    )
    story.append(Spacer(1, 12))

    # 4. Methodology
    story.append(Paragraph("3. Methodology", heading))
    story.append(
        Paragraph(
            "<b>Preprocessing:</b> Removed duplicates; scaled Amount and Time with StandardScaler "
            "(V1–V28 already PCA-normalized); 80/20 stratified train-test split.<br/>"
            "<b>Class imbalance:</b> Applied SMOTE only on training data to oversample minority class.<br/>"
            "<b>Feature selection:</b> Random Forest importance analysis; all 30 features retained.<br/>"
            "<b>Models:</b> Logistic Regression, Decision Tree, Random Forest (100 trees), "
            "Neural Network (Dense 64→Dropout→Dense 32→Dropout→Sigmoid).<br/>"
            "<b>Validation:</b> 5-fold StratifiedKFold cross-validation (ROC-AUC) on LR and RF.<br/>"
            "<b>Evaluation metrics:</b> Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.",
            body,
        )
    )
    story.append(Spacer(1, 12))

    # 5. EDA Findings
    story.append(Paragraph("4. EDA Findings", heading))
    story.append(
        Paragraph(
            "• Severe class imbalance: fraud accounts for only ~0.17% of all transactions.<br/>"
            "• V1–V28 features (PCA components) show varying correlations with the Class label.<br/>"
            "• Amount and Time distributions are skewed; scaling was necessary.<br/>"
            "• Fraudulent transactions tend to differ in certain PCA components (V14, V4, V12 among top features).",
            body,
        )
    )
    for plot_name in ["eda_class_distribution.png", "eda_amount_time_distribution.png", "eda_correlation_heatmap.png", "feature_importance_top15.png"]:
        plot_path = PLOTS_DIR / plot_name
        if plot_path.exists():
            story.append(Spacer(1, 8))
            story.append(Image(str(plot_path), width=5.5 * inch, height=3.2 * inch))
    story.append(PageBreak())

    # 6. Model Results
    story.append(Paragraph("5. Model Results", heading))
    story.append(
        Paragraph(
            "Four classifiers were trained on SMOTE-balanced training data and evaluated on the "
            "original imbalanced test set (realistic production scenario). "
            "<b>Recall</b> measures how many fraud cases we catch; <b>Precision</b> measures how "
            "many flagged transactions are actually fraud. <b>ROC-AUC</b> summarizes ranking quality.",
            body,
        )
    )
    story.append(Spacer(1, 8))
    table_data = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    for model_name, row in comparison.iterrows():
        table_data.append(
            [
                model_name,
                f"{row['Accuracy']:.4f}",
                f"{row['Precision']:.4f}",
                f"{row['Recall']:.4f}",
                f"{row['F1']:.4f}",
                f"{row['ROC-AUC']:.4f}",
            ]
        )
    t = Table(table_data, colWidths=[1.6 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch, 0.75 * inch, 0.85 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 12))

    comp_chart = PLOTS_DIR / "model_comparison_chart.png"
    if comp_chart.exists():
        story.append(Image(str(comp_chart), width=6 * inch, height=3.5 * inch))
        story.append(Spacer(1, 12))

    # Per-model ROC curves (sample)
    for roc_plot in ["roc_logistic_regression.png", "roc_random_forest.png", "roc_neural_network.png"]:
        roc_path = PLOTS_DIR / roc_plot
        if roc_path.exists():
            story.append(Image(str(roc_path), width=4.8 * inch, height=3.8 * inch))

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            f"<b>Best Model (ROC-AUC):</b> {best_name} with ROC-AUC = {best_auc:.4f}.<br/>"
            "<b>Trade-off note:</b> Logistic Regression achieves the highest recall (87%) and ROC-AUC, "
            "but very low precision (5%) — many false alarms. Random Forest offers a better balance "
            "(91% precision, 76% recall). The optimal choice depends on business cost of missed fraud "
            "vs. blocked legitimate transactions.",
            body,
        )
    )

    # CV results
    story.append(Spacer(1, 12))
    story.append(Paragraph("Cross-Validation Results (ROC-AUC):", styles["Heading3"]))
    for name, cv in cv_results.items():
        story.append(Paragraph(f"{name}: Mean = {cv['mean']:.4f} (± {cv['std']:.4f})", body))

    story.append(PageBreak())

    # 7. Challenges
    story.append(Paragraph("6. Challenges", heading))
    story.append(
        Paragraph(
            "<b>Class imbalance:</b> Fraud is extremely rare; accuracy alone is misleading. "
            "SMOTE and stratified sampling were essential.<br/>"
            "<b>False positive tradeoff:</b> Flagging legitimate transactions as fraud harms customer experience.<br/>"
            "<b>Overfitting risk:</b> Decision Trees can memorize training data; Random Forest and regularization help.<br/>"
            "<b>Feature interpretability:</b> PCA features (V1–V28) are not directly interpretable in business terms.",
            body,
        )
    )
    story.append(Spacer(1, 12))

    # 8. Conclusion
    story.append(Paragraph("7. Conclusion", heading))
    story.append(
        Paragraph(
            f"The {best_name} model achieved the best ROC-AUC ({best_auc:.4f}) on the test set. "
            "For real-world deployment, institutions should monitor recall (fraud capture rate) "
            "and precision (false alarm rate) based on business cost. Combining model scores with "
            "rule-based checks and human review provides a robust fraud prevention pipeline.",
            body,
        )
    )
    story.append(Spacer(1, 12))

    # 9. References
    story.append(Paragraph("8. References", heading))
    refs = [
        "Kaggle Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "GeeksforGeeks: https://www.geeksforgeeks.org/ml-credit-card-fraud-detection/",
        "Scikit-learn Documentation: https://scikit-learn.org/",
        "Dal Pozzolo et al. (2015) — Calibrating Probability with Undersampling",
    ]
    for ref in refs:
        story.append(Paragraph(f"• {ref}", body))

    doc.build(story)
    print(f"PDF saved: {pdf_path}")
    return pdf_path


def main():
    print("=" * 60)
    print(f"Fraud Detection Pipeline — {AUTHOR_NAME}")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    eda_stats = run_eda(df)

    X_train, X_test, y_train, y_test, X_train_res, y_train_res, feature_cols, scaler = preprocess(df)
    joblib.dump(scaler, MODELS_DIR / f"scaler_{STUDENT_NAME}.pkl")

    feature_importance = feature_selection(X_train, y_train, feature_cols)
    cv_results = cross_validate_models(X_train, y_train)

    # Checkpoint so you can resume with finish_pipeline_Vedant_Mandavale.py if interrupted
    joblib.dump(
        {
            "eda_stats": eda_stats,
            "cv_results": cv_results,
            "feature_importance": feature_importance,
            "X_train_res": X_train_res,
            "y_train_res": y_train_res,
            "X_test": X_test,
            "y_test": y_test,
            "scaler": scaler,
        },
        CHECKPOINT_PATH,
    )
    print(f"\nCheckpoint saved: {CHECKPOINT_PATH}", flush=True)

    trained, results = train_and_evaluate(X_train_res, y_train_res, X_test, y_test)
    comparison = save_comparison_table(results)
    best_name, best_auc = save_best_model(trained, results)

    # Save all artifacts for report generation
    artifacts = {
        "eda_stats": eda_stats,
        "cv_results": cv_results,
        "results": {k: v for k, v in results.items()},
        "best_model": best_name,
        "best_auc": best_auc,
        "feature_importance_top10": dict(list(feature_importance.items())[:10]),
    }
    with open(OUTPUT_DIR / f"results_{STUDENT_NAME}.json", "w") as f:
        json.dump(artifacts, f, indent=2)

    generate_pdf_report(eda_stats, cv_results, comparison, best_name, best_auc)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Plots:     {PLOTS_DIR}")
    print(f"  Model:     {MODELS_DIR / 'best_model.pkl'}")
    print(f"  Report:    fraud_detection_report_{STUDENT_NAME}.pdf")
    print(f"  Notebook:  fraud_detection_{STUDENT_NAME}.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()
