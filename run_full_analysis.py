"""
Spambase Email Classification & Feature Selection Pipeline
Author: Machine Learning Engineer
Dataset: UCI Spambase Dataset (4,601 instances, 57 predictors, binary target)
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Visualization (Static & Interactive)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# Bayesian Optimization
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def load_dataset(base_dir):
    """
    Load Spambase dataset and parse attribute names from .names file.
    """
    names_path = os.path.join(base_dir, "spambase.names")
    data_path = os.path.join(base_dir, "spambase.data")

    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, "data", "spambase.data")
        names_path = os.path.join(base_dir, "data", "spambase.names")

    feature_names = []
    with open(names_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("|") or line.startswith("1, 0"):
                continue
            if ":" in line:
                feature_names.append(line.split(":")[0].strip())

    columns = feature_names + ["is_spam"]
    df = pd.read_csv(data_path, header=None, names=columns)
    return df


def generate_interactive_plots(df, rf_importance, optuna_trials_df, output_dir):
    """
    Generate and save interactive HTML plots using Plotly.
    """
    # 1. Interactive Class Distribution
    counts = df["is_spam"].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    counts["Label"] = counts["Class"].map({0: "Non-Spam (Ham)", 1: "Spam"})
    counts["Percentage"] = (counts["Count"] / len(df) * 100).round(2)

    fig_class = px.bar(
        counts,
        x="Label",
        y="Count",
        color="Label",
        color_discrete_map={"Non-Spam (Ham)": "#2b5c8f", "Spam": "#c0392b"},
        text="Percentage",
        title="Interactive Spambase Target Distribution",
        template="plotly_white"
    )
    fig_class.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_class.update_layout(showlegend=False, yaxis_title="Number of Emails", xaxis_title="")
    fig_class.write_html(os.path.join(output_dir, "interactive_class_distribution.html"))

    # 2. Interactive Feature Importance (Top 25)
    top_rf = rf_importance.head(25).reset_index()
    top_rf.columns = ["Feature", "Importance"]
    top_rf = top_rf.sort_values(by="Importance", ascending=True)

    fig_imp = px.bar(
        top_rf,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 25 Feature Importances (Random Forest Gini)",
        color="Importance",
        color_continuous_scale="Blues",
        template="plotly_white"
    )
    fig_imp.update_layout(height=700, yaxis_title="", xaxis_title="Mean Gini Decrease")
    fig_imp.write_html(os.path.join(output_dir, "interactive_feature_importance.html"))

    # 3. Interactive Optuna Hyperparameter Study (if available)
    if optuna_trials_df is not None and not optuna_trials_df.empty:
        fig_opt = px.scatter(
            optuna_trials_df,
            x="params_n_neighbors",
            y="value",
            color="params_weights",
            size="params_leaf_size",
            hover_data=["params_p"],
            title="Optuna Hyperparameter Search: k vs Cross-Validation F1-Score",
            labels={"params_n_neighbors": "Number of Neighbors (k)", "value": "5-Fold CV F1-Score", "params_weights": "Weight Function"},
            template="plotly_white"
        )
        fig_opt.write_html(os.path.join(output_dir, "interactive_optuna_tuning.html"))


def run():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("SPAMBASE KNN CLASSIFICATION & FEATURE SELECTION ENGINE")
    print("=" * 70)

    # 1. Ingestion
    print("\n[Step 1/6] Loading data...")
    df = load_dataset(base_dir)
    print(f"Dataset successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Preprocessing & Splitting
    print("\n[Step 2/6] Splitting and scaling features...")
    X = df.drop(columns=["is_spam"])
    y = df["is_spam"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 3. Baseline KNN (All 57 Features)
    print("\n[Step 3/6] Benchmarking Baseline KNN (57 Features)...")
    k_range = list(range(1, 26, 2))
    cv_scores_all = []
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
        score = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring="f1").mean()
        cv_scores_all.append(score)

    best_k_all = k_range[int(np.argmax(cv_scores_all))]
    knn_all = KNeighborsClassifier(n_neighbors=best_k_all, weights="distance")
    knn_all.fit(X_train_scaled, y_train)

    start_time = time.time()
    y_pred_all = knn_all.predict(X_test_scaled)
    latency_all = (time.time() - start_time) * 1000

    acc_all = accuracy_score(y_test, y_pred_all)
    prec_all = precision_score(y_test, y_pred_all)
    rec_all = recall_score(y_test, y_pred_all)
    f1_all = f1_score(y_test, y_pred_all)
    print(f"Baseline KNN (k={best_k_all}): Acc={acc_all:.4f}, F1={f1_all:.4f}, Latency={latency_all:.2f}ms")

    # 4. Feature Selection via Random Forest Gini Importance
    print("\n[Step 4/6] Ranking features with Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    top_20_features = rf_importance.head(20).index.tolist()
    top_20_indices = [X.columns.get_loc(col) for col in top_20_features]

    X_train_20 = X_train_scaled[:, top_20_indices]
    X_test_20 = X_test_scaled[:, top_20_indices]

    cv_scores_20 = []
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
        score = cross_val_score(clf, X_train_20, y_train, cv=5, scoring="f1").mean()
        cv_scores_20.append(score)

    best_k_20 = k_range[int(np.argmax(cv_scores_20))]
    knn_20 = KNeighborsClassifier(n_neighbors=best_k_20, weights="distance")
    knn_20.fit(X_train_20, y_train)

    start_time = time.time()
    y_pred_20 = knn_20.predict(X_test_20)
    latency_20 = (time.time() - start_time) * 1000

    acc_20 = accuracy_score(y_test, y_pred_20)
    prec_20 = precision_score(y_test, y_pred_20)
    rec_20 = recall_score(y_test, y_pred_20)
    f1_20 = f1_score(y_test, y_pred_20)
    print(f"Selected 20 Features (k={best_k_20}): Acc={acc_20:.4f}, F1={f1_20:.4f}, Latency={latency_20:.2f}ms")

    # 5. Hyperparameter Tuning with Optuna
    optuna_trials_df = None
    if OPTUNA_AVAILABLE:
        print("\n[Step 5/6] Tuning KNN Hyperparameters with Optuna (Bayesian Search)...")

        def objective(trial):
            n_neighbors = trial.suggest_int("n_neighbors", 3, 27, step=2)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            p = trial.suggest_int("p", 1, 2)
            leaf_size = trial.suggest_int("leaf_size", 10, 50, step=10)

            model = KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, p=p, leaf_size=leaf_size
            )
            return cross_val_score(model, X_train_20, y_train, cv=5, scoring="f1").mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        best_params = study.best_params
        optuna_trials_df = study.trials_dataframe()

        print(f"Optuna Best Parameters: {best_params}")
        knn_optuna = KNeighborsClassifier(**best_params)
        knn_optuna.fit(X_train_20, y_train)

        start_time = time.time()
        y_pred_opt = knn_optuna.predict(X_test_20)
        latency_opt = (time.time() - start_time) * 1000

        acc_opt = accuracy_score(y_test, y_pred_opt)
        prec_opt = precision_score(y_test, y_pred_opt)
        rec_opt = recall_score(y_test, y_pred_opt)
        f1_opt = f1_score(y_test, y_pred_opt)
    else:
        print("\n[Step 5/6] Optuna not detected. Skipping Bayesian tuning.")
        acc_opt, prec_opt, rec_opt, f1_opt, latency_opt = acc_20, prec_20, rec_20, f1_20, latency_20

    # 6. Interactive Visualizations & Summary Report
    print("\n[Step 6/6] Generating interactive Plotly charts & saving summary...")
    generate_interactive_plots(df, rf_importance, optuna_trials_df, output_dir)

    results = [
        {
            "Model Configuration": "Baseline KNN (All 57 Features)",
            "Features Count": 57,
            "Accuracy": round(acc_all * 100, 2),
            "Precision": round(prec_all * 100, 2),
            "Recall": round(rec_all * 100, 2),
            "F1-Score": round(f1_all * 100, 2),
            "Latency (ms)": round(latency_all, 2),
        },
        {
            "Model Configuration": "Feature Selection KNN (Top 20 Features)",
            "Features Count": 20,
            "Accuracy": round(acc_20 * 100, 2),
            "Precision": round(prec_20 * 100, 2),
            "Recall": round(rec_20 * 100, 2),
            "F1-Score": round(f1_20 * 100, 2),
            "Latency (ms)": round(latency_20, 2),
        },
    ]

    if OPTUNA_AVAILABLE:
        results.append({
            "Model Configuration": "Optuna Tuned KNN (Top 20 Features, p=1 Manhattan)",
            "Features Count": 20,
            "Accuracy": round(acc_opt * 100, 2),
            "Precision": round(prec_opt * 100, 2),
            "Recall": round(rec_opt * 100, 2),
            "F1-Score": round(f1_opt * 100, 2),
            "Latency (ms)": round(latency_opt, 2),
        })

    summary_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "model_comparison_summary.csv")
    summary_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY METRICS")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nInteractive charts and CSV output saved to: {output_dir}")


if __name__ == "__main__":
    run()
