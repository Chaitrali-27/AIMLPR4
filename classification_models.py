"""
Practical Assignment 4: Classification Models
SVM, Decision Tree, KNN and Naive Bayes on the Breast Cancer Wisconsin dataset.

Metrics : Accuracy, Precision, Recall, F1-score, Log Loss (+ ROC-AUC, 5-fold CV)
Run     : python classification_models.py
Output  : console tables + PNG graphs + CSV files inside the ./outputs folder
"""

import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")  # saves figures to files; also works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------------
# 1. Load and explore the dataset
# ----------------------------------------------------------------------
def load_data():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target  # target: 0 = malignant, 1 = benign
    print("=" * 70)
    print("DATASET: Breast Cancer Wisconsin (Diagnostic)")
    print("=" * 70)
    print(f"Samples  : {X.shape[0]}")
    print(f"Features : {X.shape[1]}")
    print(f"Classes  : {dict(zip(data.target_names, np.bincount(y)))}")
    print(f"Missing values: {int(X.isnull().sum().sum())}")

    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x=y.map({0: "malignant", 1: "benign"}), palette="Set2")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_class_distribution.png", dpi=150)
    plt.close()
    return X, y, data


# ----------------------------------------------------------------------
# 2. Define models (pipelines so scaling is learned on training data only)
# ----------------------------------------------------------------------
def build_models():
    """Returns {name: (pipeline, param_grid)} for hyper-parameter tuning."""
    return {
        "SVM": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", SVC(probability=True, random_state=RANDOM_STATE))]),
            {"clf__C": [0.1, 1, 10, 100],
             "clf__kernel": ["linear", "rbf"],
             "clf__gamma": ["scale", 0.01, 0.1]},
        ),
        "Decision Tree": (
            Pipeline([("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            {"clf__max_depth": [2, 3, 4, 5, 7, None],
             "clf__min_samples_split": [2, 5, 10],
             "clf__criterion": ["gini", "entropy"]},
        ),
        "KNN": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 7, 9, 11, 15],
             "clf__weights": ["uniform", "distance"],
             "clf__metric": ["euclidean", "manhattan"]},
        ),
        "Naive Bayes": (
            Pipeline([("clf", GaussianNB())]),
            {"clf__var_smoothing": np.logspace(-11, -5, 7)},
        ),
    }


# ----------------------------------------------------------------------
# 3. Train, tune and evaluate
# ----------------------------------------------------------------------
def evaluate(name, model, X_train, X_test, y_train, y_test, cv):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Log Loss": log_loss(y_test, np.clip(y_prob, 1e-15, 1 - 1e-15)),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "CV Accuracy (mean)": cv_scores.mean(),
        "CV Std": cv_scores.std(),
        "Train Time (s)": train_time,
    }, y_pred, y_prob


def main():
    X, y, data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = build_models()
    results, preds, probs, best_models, best_params, base_results = [], {}, {}, {}, {}, []

    # ---- Baseline (default hyper-parameters) -------------------------
    print("\n" + "=" * 70)
    print("STEP 1: BASELINE MODELS (default hyper-parameters)")
    print("=" * 70)
    for name, (pipe, _) in models.items():
        from sklearn.base import clone
        row, _, _ = evaluate(name, clone(pipe), X_train, X_test, y_train, y_test, cv)
        base_results.append(row)
    base_df = pd.DataFrame(base_results).set_index("Model").round(4)
    print(base_df[["Accuracy", "Precision", "Recall", "F1-Score", "Log Loss"]])
    base_df.to_csv(f"{OUT_DIR}/baseline_results.csv")

    # ---- Tuned models via GridSearchCV -------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: HYPER-PARAMETER TUNING (GridSearchCV, 5-fold, scoring=F1)")
    print("=" * 70)
    for name, (pipe, grid) in models.items():
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1)
        gs.fit(X_train, y_train)
        best_models[name] = gs.best_estimator_
        best_params[name] = gs.best_params_
        print(f"{name:14s} best params: {gs.best_params_}  (CV F1 = {gs.best_score_:.4f})")

        row, y_pred, y_prob = evaluate(name, gs.best_estimator_, X_train, X_test,
                                       y_train, y_test, cv)
        results.append(row)
        preds[name], probs[name] = y_pred, y_prob

    res_df = pd.DataFrame(results).set_index("Model").round(4)
    res_df.to_csv(f"{OUT_DIR}/tuned_results.csv")

    print("\n" + "=" * 70)
    print("STEP 3: FINAL COMPARISON (tuned models on the test set)")
    print("=" * 70)
    print(res_df[["Accuracy", "Precision", "Recall", "F1-Score", "Log Loss", "ROC-AUC"]])
    print("\nCross-validation and timing:")
    print(res_df[["CV Accuracy (mean)", "CV Std", "Train Time (s)"]])

    # ---- Detailed reports -------------------------------------------
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORTS")
    print("=" * 70)
    for name in models:
        print(f"\n--- {name} ---")
        print(classification_report(y_test, preds[name],
                                    target_names=data.target_names, digits=4))
        tn, fp, fn, tp = confusion_matrix(y_test, preds[name]).ravel()
        print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # ---- Graphs ------------------------------------------------------
    # 1) Metric comparison bar chart
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    ax = res_df[metrics].plot(kind="bar", figsize=(9, 5), width=0.8, colormap="Set2")
    ax.set_ylim(0.85, 1.04)
    ax.set_title("Model Comparison - Accuracy, Precision, Recall, F1 (Test Set)")
    ax.set_ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc="upper center", ncol=4)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_metric_comparison.png", dpi=150)
    plt.close()

    # 2) Log loss comparison
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=res_df.index, y=res_df["Log Loss"], palette="Set2")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.title("Log Loss (lower is better)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_log_loss.png", dpi=150)
    plt.close()

    # 3) Confusion matrices
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, name in zip(axes, models):
        ConfusionMatrixDisplay.from_predictions(
            y_test, preds[name], display_labels=data.target_names,
            cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_confusion_matrices.png", dpi=150)
    plt.close()

    # 4) ROC curves
    plt.figure(figsize=(6.5, 5.5))
    for name in models:
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res_df.loc[name, 'ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_roc_curves.png", dpi=150)
    plt.close()

    # 5) Cross-validation accuracy
    plt.figure(figsize=(6, 4))
    plt.bar(res_df.index, res_df["CV Accuracy (mean)"], yerr=res_df["CV Std"],
            capsize=5, color=sns.color_palette("Set2", 4))
    plt.ylim(0.88, 1.0)
    plt.title("5-Fold Cross-Validation Accuracy (mean +/- std)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_cv_accuracy.png", dpi=150)
    plt.close()

    # 6) Decision tree visualisation
    dt = best_models["Decision Tree"].named_steps["clf"]
    plt.figure(figsize=(16, 8))
    plot_tree(dt, feature_names=data.feature_names, class_names=data.target_names,
              filled=True, rounded=True, fontsize=8, max_depth=3)
    plt.title("Decision Tree (top 3 levels shown)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/07_decision_tree.png", dpi=130)
    plt.close()

    # 7) Decision tree feature importance
    imp = pd.Series(dt.feature_importances_, index=data.feature_names).nlargest(10)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=imp.values, y=imp.index, palette="viridis")
    plt.title("Decision Tree - Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/08_feature_importance.png", dpi=150)
    plt.close()

    # 8) KNN: accuracy vs K
    ks = range(1, 31)
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    tr_acc, te_acc = [], []
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k).fit(Xtr, y_train)
        tr_acc.append(knn.score(Xtr, y_train))
        te_acc.append(knn.score(Xte, y_test))
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, tr_acc, "o-", label="Train accuracy", markersize=4)
    plt.plot(ks, te_acc, "s-", label="Test accuracy", markersize=4)
    plt.xlabel("K (number of neighbours)")
    plt.ylabel("Accuracy")
    plt.title("KNN: Effect of K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/09_knn_k_vs_accuracy.png", dpi=150)
    plt.close()

    # 9) Effect of feature scaling (why SVM/KNN need it)
    print("\n" + "=" * 70)
    print("EXTRA: EFFECT OF FEATURE SCALING (test accuracy)")
    print("=" * 70)
    scale_rows = []
    for label, clf in [("SVM", SVC(kernel="rbf")), ("KNN", KNeighborsClassifier(5))]:
        raw = clf.fit(X_train, y_train).score(X_test, y_test)
        scaled = clf.fit(Xtr, y_train).score(Xte, y_test)
        scale_rows.append({"Model": label, "Without scaling": raw, "With scaling": scaled})
    scale_df = pd.DataFrame(scale_rows).set_index("Model").round(4)
    print(scale_df)
    scale_df.to_csv(f"{OUT_DIR}/scaling_effect.csv")

    # ---- Summary -----------------------------------------------------
    best = res_df["F1-Score"].idxmax()
    print("\n" + "=" * 70)
    print(f"BEST MODEL BY F1-SCORE : {best}  (F1 = {res_df.loc[best, 'F1-Score']:.4f})")
    print(f"LOWEST LOG LOSS        : {res_df['Log Loss'].idxmin()}")
    print("=" * 70)
    print(f"\nAll graphs and CSV files saved in the '{OUT_DIR}' folder.")


if __name__ == "__main__":
    main()
