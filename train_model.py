# ============================================================
# train_model.py
# Unified version — supports both:
#  1. Batch training & Excel summary (like your 18-model script)
#  2. Single-model run for Flask dashboard (like your web code)
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import cycle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "Reducing_Agricultural_Water_Overuse_India_Dataset (4).csv"
OUT_DIR = BASE_DIR / "outputs"
GRAPH_DIR = OUT_DIR / "graphs"
OUT_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_SUMMARY = OUT_DIR / "Model_Performance_Results.xlsx"

# ------------------------------------------------------------
# LOAD & PREPROCESS DATA
# ------------------------------------------------------------
if not Path(DATA_FILE).exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
print(f"✅ Dataset loaded successfully! Shape: {df.shape}")

# Encode categorical columns
le_state, le_crop, le_target = LabelEncoder(), LabelEncoder(), LabelEncoder()
df["State_enc"] = le_state.fit_transform(df["State"].astype(str))
df["Crop_enc"] = le_crop.fit_transform(df["Crop_Type"].astype(str))
df["Target_enc"] = le_target.fit_transform(df["Usage_Status"].astype(str))

FEATURES = [
    "State_enc",
    "Crop_enc",
    "Soil_Moisture(%)",
    "Temperature(°C)",
    "Humidity(%)",
    "Rainfall(mm)",
    "Groundwater_Level(m)",
    "Irrigation_Frequency(days/week)",
    "Fertilizer_Use(kg/acre)",
    "Electricity_Usage(kWh)",
    "Crop_Water_Requirement(Liters)",
    "Nash_Payoff"
]
X = df[FEATURES]
y = df["Target_enc"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# MODEL COLLECTION
# ------------------------------------------------------------
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RidgeClassifier": RidgeClassifier(),
    "SGDClassifier": SGDClassifier(max_iter=1000),
    "Perceptron": Perceptron(max_iter=1000),
    "PassiveAggressive": PassiveAggressiveClassifier(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "BaggingClassifier": BaggingClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
    "BernoulliNB": BernoulliNB(),
    "SVC": SVC(probability=True, kernel="rbf"),
    "LinearSVC": LinearSVC(max_iter=2000)
}

try:
    from xgboost import XGBClassifier
    models["XGBClassifier"] = XGBClassifier(eval_metric='logloss', n_estimators=200, random_state=42)
    print("✅ XGBoost Loaded")
except Exception:
    print("⚠️ XGBoost not installed")

try:
    from lightgbm import LGBMClassifier
    models["LGBMClassifier"] = LGBMClassifier(n_estimators=200, random_state=42)
    print("✅ LightGBM Loaded")
except Exception:
    print("⚠️ LightGBM not installed")

# ------------------------------------------------------------
# COMMON TRAINING FUNCTION
# ------------------------------------------------------------
def run_model(model_name):
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found!")

    model = models[model_name]
    print(f"\n🚀 Training: {model_name}")

    # Handle LinearSVC calibration
    if model_name == "LinearSVC":
        from sklearn.calibration import CalibratedClassifierCV
        model.fit(X_train, y_train)
        clf = CalibratedClassifierCV(model, cv="prefit")
        clf.fit(X_train, y_train)
    else:
        clf = model
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100

    # ROC/AUC
    auc_val, y_proba = None, None
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)
    elif hasattr(clf, "decision_function"):
        dec = clf.decision_function(X_test)
        if dec.ndim == 1:
            probs = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
            y_proba = np.vstack([1 - probs, probs]).T
        else:
            y_proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)

    roc_path = GRAPH_DIR / f"{model_name}_roc.png"
    cm_path = GRAPH_DIR / f"{model_name}_cm.png"

    if y_proba is not None:
        n_classes = len(np.unique(y_test))
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        plt.figure(figsize=(6, 4))
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, color="orange", lw=2, label=f"AUC={auc_val:.3f}")
        else:
            aucs = []
            for j, color in zip(range(n_classes), cycle(["orange", "blue", "green", "red", "purple"])):
                fpr, tpr, _ = roc_curve(y_test_bin[:, j], y_proba[:, j])
                aucs.append(auc(fpr, tpr))
                plt.plot(fpr, tpr, lw=2, label=f"Class {j} AUC={aucs[-1]:.3f}")
            auc_val = np.mean(aucs)
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    for a in range(cm.shape[0]):
        for b in range(cm.shape[1]):
            plt.text(b, a, cm[a, b], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    return {
        "Model": model_name,
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1": round(f1, 3),
        "AUC": round(auc_val * 100, 3) if auc_val else "-",
        "ROC": str(roc_path),
        "CM": str(cm_path)
    }

# ------------------------------------------------------------
# 1️⃣ RUN SINGLE MODEL (for Flask)
# ------------------------------------------------------------
def run_single_model(model_name):
    result = run_model(model_name)
    # Accuracy bar
    plt.figure(figsize=(4, 3))
    plt.bar([model_name], [result["Accuracy"]], color="skyblue")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy - {model_name}")
    acc_path = GRAPH_DIR / f"{model_name}_accuracy.png"
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()
    result["AccuracyGraph"] = str(acc_path)
    return result

# ------------------------------------------------------------
# 2️⃣ RUN ALL MODELS (for batch offline use)
# ------------------------------------------------------------
def run_all_models():
    results = []
    for i, name in enumerate(models.keys(), 1):
        try:
            r = run_model(name)
            results.append(r)
            print(f"✅ {name} done | Acc={r['Accuracy']:.2f}%")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append({"Model": name, "Error": str(e)})

    df_results = pd.DataFrame(results)
    df_results.to_excel(EXCEL_SUMMARY, index=False)
    print(f"\n📊 Results saved to {EXCEL_SUMMARY}")
    print(f"📈 Graphs saved to {GRAPH_DIR}")
    return df_results

# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    # Option 1: Single model (for quick test)
  #  result = run_single_model("RandomForest")
   # print("\nSingle Model Result:\n", result)

    # Option 2: Uncomment to train all
     run_all_models()
