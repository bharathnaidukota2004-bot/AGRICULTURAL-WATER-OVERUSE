# train_and_save_models.py
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB

# Optional
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "Reducing_Agricultural_Water_Overuse_India_Dataset (4).csv"
OUT_DIR = BASE_DIR / "artifacts"
GRAPH_DIR = OUT_DIR / "graphs"
OUT_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
METRICS_XLSX = OUT_DIR / "model_metrics.xlsx"

# ---------------- LOAD ----------------
if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

# ---------------- ENSURE columns & compute Nash ----------------
# If columns like Crop_Water_Requirement(Liters) or Available_Water(Liters) missing, create dummy zeros
for col in ["Crop_Water_Requirement(Liters)", "Available_Water(Liters)", "Nash_Payoff"]:
    if col not in df.columns:
        df[col] = 0.0

# Compute Nash_Payoff if missing or all zeros (reasonable formula used throughout)
if df["Nash_Payoff"].isna().all() or (df["Nash_Payoff"] == 0).all():
    df["Nash_Payoff"] = (0.6 * df["Crop_Water_Requirement(Liters)"] / (df["Available_Water(Liters)"] + 1.0)) - 0.4 * df["Electricity_Usage(kWh)"]

# Clean and fill missing
df = df.fillna(method="ffill").fillna(0)

# ---------------- ENCODE & FEATURES ----------------
le_state = LabelEncoder()
le_crop = LabelEncoder()
le_target = LabelEncoder()

df["State_enc"] = le_state.fit_transform(df["State"].astype(str))
df["Crop_enc"]  = le_crop.fit_transform(df["Crop_Type"].astype(str))
df["Target_enc"] = le_target.fit_transform(df["Usage_Status"].astype(str))

FEATURES = [
    "State_enc", "Crop_enc", "Soil_Moisture(%)", "Temperature(°C)",
    "Humidity(%)", "Rainfall(mm)", "Groundwater_Level(m)",
    "Irrigation_Frequency(days/week)", "Fertilizer_Use(kg/acre)",
    "Electricity_Usage(kWh)", "Crop_Water_Requirement(Liters)",
    "Available_Water(Liters)", "Nash_Payoff"
]

# add any missing feature columns with zeros
for f in FEATURES:
    if f not in df.columns:
        df[f] = 0.0

X = df[FEATURES].astype(float)
y = df["Target_enc"]

# scale numeric features (keep scaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ---------------- MODELS ----------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RidgeClassifier": RidgeClassifier(),
    "SGDClassifier": SGDClassifier(max_iter=2000),
    "Perceptron": Perceptron(max_iter=2000),
    "PassiveAggressive": PassiveAggressiveClassifier(max_iter=2000),
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

if _HAS_XGB:
    models["XGBClassifier"] = XGBClassifier(eval_metric='logloss', n_estimators=200, random_state=42)
if _HAS_LGBM:
    models["LGBMClassifier"] = LGBMClassifier(n_estimators=200, random_state=42)

# ---------------- TRAIN & EVAL ----------------
rows = []
for i, (name, model) in enumerate(models.items(), start=1):
    print(f"[{i}/{len(models)}] Training {name}...")
    try:
        # calibrate LinearSVC
        if name == "LinearSVC":
            from sklearn.calibration import CalibratedClassifierCV
            model.fit(X_train, y_train)
            clf = CalibratedClassifierCV(model, cv="prefit")
            clf.fit(X_train, y_train)
        else:
            clf = model
            clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # get probabilities if available
        y_proba = None
        auc_val = "-"
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)
        elif hasattr(clf, "decision_function"):
            dec = clf.decision_function(X_test)
            if dec.ndim == 1:
                probs = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
                y_proba = np.vstack([1-probs, probs]).T
            else:
                y_proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)

        # ROC plot
        roc_path = ""
        if y_proba is not None:
            n_classes = len(np.unique(y_test))
            plt.figure(figsize=(6,4))
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:,1])
                auc_val = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}", color="orange")
            else:
                for j in range(n_classes):
                    fpr, tpr, _ = roc_curve(label_binarize(y_test, classes=np.unique(y_test))[:,j], y_proba[:,j])
                    plt.plot(fpr, tpr, label=f"Class {j}")
                auc_val = "-"
            plt.plot([0,1],[0,1],"--", color="gray")
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(loc="lower right")
            roc_path = GRAPH_DIR / f"{i:02d}_{name}_roc.png"
            plt.title(f"ROC - {name}")
            plt.savefig(roc_path); plt.close()
            roc_path = str(roc_path)

        # confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        cm_path = GRAPH_DIR / f"{i:02d}_{name}_cm.png"
        plt.figure(figsize=(4,3))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"CM - {name}")
        plt.colorbar()
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                plt.text(c, r, cm[r,c], ha="center", va="center")
        plt.tight_layout(); plt.savefig(cm_path); plt.close()
        cm_path = str(cm_path)

        rows.append({
            "Model": name, "Accuracy": round(acc*100,3), "Precision": round(prec*100,3),
            "Recall": round(rec*100,3), "F1": round(f1*100,3), "AUC": (round(auc_val,3) if isinstance(auc_val, float) else "-"),
            "ROC": roc_path, "CM": cm_path
        })

        # Save individual model to artifacts
        joblib.dump(clf, OUT_DIR / f"{name}.joblib")

    except Exception as e:
        print(f"Model {name} failed: {e}")
        rows.append({"Model": name, "Error": str(e)})

# save scaler and encoders
joblib.dump(scaler, OUT_DIR / "scaler.joblib")
joblib.dump(le_state, OUT_DIR / "le_state.joblib")
joblib.dump(le_crop, OUT_DIR / "le_crop.joblib")
joblib.dump(le_target, OUT_DIR / "le_target.joblib")
# save features list
joblib.dump(FEATURES, OUT_DIR / "features.joblib")

# write metrics excel
pd.DataFrame(rows).sort_values("Accuracy", ascending=False).to_excel(METRICS_XLSX, index=False)
print("Saved metrics to:", METRICS_XLSX)
print("Saved models & artifacts to:", OUT_DIR)
