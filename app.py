from flask import Flask, render_template, request, session, send_file, url_for, jsonify
import numpy as np
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# classifiers
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier,
    PassiveAggressiveClassifier, SGDClassifier, Perceptron
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Optional: xgboost if installed
try:
    from xgboost import XGBClassifier
    _HAS_XG = True
except Exception:
    _HAS_XG = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'mitigating-water-overuse-dev-key')

# ==========================================================
# CONFIGURATION
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "dataset", "Reducing_Agricultural_Water_Overuse_India_Dataset (4).csv")
STATIC_DIR = os.path.join(BASE_DIR, "static")
GRAPH_DIR = os.path.join(STATIC_DIR, 'roc_accuracy')
os.makedirs(GRAPH_DIR, exist_ok=True)

# ==========================================================
# MODEL LIST
# ==========================================================
ML_MODELS = [
    'Logistic Regression', 'KNeighborsClassifier', 'DecisionTreeClassifier',
    'RandomForestClassifier', 'SVC', 'GaussianNB', 'LinearDiscriminantAnalysis',
    'QuadraticDiscriminantAnalysis', 'GradientBoostingClassifier', 'AdaBoostClassifier',
    'BaggingClassifier', 'ExtraTreesClassifier', 'RidgeClassifier', 'PassiveAggressiveClassifier',
    'SGDClassifier', 'Perceptron', 'BernoulliNB'
]
if _HAS_XG:
    ML_MODELS.append('XGBClassifier')

# ==========================================================
# LOAD EXCEL DATA
# ==========================================================
def load_data():
    # Load dataset (CSV or Excel)
    if EXCEL_PATH.endswith('.csv'):
        df = pd.read_csv(EXCEL_PATH)
    else:
        df = pd.read_excel(EXCEL_PATH)

    # Clean up column names
    df.columns = df.columns.str.strip()

    # Encode categorical features
    le_state = LabelEncoder()
    le_crop = LabelEncoder()
    le_target = LabelEncoder()

    if 'State' in df.columns:
        df["State"] = le_state.fit_transform(df["State"].astype(str))
    else:
        raise KeyError("Missing 'State' column in dataset")

    if 'Crop_Type' in df.columns:
        df["Crop_Type"] = le_crop.fit_transform(df["Crop_Type"].astype(str))
    else:
        raise KeyError("Missing 'Crop_Type' column in dataset")

    if 'Usage_Status' in df.columns:
        df["Usage_Status"] = le_target.fit_transform(df["Usage_Status"].astype(str))
    else:
        raise KeyError("Missing 'Usage_Status' column in dataset")

    # Automatically calculate Nash Payoff if not present
    if "Nash_Payoff" not in df.columns:
        if "Available_Water(Liters)" not in df.columns:
            df["Available_Water(Liters)"] = 50000.0
        df["Nash_Payoff"] = (
            0.6 * df["Crop_Water_Requirement(Liters)"] /
            (df["Available_Water(Liters)"] + 1)
        ) - 0.4 * df["Electricity_Usage(kWh)"]

    # Ensure all required columns exist, fill missing with 0
    required_cols = [
        "State", "Crop_Type", "Soil_Moisture(%)", "Rainfall(mm)",
        "Irrigation_Frequency(days/week)", "Groundwater_Level(m)",
        "Crop_Water_Requirement(Liters)", "Available_Water(Liters)",
        "Electricity_Usage(kWh)", "Temperature(°C)", "Nash_Payoff"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[required_cols]
    y = df["Usage_Status"]

    return train_test_split(X, y, test_size=0.2, random_state=42), le_state, le_crop, le_target

# ==========================================================
# MODEL CREATOR
# ==========================================================
def get_model_instance(name):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVC": SVC(probability=True, kernel='rbf'),
        "GaussianNB": GaussianNB(),
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "AdaBoostClassifier": AdaBoostClassifier(),
        "BaggingClassifier": BaggingClassifier(),
        "ExtraTreesClassifier": ExtraTreesClassifier(),
        "RidgeClassifier": RidgeClassifier(),
        "PassiveAggressiveClassifier": PassiveAggressiveClassifier(max_iter=1000),
        "SGDClassifier": SGDClassifier(max_iter=1000),
        "Perceptron": Perceptron(max_iter=1000),
        "BernoulliNB": BernoulliNB()
    }
    if _HAS_XG:
        models["XGBClassifier"] = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    return models[name]

# ==========================================================
# RUN MODEL AND GENERATE GRAPHS
# ==========================================================
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

def run_and_generate(model_name):
    (X_train, X_test, y_train, y_test), _, _, _ = load_data()
    model = get_model_instance(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Compute probabilities ---
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        dec = model.decision_function(X_test)
        if dec.ndim == 1:
            y_proba = np.vstack([1 - dec, dec]).T
        else:
            y_proba = (dec - dec.min()) / (dec.max() - dec.min())
    else:
        y_proba = np.zeros((len(y_pred), len(np.unique(y_test))))
        for i, c in enumerate(np.unique(y_test)):
            y_proba[:, i] = (y_pred == c).astype(float)

    acc = float(accuracy_score(y_test, y_pred))

    # --- Accuracy Plot ---
    acc_path = os.path.join(GRAPH_DIR, f"{model_name}_accuracy.png")
    plt.figure(figsize=(6, 3.5))
    plt.bar([model_name], [acc], color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy - {model_name}")
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    # --- ROC Curve ---
    roc_path = os.path.join(GRAPH_DIR, f"{model_name}_roc.png")
    try:
        n_classes = len(np.unique(y_test))
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        plt.figure(figsize=(6, 4))
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, color='orange', label=f"AUC = {roc_auc:.2f}")
        else:
            aucs = []
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")
            mean_auc = np.mean(aucs)
            plt.title(f"ROC (multi-class) - {model_name} | Avg AUC = {mean_auc:.2f}")
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
    except Exception as e:
        print(f"ROC Generation Skipped for {model_name}: {e}")
        roc_path = None

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(GRAPH_DIR, f"{model_name}_cm.png")
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    return {
        "accuracy": os.path.relpath(acc_path, start=STATIC_DIR).replace("\\", "/"),
        "roc": os.path.relpath(roc_path, start=STATIC_DIR).replace("\\", "/") if roc_path else None,
        "cm": os.path.relpath(cm_path, start=STATIC_DIR).replace("\\", "/"),
        "score": round(acc, 4)
    }

# ==========================================================
# ROUTES (all same as original)
# ==========================================================
@app.route('/')
def homepage():
    return render_template('homepage.html', models=ML_MODELS)

@app.route('/home')
def home():
    return render_template('home.html', models=ML_MODELS)

@app.route('/homepage.html')
def homepage1():
    return render_template('homepage.html', models=ML_MODELS)

@app.route('/form.html')
def form_page():
    return render_template('form.html', models=ML_MODELS)

@app.route('/contact.html')
def contact():
    return render_template('contact.html', models=ML_MODELS)

@app.route('/form', methods=['GET', 'POST'])
def form():
    return render_template('form.html', models=ML_MODELS)

@app.route('/thank_you', methods=['POST'])
def thank_you():
    form_dict = request.form.to_dict()
    if 'model' not in form_dict or not form_dict['model']:
        return "Error: Please select a model.", 400
    session['form_data'] = form_dict
    return render_template('thank_you.html')

@app.route('/predict')
def predict_page():
    try:
        form_data = session.get('form_data')
        if not form_data:
            return "No form data found. Please fill out the form first."

        model_name = form_data.get('model')
        if not model_name:
            return "No model selected for prediction.", 400

        # Load training data and encoders
        (X_train, X_test, y_train, y_test), le_state, le_crop, le_target = load_data()

        # Encode categorical inputs using the trained encoders
        state_input = form_data.get('state')
        crop_input = form_data.get('crop_type')

        try:
            state = le_state.transform([state_input])[0]
        except ValueError:
            state = 0

        try:
            crop_type = le_crop.transform([crop_input])[0]
        except ValueError:
            crop_type = 0

        # Numeric inputs
        soil = float(form_data.get('soil_moisture', 0))
        rainfall = float(form_data.get('rainfall', 0))
        irrig = float(form_data.get('irrigation', 0))
        gw = float(form_data.get('groundwater', 0))
        crop_req = float(form_data.get('crop_water_requirement', 0))
        avail = float(form_data.get('available_water', 0))
        elec = float(form_data.get('electricity', 0))
        temp = float(form_data.get('temperature', 0))
        nash = float(form_data.get('nash_payoff', 0))

        # Prepare input vector for prediction
        input_values = [state, crop_type, soil, rainfall, irrig, gw, crop_req, avail, elec, temp, nash]
        required_cols = [
            "State", "Crop_Type", "Soil_Moisture(%)", "Rainfall(mm)",
            "Irrigation_Frequency(days/week)", "Groundwater_Level(m)",
            "Crop_Water_Requirement(Liters)", "Available_Water(Liters)",
            "Electricity_Usage(kWh)", "Temperature(°C)", "Nash_Payoff"
        ]
        input_data = pd.DataFrame([input_values], columns=required_cols)

        # Train and test model
        model = get_model_instance(model_name)
        model.fit(X_train, y_train)
        prediction = model.predict(input_data)[0]

        # Accurately decode status using target label encoder classes
        try:
            status_label = le_target.inverse_transform([prediction])[0]
        except Exception:
            status_label = "Optimal" if prediction == 0 else "Overuse"

        if status_label == "Optimal":
            result = "Optimal Usage"
        elif status_label == "Overuse":
            result = "Overuse Detected"
        else:
            result = "Underuse Detected"

        y_pred_test = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_test)

        # Generate performance graphs
        results = run_and_generate(model_name)

        # Render result page with dynamic data
        return render_template(
            'result.html',
            name=form_data.get('name', 'User'),
            result=result,
            model_used=model_name,
            accuracy_score=round(accuracy * 100, 2),
            roc_image=f"/static/{results['roc']}" if results['roc'] else None,
            cm_image=f"/static/{results['cm']}",
            acc_image=f"/static/{results['accuracy']}",
            user_inputs=form_data
        )

    except Exception as e:
        return f"Error occurred: {e}"

@app.route('/models.html')
def models_page():
    return render_template('models.html', models=ML_MODELS)

@app.route('/about.html')
def about_html():
    return render_template('about.html', models=ML_MODELS)

@app.route("/run_model/<model_name>")
def run_model(model_name):
    try:
        results = run_and_generate(model_name)
        output = {
            "roc": f"/static/{results['roc']}" if results['roc'] else None,
            "accuracy_graph": f"/static/{results['accuracy']}",
            "confusion_matrix": f"/static/{results['cm']}",
            "accuracy_score": results['score']
        }
        return jsonify(output)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/compare_models.html')
@app.route('/compare_models')
def compare_models():
    return render_template('compare_models.html', models=ML_MODELS)

@app.route("/get_graphs/<model_name>")
def get_graphs(model_name):
    try:
        results = run_and_generate(model_name)
        return jsonify({
            "roc": f"/static/{results['roc']}" if results['roc'] else "",
            "accuracy": f"/static/{results['accuracy']}",
            "cm": f"/static/{results['cm']}",
            "score": results['score']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/graphs')
@app.route('/graphs.html')
def graphs():
    return render_template('graphs.html', graphs={})

@app.route('/predict.html')
def predict_loading():
    return render_template('predict.html')

# ==========================================================
# START APP
# ==========================================================
if __name__ == '__main__':
    app.run(debug=True)
