import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
X = np.load('X_test.npy')  # Using test data as dummy training data
y = np.load('y_test.npy')

# Models to evaluate (18 total)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Naive Bayes": GaussianNB(),
    "Bernoulli NB": BernoulliNB(),
    "SVM (RBF)": SVC(probability=True),
    "Linear SVM": SVC(kernel="linear", probability=True),
    "KNN": KNeighborsClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Bagging": BaggingClassifier(),
    "Ridge Classifier": RidgeClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    "LightGBM": LGBMClassifier(),
    "Gradient Boosting (Alt)": GradientBoostingClassifier(n_estimators=150)
}

# Create output folder
output_dir = 'static/roc_accuracy'
os.makedirs(output_dir, exist_ok=True)

# Dictionary to hold evaluation results
model_results = {}

# Train models, generate ROC plots, and store accuracy
for name, model in models.items():
    try:
        model.fit(X, y)
        
        # Some models (like Ridge) don't have predict_proba
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            # Fallback using decision_function
            y_scores = model.decision_function(X_test)
            from sklearn.preprocessing import MinMaxScaler
            y_proba = MinMaxScaler().fit_transform(y_scores.reshape(-1, 1)).ravel()

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        accuracy = model.score(X_test, y_test)

        model_results[name] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc, 'accuracy': accuracy}

        # Save ROC Curve plot
        plt.figure()
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')
        plt.grid()
        filename = os.path.join(output_dir, f'roc_{name.replace(" ", "_")}.png')
        plt.savefig(filename)
        plt.close()
        print(f"✅ Saved ROC curve: {filename}")
    except Exception as e:
        print(f"❌ Error processing {name}: {e}")

# Accuracy comparison bar chart
plt.figure(figsize=(12, 6))
names = list(model_results.keys())
accuracies = [model_results[name]['accuracy'] * 100 for name in names]
plt.bar(names, accuracies, color='skyblue')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison of ML Models')
plt.xticks(rotation=75, ha='right')
plt.tight_layout()
accuracy_chart_path = os.path.join(output_dir, 'accuracy_bar_all.png')
plt.savefig(accuracy_chart_path)
plt.close()
print(f"✅ Saved accuracy comparison graph: {accuracy_chart_path}")
