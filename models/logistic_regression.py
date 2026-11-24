import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

def train_and_get_graphs(output_folder="static/graphs"):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load dataset
    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression ROC Curve")
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_folder, "logistic_regression_roc.png")
    plt.savefig(roc_path)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Logistic Regression Confusion Matrix")
    cm_path = os.path.join(output_folder, "logistic_regression_cm.png")
    plt.savefig(cm_path)
    plt.close()

    # Accuracy Graph
    plt.figure()
    plt.bar(["Logistic Regression"], [accuracy], color="green")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    acc_path = os.path.join(output_folder, "logistic_regression_accuracy.png")
    plt.savefig(acc_path)
    plt.close()

    return accuracy, roc_path, cm_path, acc_path
