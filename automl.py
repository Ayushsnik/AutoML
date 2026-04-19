import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report
from sklearn.utils.multiclass import type_of_target

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


# --------------------------------------------------
# 1️⃣ Load Dataset
# --------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    return df


# --------------------------------------------------
# 2️⃣ Detect Problem Type
# --------------------------------------------------
def detect_problem(y):
    target_type = type_of_target(y)
    if target_type in ["binary", "multiclass"]:
        return "classification"
    else:
        return "regression"


# --------------------------------------------------
# 3️⃣ Preprocessing
# --------------------------------------------------
def preprocess_data(df, target_column):

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Fill missing numeric values
    X = X.fillna(X.mean(numeric_only=True))

    # Encode categorical features
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


# --------------------------------------------------
# 4️⃣ Train & Tune Models
# --------------------------------------------------
def train_models(X_train, X_test, y_train, y_test, problem_type):

    results = {}

    # ---------------- CLASSIFICATION ----------------
    if problem_type == "classification":

        models = {
            "Random Forest": (
                RandomForestClassifier(),
                {"n_estimators": [50, 100],
                 "max_depth": [None, 10, 20]}
            ),
            "SVM": (
                SVC(),
                {"C": [0.1, 1, 10],
                 "kernel": ["linear", "rbf"]}
            ),
            "Logistic Regression": (
                LogisticRegression(max_iter=1000),
                {"C": [0.1, 1, 10]}
            )
        }

        for name, (model, params) in models.items():
            print(f"\nTraining {name}...")

            grid = GridSearchCV(model, params, cv=3, scoring="accuracy")
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            preds = best_model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            results[name] = (acc, best_model)

            print(f"{name} Best Params: {grid.best_params_}")
            print(f"{name} Accuracy: {acc:.4f}")

        best_model_name = max(results, key=lambda x: results[x][0])
        best_score = results[best_model_name][0]
        best_model = results[best_model_name][1]

        # Confusion Matrix
        preds = best_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)

        plt.figure()
        sns.heatmap(cm, annot=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        print("\nClassification Report:")
        print(classification_report(y_test, preds))

        # Feature Importance (only for tree models)
        if hasattr(best_model, "feature_importances_"):
            plt.figure()
            plt.bar(range(len(best_model.feature_importances_)),
                    best_model.feature_importances_)
            plt.title("Feature Importance")
            plt.xlabel("Feature Index")
            plt.ylabel("Importance")
            plt.show()

        ylabel = "Accuracy"

    # ---------------- REGRESSION ----------------
    else:

        models = {
            "Random Forest Regressor": (
                RandomForestRegressor(),
                {"n_estimators": [50, 100],
                 "max_depth": [None, 10, 20]}
            ),
            "SVR": (
                SVR(),
                {"C": [0.1, 1, 10],
                 "kernel": ["linear", "rbf"]}
            ),
            "Linear Regression": (
                LinearRegression(),
                {}
            )
        }

        for name, (model, params) in models.items():
            print(f"\nTraining {name}...")

            if params:
                grid = GridSearchCV(model, params, cv=3, scoring="r2")
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                print(f"{name} Best Params: {grid.best_params_}")
            else:
                model.fit(X_train, y_train)
                best_model = model

            preds = best_model.predict(X_test)
            score = r2_score(y_test, preds)

            results[name] = (score, best_model)
            print(f"{name} R2 Score: {score:.4f}")

        best_model_name = max(results, key=lambda x: results[x][0])
        best_score = results[best_model_name][0]
        best_model = results[best_model_name][1]

        ylabel = "R2 Score"

    # Model Comparison Graph
    model_names = list(results.keys())
    scores = [results[name][0] for name in model_names]

    plt.figure()
    plt.bar(model_names, scores)
    plt.title("Model Comparison")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.show()

    print("\n==============================")
    print("Best Model:", best_model_name)
    print("Best Score:", round(best_score, 4))
    print("==============================")

    return best_model


# --------------------------------------------------
# 5️⃣ Save Model
# --------------------------------------------------
def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.pkl")
    print("Model Saved Successfully!")


# --------------------------------------------------
# 🚀 MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":

    path = "data/sample-data.csv"
    target_column = "label"

    df = load_data(path)

    problem_type = detect_problem(df[target_column])
    print("Detected Problem Type:", problem_type)

    X, y = preprocess_data(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_model = train_models(
        X_train, X_test, y_train, y_test, problem_type
    )

    save_model(best_model)
