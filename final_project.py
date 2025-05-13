# Tina Nguyen nbn210002
# Catherine Le cnl210004

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier


def plot_roc_curve(fpr, tpr, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def display_confusion_matrix(cm, model_name, labels=None):
    # Display confusion matrix as a pop-up plot.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()


def evaluate_model(model, X_test, y_test, model_name):
    # Generate predictions and probabilities
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    # Standard accuracy and AUC
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Confusion matrix and ROC curve data
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # Compute true/false positive rates from confusion matrix
    TN, FP, FN, TP = cm.ravel()
    calculated_tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    calculated_fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

    # Find optimal threshold (maximizing TPR - FPR) and optimized accuracy
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    y_opt = (y_proba >= best_threshold).astype(int)
    opt_acc = accuracy_score(y_test, y_opt)

    # Print metrics
    print(f"=== {model_name} ===")
    print(f"Accuracy @ 0.5: {acc:.4f}")
    print(f"Optimized threshold: {best_threshold:.4f}")
    print(f"Accuracy @ optimal threshold: {opt_acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"True Positive Rate (TPR): {calculated_tpr:.4f}")
    print(f"False Positive Rate (FPR): {calculated_fpr:.4f}")

    # Display confusion matrix and ROC curve
    display_confusion_matrix(cm, model_name, labels=['Not Fraud','Is Fraud'])
    plot_roc_curve(fpr, tpr, model_name)

def train_logistic_regression(X_train, y_train, X_test, y_test):
    # Standard Logistic Regression with default parameters
    model = LogisticRegression()
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, "Logistic Regression")

def train_tuned_logistic(X_train, y_train, X_test, y_test):
    # Tuned Logistic Regression using GridSearchCV
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200]
    }

    model = GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("\nBest parameters found:", model.best_params_)
    evaluate_model(model.best_estimator_, X_test,
                   y_test, "Tuned Logistic Regression")

def train_decision_tree(X_train, y_train, X_test, y_test):
    # Decision Tree
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, "Decision Tree")

def train_tuned_decision_tree(X_train, y_train, X_test, y_test):
    # Tuned Decision Tree using GridSearchCV
    param_grid = {
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(),
        param_grid,
        cv = 5,
        scoring = 'roc_auc',
        n_jobs = -1,
        verbose = 1,
        refit = True
    )
    grid.fit(X_train, y_train)
    print("Best parameters found:", grid.best_params_)

    best_DT = grid.best_estimator_
    evaluate_model(best_DT, X_test, y_test, "Tuned DT")

def train_knn(X_train, y_train, X_test, y_test):
    # K-Nearest Neighbors
    model = knn(n_neighbors=5)
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, "KNN (k=5)")

def train_knn_tuned(X_train, y_train, X_test, y_test):
    # Tuned K-Nearest Neighbors using GridSearchCV
    param_grid = {
        'n_neighbors': list(range(2, 25)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # Manhattan vs Euclidean
    }
    grid = GridSearchCV(
        knn(),
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    grid.fit(X_train, y_train)
    print("Best parameters found:", grid.best_params_)

    best_knn = grid.best_estimator_
    evaluate_model(best_knn, X_test, y_test, "Tuned KNN")

def train_neural_network(X_train, y_train, X_test, y_test):
    # Neural Network
    model = MLPClassifier()
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, "Neural Network")

def train_tuned_neural_network(X_train, y_train, X_test, y_test):
    # Tuned Neural Network
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [1e-4, 1e-3, 1e-2],
        'learning_rate': ['constant', 'adaptive']
    }
    grid = GridSearchCV(
        MLPClassifier(
            max_iter=1000,
            solver='adam',
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        ),
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    grid.fit(X_train, y_train)
    print("Best parameters found:", grid.best_params_)

    best_nn = grid.best_estimator_
    evaluate_model(best_nn, X_test, y_test, "Tuned Neural Network")

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    # Gradient Boosting
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, "Gradient Boosting")

def train_gradient_boosting_tuned(X_train, y_train, X_test, y_test):
    # Tuned Gradient Boosting via GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'max_features': [None, 'sqrt', 'log2']
    }
    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    grid.fit(X_train, y_train)
    print("Best parameters found (GradBoost):", grid.best_params_)
    evaluate_model(grid.best_estimator_, X_test, y_test, "Tuned Gradient Boosting")

def main():
    # Load and split data
    df = pd.read_csv("data/balanced_1to1_fraud_data.csv")
    X = df.drop(columns=['Is Fraudulent'])
    y = df['Is Fraudulent']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models
    train_logistic_regression(X_train, y_train, X_test, y_test)
    train_tuned_logistic(X_train, y_train, X_test, y_test)

    train_knn(X_train, y_train, X_test, y_test)
    train_knn_tuned(X_train, y_train, X_test, y_test)

    train_decision_tree(X_train, y_train, X_test, y_test)
    train_tuned_decision_tree(X_train, y_train, X_test, y_test)

    train_neural_network(X_train, y_train, X_test, y_test)
    train_tuned_neural_network(X_train, y_train, X_test, y_test)

    train_gradient_boosting(X_train, y_train, X_test, y_test)    
    train_gradient_boosting_tuned(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()
