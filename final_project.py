import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier as knn
import matplotlib.pyplot as plt

def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    sklearn_cm = confusion_matrix(y_train, y_pred)
    print("Confusion Matrix:\n", sklearn_cm)

def train_decision_tree(X_train, y_train):
    """
    Train a decision tree.
    """
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    sklearn_cm = confusion_matrix(y_train, y_pred)
    print("Confusion Matrix for Decision Trees:\n", sklearn_cm)


def train_k_nearest_neighbors(X_train, y_train, X_test, y_test):
    """
    Train a KNN model
    """
    K = []
    train = []
    test = []
    scores = {}
    for k in range(2, 25):
        model = knn(n_neighbors = k)
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        K.append(k)

        train.append(train_score)
        test.append(test_score)
        scores[k] = [train_score, test_score]

    plt.figure(figsize=(8, 4))
    plt.plot(K, train, marker='o', label='Training Accuracy', color='lightsteelblue')
    plt.plot(K, test, marker='o', label='Test Accuracy', color='pink')
    plt.xlabel('Values of k')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy: Training vs Test')
    plt.legend()
    plt.grid(True)
    plt.show()

    best_k = K[train.index(max(train))]
    print("Best k based on training accuracy:", best_k)

    model = knn(n_neighbors = best_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    sklearn_cm = confusion_matrix(y_train, y_pred)
    print("Confusion Matrix for KNN: \n", sklearn_cm)




def main():
    """
    Splitting the dataset into train and test sets ------------------------------------------
    """
    df = pd.read_csv("data/balanced_1to3_fraud_data.csv")

    # Split features and labels
    X = df.drop(columns=['Is Fraudulent'])
    y = df['Is Fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_csv("data/ecommerceX_train.csv", index=False)
    X_test.to_csv("data/ecommerceX_test.csv", index=False)
    y_train.to_csv("data/ecommerceY_train.csv", index=False)
    y_test.to_csv("data/ecommerceY_test.csv", index=False)

    """
    Training using Logistic Regression ------------------------------------------------------
    """

    train_logistic_regression(X_train, y_train)

    """
    Training using Decision Tree--------------------------------------------------------------
    """
    train_decision_tree(X_train, y_train)

    """
    Training using KNN------------------------------------------------------------------------
    """
    train_k_nearest_neighbors(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()
