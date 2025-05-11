import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    sklearn_cm = confusion_matrix(y_train, y_pred)
    print("Confusion Matrix:\n", sklearn_cm)


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


if __name__ == '__main__':
    main()
