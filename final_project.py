import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------- Splitting Kaggle dataset into train and test sets -----------------------
df = pd.read_csv("merged_dataset.csv")

# we don't need the 'Transaction.Date' column for our analysis, so we removed it.
df.drop(columns=['Transaction.Date'], inplace=True)

categorical_columns = ['source', 'browser', 'sex', 'Payment.Method',
                       'Product.Category', 'Device.Used']

# Convert categorical columns to numerical using Label Encoding for easier feature analysis.
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Split features and labels
X = df.drop(columns=['Is.Fraudulent'])
y = df['Is.Fraudulent']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.to_csv("ecommerceX_train.csv", index=False)
X_test.to_csv("ecommerceX_test.csv", index=False)
y_train.to_csv("ecommerceY_train.csv", index=False)
y_test.to_csv("ecommerceY_test.csv", index=False)


# ----------------------- Training the Perceptron model -----------------------
def train_perceptron(X_train, y_train):
    # Initialize the Perceptron model
    perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)

    # Fit the model to the training data
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_train)
    sklearn_cm = confusion_matrix(y_train, y_pred)
    print("Confusion Matrix for perceptron:\n", sklearn_cm)


train_perceptron(X_train, y_train)
