import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("merged_dataset.csv")  # Replace with your actual filename

# Drop Transaction.Date column
df.drop(columns=['Transaction.Date'], inplace=True)

# Encode categorical columns
categorical_columns = ['source', 'browser', 'sex', 'Payment.Method',
                       'Product.Category', 'Device.Used']

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
