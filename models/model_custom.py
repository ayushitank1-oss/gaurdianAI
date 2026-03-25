import pandas as pd
import numpy as np
import joblib
import os

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for i in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if i % 100 == 0:
                cost = (-1 / n_samples) * np.sum(y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15))
                print(f"Iteration {i}: Cost {cost:.4f}")

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in probabilities]

def train_custom_model():
    # 1. Load Data (Using cleaned data if available, else original)
    data_path = 'insurance_fraud_data_cleaned.csv'
    if not os.path.exists(data_path):
        data_path = 'insurance_fraud_data.csv'
    
    df = pd.read_csv(data_path)
    print(f"Data loaded from {data_path}")

    # 2. Preprocessing (Consistent with model.py)
    df = df.drop(['claim_number', 'claim_date', 'zip_code'], axis=1)
    
    # Target variable encoding
    if df['fraud reported'].dtype == 'O':
        df['fraud reported'] = df['fraud reported'].map({'Y': 1, 'N': 0})
    
    # Categorical encoding
    categorical_cols = ['gender', 'property_status', 'claim_day_of_week', 'accident_site', 'channel', 'vehicle_category', 'vehicle_color']
    label_encoders = {}
    from sklearn.preprocessing import LabelEncoder, StandardScaler # Only for preprocessing as allowed
    
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    
    X = df.drop('fraud reported', axis=1).values
    y = df['fraud reported'].values

    # 3. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train/Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 5. Train Custom Model
    print("Starting custom training...")
    model = CustomLogisticRegression(learning_rate=0.1, iterations=1000)
    model.fit(X_train, y_train)

    # 6. Evaluation
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Custom Model Accuracy: {accuracy:.4f}")

    # 7. Save Artifacts (Compatible with existing apps)
    artifacts = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'columns': df.drop('fraud reported', axis=1).columns.tolist(),
        'categorical_cols': categorical_cols,
        'type': 'custom_logistic'
    }
    joblib.dump(artifacts, 'fraud_model_custom_artifacts.pkl')
    print("Custom model artifacts saved to fraud_model_custom_artifacts.pkl")

if __name__ == "__main__":
    train_custom_model()
