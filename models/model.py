import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    # 1. Load Data
    df = pd.read_csv('insurance_fraud_data.csv')
    print("Data loaded successfully.")
    
    # 1.1 Outlier Handling (added from EDA)
    fin_cols = ['annual_income', 'total_claim', 'injury_claim', 'annual premium', 'vehicle_price']
    for col in fin_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    print("Outliers handled.")
    
    # 2. Preprocessing
    # Drop unique ID which doesn't contribute to prediction
    df = df.drop(['claim_number', 'claim_date', 'zip_code'], axis=1)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['gender', 'property_status', 'claim_day_of_week', 'accident_site', 'channel', 'vehicle_category', 'vehicle_color']
    
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    
    # Target variable encoding
    df['fraud reported'] = df['fraud reported'].map({'Y': 1, 'N': 0})
    
    # 3. Split features and target
    X = df.drop('fraud reported', axis=1)
    y = df['fraud reported']
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # 7. Evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 8. Save Model and Artifacts
    artifacts = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'columns': X.columns.tolist(),
        'categorical_cols': categorical_cols
    }
    joblib.dump(artifacts, 'fraud_model_artifacts.pkl')
    print("Model and artifacts saved to fraud_model_artifacts.pkl")

if __name__ == "__main__":
    train_model()
