import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from model_custom import CustomLogisticRegression

def evaluate_and_diagnose():
    # 1. Load Data
    data_path = '../data/insurance_fraud_data_cleaned.csv'
    if not os.path.exists(data_path):
        data_path = '../data/insurance_fraud_data.csv'
    
    df = pd.read_csv(data_path)
    
    # Preprocessing
    df = df.drop(['claim_number', 'claim_date', 'zip_code'], axis=1)
    if df['fraud reported'].dtype == 'O':
        df['fraud reported'] = df['fraud reported'].map({'Y': 1, 'N': 0})
    
    categorical_cols = ['gender', 'property_status', 'claim_day_of_week', 'accident_site', 'channel', 'vehicle_category', 'vehicle_color']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    X = df.drop('fraud reported', axis=1).values
    y = df['fraud reported'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 2. Load Models
    results = {}
    
    # Random Forest Evaluation
    try:
        rf_artifacts = joblib.load('fraud_model_artifacts.pkl')
        rf_model = rf_artifacts['model']
        
        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)
        
        results['Random Forest'] = {
            'train_acc': accuracy_score(y_train, y_train_pred),
            'test_acc': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'conf_matrix': confusion_matrix(y_test, y_test_pred)
        }
    except Exception as e:
        print(f"RF Error: {e}")

    # Custom Model Evaluation
    try:
        custom_artifacts = joblib.load('fraud_model_custom_artifacts.pkl')
        custom_model = custom_artifacts['model']
        
        y_train_pred_c = custom_model.predict(X_train)
        y_test_pred_c = custom_model.predict(X_test)
        
        results['Custom Logistic'] = {
            'train_acc': accuracy_score(y_train, y_train_pred_c),
            'test_acc': accuracy_score(y_test, y_test_pred_c),
            'precision': precision_score(y_test, y_test_pred_c, zero_division=0),
            'recall': recall_score(y_test, y_test_pred_c, zero_division=0),
            'f1': f1_score(y_test, y_test_pred_c, zero_division=0),
            'conf_matrix': confusion_matrix(y_test, y_test_pred_c)
        }
    except Exception as e:
        print(f"Custom Error: {e}")

    # 3. Print Report
    print("="*50)
    print("MODEL PERFORMANCE REPORT")
    print("="*50)
    
    for name, m in results.items():
        print(f"\n--- {name} ---")
        print(f"Training Accuracy: {m['train_acc']:.4f}")
        print(f"Testing Accuracy:  {m['test_acc']:.4f}")
        print(f"Precision:         {m['precision']:.4f}")
        print(f"Recall:            {m['recall']:.4f}")
        print(f"F1-Score:          {m['f1']:.4f}")
        
        # Check for Overfitting/Underfitting
        gap = m['train_acc'] - m['test_acc']
        if gap > 0.15: # Relaxed threshold for synthetic data
            print("DIAGNOSIS: Potential Overfitting")
        elif m['train_acc'] < 0.6:
            print("DIAGNOSIS: Potential Underfitting")
        else:
            print("DIAGNOSIS: Good fit")

    # 4. Generate Visuals
    plt.figure(figsize=(12, 5))
    for i, (name, m) in enumerate(results.items()):
        plt.subplot(1, 2, i+1)
        sns.heatmap(m['conf_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('model_evaluation_confusion_matrices.png')
    plt.close()
    print("\nEvaluation artifacts saved: model_evaluation_confusion_matrices.png")

if __name__ == "__main__":
    evaluate_and_diagnose()
