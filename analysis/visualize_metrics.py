import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from model_custom import CustomLogisticRegression

def load_and_preprocess():
    data_path = '../data/insurance_fraud_data_cleaned.csv'
    if not os.path.exists(data_path):
        data_path = '../data/insurance_fraud_data.csv'
    
    df = pd.read_csv(data_path)
    df = df.drop(['claim_number', 'claim_date', 'zip_code'], axis=1)
    
    if df['fraud reported'].dtype == 'O':
        df['fraud reported'] = df['fraud reported'].map({'Y': 1, 'N': 0})
    
    categorical_cols = ['gender', 'property_status', 'claim_day_of_week', 'accident_site', 'channel', 'vehicle_category', 'vehicle_color']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    X = df.drop('fraud reported', axis=1)
    y = df['fraud reported']
    
    feature_cols = X.columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_cols

def visualize_performance():
    X_scaled, y, feature_cols = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load Models
    models = {}
    
    # 1. XGBoost (Optimized)
    try:
        xgb_artifacts = joblib.load('../models/fraud_model_artifacts.pkl')
        models['XGBoost (Optimized)'] = xgb_artifacts['model']
    except:
        print("XGBoost artifacts not found.")

    # 2. Custom Logistic
    try:
        custom_artifacts = joblib.load('../models/fraud_model_custom_artifacts.pkl')
        models['Custom Logistic'] = custom_artifacts['model']
    except:
        print("Custom Logistic artifacts not found.")

    # plots setup
    plt.figure(figsize=(18, 12))
    sns.set_style('whitegrid')

    # ROC Curve
    plt.subplot(2, 2, 1)
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1] if name != 'Custom Logistic' else model.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # Precision-Recall Curve
    plt.subplot(2, 2, 2)
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1] if name != 'Custom Logistic' else model.predict_proba(X_test)
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            avg_p = average_precision_score(y_test, y_prob)
            plt.plot(recall, precision, label=f'{name} (AP = {avg_p:.2f})')
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')

    # Feature Importance (for XGBoost)
    plt.subplot(2, 2, 3)
    if 'XGBoost (Optimized)' in models:
        xgb_model = models['XGBoost (Optimized)']
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[-10:] # Top 10
        plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
        plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
        plt.title('Top 10 Feature Importances (XGBoost)')
        plt.xlabel('Relative Importance')

    # Model Comparison Metrics
    plt.subplot(2, 2, 4)
    metrics_data = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics_data.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0)
        })
    
    metrics_df = pd.DataFrame(metrics_data).melt(id_vars='Model', var_name='Metric', value_name='Score')
    sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_df, palette='viridis')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('comprehensive_performance_report.png')
    plt.close()
    print("Comprehensive visualization saved: comprehensive_performance_report.png")

if __name__ == "__main__":
    visualize_performance()
