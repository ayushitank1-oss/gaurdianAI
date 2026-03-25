import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess():
    data_path = '../data/insurance_fraud_data_cleaned.csv'
    if not os.path.exists(data_path):
        data_path = '../data/insurance_fraud_data.csv'
    
    df = pd.read_csv(data_path)
    df = df.drop(['claim_number', 'claim_date', 'zip_code'], axis=1)
    
    if df['fraud reported'].dtype == 'O':
        df['fraud reported'] = df['fraud reported'].map({'Y': 1, 'N': 0})
    
    categorical_cols = ['gender', 'property_status', 'claim_day_of_week', 'accident_site', 'channel', 'vehicle_category', 'vehicle_color']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    X = df.drop('fraud reported', axis=1)
    y = df['fraud reported']
    
    feature_cols = X.columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, label_encoders, feature_cols, categorical_cols

def hyperparameter_tune(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models_to_tune = {
        'RandomForest': (RandomForestClassifier(class_weight='balanced', random_state=42), {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }),
        'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'scale_pos_weight': [1, 3] # Handling imbalance
        })
    }
    
    best_models = {}
    
    for name, (model, params) in models_to_tune.items():
        print(f"Tuning {name}...")
        search = RandomizedSearchCV(model, params, n_iter=10, cv=cv, scoring='f1', n_jobs=-1, random_state=42)
        search.fit(X, y)
        best_models[name] = search.best_estimator_
        print(f"Best {name} Score: {search.best_score_:.4f}")
        print(f"Best Params: {search.best_params_}")
        
    return best_models

def main():
    X_all, y_all, scaler, label_encoders, feature_cols, categorical_cols = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    
    best_models = hyperparameter_tune(X_train, y_train)
    
    print("\n--- Final Model Comparison (Test Set) ---")
    results = []
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'F1': f1,
            'ROC-AUC': auc
        })
        print(f"{name}: Acc={acc:.4f}, F1={f1:.4f}, ROC-AUC={auc:.4f}")

    # Select best model based on F1 (important for fraud)
    results_df = pd.DataFrame(results)
    best_name = results_df.sort_values(by='F1', ascending=False).iloc[0]['Model']
    final_model = best_models[best_name]
    
    print(f"\nFinal Selected Model: {best_name}")
    
    # Save optimized artifacts
    artifacts = {
        'model': final_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'columns': feature_cols,
        'categorical_cols': categorical_cols,
        'model_name': best_name,
        'timestamp': time.ctime()
    }
    
    # Overwrite the main artifacts file for production use
    joblib.dump(artifacts, 'fraud_model_artifacts.pkl')
    print("Optimized artifacts saved to fraud_model_artifacts.pkl")

if __name__ == "__main__":
    main()
