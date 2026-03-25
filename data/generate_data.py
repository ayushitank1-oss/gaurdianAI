import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_data(n_samples=1000):
    np.random.seed(42)
    random.seed(42)
    
    data = []
    start_date = datetime(2022, 1, 1)
    
    for i in range(n_samples):
        claim_number = f"CLM{1000 + i}"
        age_of_driver = np.random.randint(18, 85)
        gender = random.choice(['M', 'F'])
        marital_status = random.choice([0, 1])
        safety_rating = np.random.randint(2, 101)
        annual_income = np.random.randint(20000, 150000)
        high_education = random.choice([0, 1])
        address_change = random.choice([0, 1])
        property_status = random.choice(['Own', 'Rent'])
        zip_code = np.random.randint(10000, 99999)
        
        days_offset = np.random.randint(0, 730)
        claim_date = start_date + timedelta(days=days_offset)
        claim_day_of_week = claim_date.strftime('%A')
        
        accident_site = random.choice(['Highway', 'Local', 'Parking Lot'])
        past_num_of_claims = np.random.randint(0, 6)
        witness_present = random.choice([0, 1])
        liab_prct = np.random.randint(0, 101)
        channel = random.choice(['Broker', 'Phone', 'Online'])
        police_report = random.choice([0, 1])
        age_of_vehicle = np.random.randint(0, 15)
        vehicle_category = random.choice(['Compact', 'Large', 'Medium'])
        vehicle_price = np.random.randint(5000, 80000)
        vehicle_color = random.choice(['White', 'Black', 'Silver', 'Red', 'Blue'])
        
        total_claim = np.random.randint(1000, 50000)
        injury_claim = int(total_claim * np.random.uniform(0.1, 0.6))
        policy_deductible = random.choice([500, 1000, 1500, 2000])
        annual_premium = np.random.randint(500, 3000)
        days_open = np.random.randint(1, 100)
        form_defects = np.random.randint(0, 14)
        
        # Logic for fraud (probabilistic)
        fraud_prob = 0.05
        if form_defects > 8: fraud_prob += 0.2
        if past_num_of_claims > 3: fraud_prob += 0.15
        if witness_present == 0: fraud_prob += 0.1
        if safety_rating < 30: fraud_prob += 0.1
        
        fraud_reported = 'Y' if np.random.random() < fraud_prob else 'N'
        
        data.append([
            claim_number, age_of_driver, gender, marital_status, safety_rating,
            annual_income, high_education, address_change, property_status,
            zip_code, claim_date.strftime('%Y-%m-%d'), claim_day_of_week,
            accident_site, past_num_of_claims, witness_present, liab_prct,
            channel, police_report, age_of_vehicle, vehicle_category,
            vehicle_price, vehicle_color, total_claim, injury_claim,
            policy_deductible, annual_premium, days_open, form_defects,
            fraud_reported
        ])
        
    columns = [
        'claim_number', 'age_of_driver', 'gender', 'marital_status', 'safety_rating',
        'annual_income', 'high_education', 'address_change', 'property_status',
        'zip_code', 'claim_date', 'claim_day_of_week', 'accident_site',
        'past_num_of_claims', 'witness_present', 'liab_prct', 'channel',
        'police_report', 'age_of_vehicle', 'vehicle_category', 'vehicle_price',
        'vehicle_color', 'total_claim', 'injury_claim', 'policy deductible',
        'annual premium', 'days open', 'form defects', 'fraud reported'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('insurance_fraud_data.csv', index=False)
    print("Dataset generated: insurance_fraud_data.csv")

if __name__ == "__main__":
    generate_data(2000)
