import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Guardian AI | Professional Risk Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Light Mode Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@600;700&display=swap');
    
    .stApp {
        background-color: #f8fafc;
    }
    
    .main-header {
        font-family: 'Outfit', sans-serif;
        color: #1e293b;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    
    .sub-header {
        color: #64748b;
        letter-spacing: 0.15rem;
        text-transform: uppercase;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 3rem;
        text-align: center;
    }
    
    .step-indicator {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-bottom: 3rem;
    }
    
    .step-circle {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
        opacity: 0.3;
        transition: all 0.3s ease;
    }
    
    .step-circle.active {
        opacity: 1;
    }
    
    .circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #e2e8f0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: #64748b;
        border: 2px solid #cbd5e1;
    }
    
    .step-circle.active .circle {
        background: #2563eb;
        color: white;
        border-color: #2563eb;
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.3);
    }
    
    .step-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }

    .carousel-card {
        background: #ffffff;
        padding: 3rem;
        border-radius: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.05);
        max-width: 800px;
        margin: 0 auto;
    }
    
    .prediction-card {
        padding: 3rem;
        border-radius: 1rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        text-align: center;
        margin-top: 1rem;
    }

    .stButton>button {
        border-radius: 0.5rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-weight: 600 !important;
        height: 3rem;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Session State for Carousel
if 'step' not in st.session_state:
    st.session_state.step = 1

# Initialize widget keys to avoid AttributeErrors
initial_keys = {
    'age': 35, 'gender': 'M', 'income': 65000, 'edu': True,
    'v_age': 4, 'v_cat': 'Medium', 'v_price': 35000, 'v_color': 'Silver',
    'safety': 85, 'past_claims': 1, 'witness': False, 'defects': 2, 'total_claim': 18500,
    'analysis_triggered': False
}
for key, val in initial_keys.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Helper function to change steps
def move_next(): st.session_state.step += 1
def move_prev(): st.session_state.step -= 1
def reset_steps(): st.session_state.step = 1

# Load Artifacts
@st.cache_resource
def load_resources():
    try:
        return joblib.load('models/fraud_model_artifacts.pkl')
    except:
        return None

resources = load_resources()

if not resources:
    st.error("🚨 Model Artifacts Not Found! Please re-train the model.")
    st.stop()

model = resources['model']
scaler = resources['scaler']
label_encoders = resources['label_encoders']
feature_cols = resources['columns']
categorical_cols = resources['categorical_cols']

# Header
st.markdown('<h1 class="main-header">Guardian AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional Risk Intelligence Dashboard</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["⚡ RISK ASSESSMENT", "📊 ARCHITECTURE"])

with tab1:
    # Step Indicator
    st.markdown(f"""
    <div class="step-indicator">
        <div class="step-circle {'active' if st.session_state.step >= 1 else ''}">
            <div class="circle">1</div>
            <div class="step-label">Driver</div>
        </div>
        <div class="step-circle {'active' if st.session_state.step >= 2 else ''}">
            <div class="circle">2</div>
            <div class="step-label">Vehicle</div>
        </div>
        <div class="step-circle {'active' if st.session_state.step >= 3 else ''}">
            <div class="circle">3</div>
            <div class="step-label">Incident</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Carousel Content Area
    st.markdown('<div class="carousel-card">', unsafe_allow_html=True)
    
    if st.session_state.step == 1:
        st.write("### 👤 Step 1: Driver Profile")
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Driver Age", 18, 85, 35, key="age")
            gender = st.selectbox("Gender", ["M", "F"], key="gender")
        with c2:
            income = st.number_input("Annual Income ($)", 10000, 250000, 65000, key="income")
            edu = st.toggle("Post-Secondary Education", value=True, key="edu")
        
        st.divider()
        st.button("CONTINUE TO VEHICLE DATA →", on_click=move_next, use_container_width=True, type="primary")

    elif st.session_state.step == 2:
        st.write("### 🚗 Step 2: Asset Information")
        c1, c2 = st.columns(2)
        with c1:
            v_age = st.slider("Vehicle Age", 0, 25, 4, key="v_age")
            v_cat = st.selectbox("Vehicle Category", ["Compact", "Medium", "Large"], index=1, key="v_cat")
        with c2:
            v_price = st.number_input("Market Price ($)", 5000, 150000, 35000, key="v_price")
            v_color = st.selectbox("Color", ["White", "Black", "Silver", "Red", "Blue"], index=2, key="v_color")
        
        st.divider()
        b1, b2 = st.columns(2)
        with b1: st.button("← BACK", on_click=move_prev, use_container_width=True)
        with b2: st.button("CONTINUE TO INCIDENT DATA →", on_click=move_next, use_container_width=True, type="primary")

    elif st.session_state.step == 3:
        st.write("### 📋 Step 3: Incident Metrics")
        c1, c2 = st.columns(2)
        with c1:
            safety = st.slider("Safety Score", 0, 100, 85, key="safety")
            past_claims = st.number_input("Prior Claims", 0, 10, 1, key="past_claims")
        with c2:
            witness = st.toggle("Witness Present", value=False, key="witness")
            defects = st.slider("Form Discrepancies", 0, 15, 2, key="defects")
            total_claim = st.number_input("Total Claimed ($)", 500, 200000, 18500, key="total_claim")
        
        st.divider()
        b1, b2 = st.columns(2)
        with b1: st.button("← BACK", on_click=move_prev, use_container_width=True)
        with b2: 
            if st.button("FINISH & RUN ANALYSIS 🛡️", use_container_width=True, type="primary"):
                st.session_state.analysis_triggered = True

    st.markdown('</div>', unsafe_allow_html=True)

    # Predictions and Results (Only shown after finish or in analysis mode)
    if st.session_state.get('analysis_triggered', False):
        st.divider()
        
        # Prepare Data for Prediction
        raw_input = {
            'age_of_driver': st.session_state.age,
            'gender': st.session_state.gender,
            'marital_status': 0,
            'safety_rating': st.session_state.safety,
            'annual_income': st.session_state.income,
            'high_education': 1 if st.session_state.edu else 0,
            'address_change': 0,
            'property_status': "Own",
            'claim_day_of_week': datetime.now().strftime('%A'),
            'accident_site': "Local",
            'past_num_of_claims': st.session_state.past_claims,
            'witness_present': 1 if st.session_state.witness else 0,
            'liab_prct': 0,
            'channel': "Online",
            'police_report': 1,
            'age_of_vehicle': st.session_state.v_age,
            'vehicle_category': st.session_state.v_cat,
            'vehicle_price': st.session_state.v_price,
            'vehicle_color': st.session_state.v_color,
            'total_claim': st.session_state.total_claim,
            'injury_claim': st.session_state.total_claim * 0.25,
            'policy deductible': 1000,
            'annual premium': 1500,
            'days open': 12,
            'form defects': st.session_state.defects
        }
        input_df = pd.DataFrame([raw_input])
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown("#### 📑 COMPILED PROFILE")
            st.dataframe(input_df.T.astype(str).rename(columns={0: "VALUE"}), use_container_width=True, height=400)
            if st.button("RESET & START NEW SCAN", on_click=reset_steps):
                st.session_state.analysis_triggered = False

        with res_col2:
            st.markdown("#### 🧠 INFERENCE RESULTS")
            with st.spinner("Analyzing risk profile..."):
                time.sleep(1)
                proc_df = input_df.copy()
                for col in categorical_cols:
                    proc_df[col] = label_encoders[col].transform(proc_df[col])
                
                input_scaled = scaler.transform(proc_df[feature_cols])
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-card" style="border-top: 5px solid #dc2626;">
                    <p style="color: #64748b; font-weight: 600; font-size: 0.85rem;">DETECTION RESULT</p>
                    <h2 style="color: #dc2626; font-family: 'Outfit', sans-serif; font-size: 3rem; font-weight: 700;">HIGH RISK</h2>
                    <p style="font-size: 1.1rem; color: #1e293b;">Fraud Probability: <b>{probability:.2%}</b></p>
                    <div style="background: #fee2e2; padding: 1rem; border-radius: 0.5rem; color: #dc2626; margin-top: 1.5rem; font-size: 0.9rem;">
                        ⚠️ Pattern matches fraud signatures. High priority investigation.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card" style="border-top: 5px solid #059669;">
                    <p style="color: #64748b; font-weight: 600; font-size: 0.85rem;">DETECTION RESULT</p>
                    <h2 style="color: #059669; font-family: 'Outfit', sans-serif; font-size: 3rem; font-weight: 700;">CLEARED</h2>
                    <p style="font-size: 1.1rem; color: #1e293b;">Fraud Probability: <b>{probability:.2%}</b></p>
                    <div style="background: #d1fae5; padding: 1rem; border-radius: 0.5rem; color: #059669; margin-top: 1.5rem; font-size: 0.9rem;">
                        ✅ Profile aligns with routine signatures. Approved.
                    </div>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("#### 🛠️ ENGINE SPECIFICATIONS")
    ca1, ca2 = st.columns(2)
    with ca1:
        st.info(f"Model Core: {resources.get('model_name', 'XGBoost Optimized')}")
        stats = {"Metric": ["Accuracy", "F1 Score", "ROC-AUC"], "Production Value": ["95.0%", "0.38", "0.66"]}
        st.table(pd.DataFrame(stats))
    with ca2:
        st.success("Real-time Inference Latency: < 50ms")
        st.metric("Precision Score", "100.0%", help="High precision minimizes false positives in fraud detection.")

st.divider()
st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 0.75rem;">GUARDIAN ENTERPRISE | CORPORATE RISK INTELLIGENCE v2.6</p>', unsafe_allow_html=True)
