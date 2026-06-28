
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set page styling and configuration
st.set_page_config(
    page_title="Employee Attrition Prediction System",
    page_icon="💼",
    layout="wide"
)

# Custom CSS for modern, professional styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .reportview-container {
        background: #f8f9fa;
    }
    .stButton>button {
        background-color: #2b5c8f;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #1e3f63;
        color: white;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

import os

# Cache the data loader and preprocessor to run instantly on user clicks
@st.cache_resource
def load_resources():
    # Build absolute paths based on the script location so it works locally and on Cloud
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'final_optimized_rf_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    data_path = os.path.join(base_dir, 'data', 'attrition_clean.csv')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    clean_df = pd.read_csv(data_path)
    X_train_raw = clean_df.drop(columns=['Attrition'], errors='ignore')
    X_train_encoded = X_train_raw.copy()
    
    binary_mappings = {'OverTime': {'Yes': 1, 'No': 0}, 'Gender': {'Male': 1, 'Female': 0}}
    for col, mapping in binary_mappings.items():
        X_train_encoded[col] = X_train_encoded[col].map(mapping)
        
    multi_cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
    X_train_encoded = pd.get_dummies(X_train_encoded, columns=multi_cat_cols, drop_first=False)
    
    X_train_scaled_cols = X_train_encoded.columns.tolist()
    
    defaults = {}
    for col in X_train_raw.columns:
        if X_train_raw[col].dtype == 'object':
            defaults[col] = X_train_raw[col].mode()[0]
        else:
            defaults[col] = int(X_train_raw[col].median())
            
    return model, scaler, X_train_scaled_cols, defaults

# Load resources
try:
    model, scaler, scaled_cols, defaults = load_resources()
except Exception as e:
    st.error(f"Error loading models or datasets. Please run the Day 4 optimization notebook first. Error: {e}")
    st.stop()

# App Header
st.title("💼 Employee Attrition Prediction Dashboard")
st.write("Evaluate the risk of employee turnover using our optimized Random Forest model.")
st.markdown("---")

# Setup Tabs
tab1, tab2 = st.tabs(["👤 Single Employee Predictor", "📁 Bulk CSV Predictor"])

# ==========================================
# TAB 1: SINGLE PREDICTOR
# ==========================================
with tab1:
    st.write("Enter an employee's professional and demographic details below to evaluate their risk of leaving.")
    
    # Layout: 2 Columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Demographic & Role Information")
        age = st.slider("Age", 18, 60, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        distance = st.slider("Distance From Home (miles)", 1, 30, 9)
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician", 
            "Manufacturing Director", "Healthcare Representative", "Manager", 
            "Sales Representative", "Research Director", "Human Resources"
        ])
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        business_travel = st.selectbox("Business Travel Frequency", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

    with col2:
        st.subheader("💰 Work & Satisfaction Details")
        monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
        overtime = st.selectbox("Overtime Status", ["Yes", "No"])
        job_satisfaction = st.slider("Job Satisfaction Rating (1-4)", 1, 4, 3)
        env_satisfaction = st.slider("Environment Satisfaction Rating (1-4)", 1, 4, 3)
        work_life_balance = st.slider("Work-Life Balance Rating (1-4)", 1, 4, 3)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_with_manager = st.slider("Years with Current Manager", 0, 20, 3)

    # Button to Predict
    st.markdown("### Evaluation")
    if st.button("Run Attrition Risk Predictor"):
        user_input = defaults.copy()
        user_input['Age'] = age
        user_input['Gender'] = gender
        user_input['MaritalStatus'] = marital_status
        user_input['DistanceFromHome'] = distance
        user_input['JobRole'] = job_role
        user_input['Department'] = department
        user_input['BusinessTravel'] = business_travel
        user_input['MonthlyIncome'] = monthly_income
        user_input['OverTime'] = overtime
        user_input['JobSatisfaction'] = job_satisfaction
        user_input['EnvironmentSatisfaction'] = env_satisfaction
        user_input['WorkLifeBalance'] = work_life_balance
        user_input['YearsAtCompany'] = years_at_company
        user_input['YearsWithCurrManager'] = years_with_manager
        
        input_df = pd.DataFrame([user_input])
        
        binary_mappings = {'OverTime': {'Yes': 1, 'No': 0}, 'Gender': {'Male': 1, 'Female': 0}}
        for col, mapping in binary_mappings.items():
            input_df[col] = input_df[col].map(mapping)
            
        multi_cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
        input_df_encoded = pd.get_dummies(input_df, columns=multi_cat_cols, drop_first=False)
        
        input_df_final = pd.DataFrame(columns=scaled_cols)
        for col in scaled_cols:
            if col in input_df_encoded.columns:
                input_df_final.loc[0, col] = input_df_encoded.loc[0, col]
            else:
                input_df_final.loc[0, col] = 0
                
        input_df_final = input_df_final.astype(float)
        
        numeric_cols = scaler.feature_names_in_
        input_df_final[numeric_cols] = scaler.transform(input_df_final[numeric_cols])
        
        input_df_scaled = input_df_final
        
        prediction = model.predict(input_df_scaled)[0]
        probability = model.predict_proba(input_df_scaled)[0][1] 
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            if prediction == 1:
                st.error("⚠️ Attrition Risk: **HIGH RISK**")
                st.write("This employee is likely to leave the organization. Proactive retention measures are recommended.")
            else:
                st.success("✅ Attrition Risk: **LOW RISK**")
                st.write("This employee is likely to stay. Continue maintaining engagement and job satisfaction.")
                
        with p_col2:
            st.metric(label="Attrition Probability", value=f"{probability * 100:.2f}%")
            st.progress(probability)
            
        # --- Feature Importance Plot ---
        st.markdown("---")
        st.subheader("🔍 What drives this model?")
        st.write("Top 10 features the model considers most important for predicting attrition:")
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': scaled_cols, 'Importance': importances})
        top_10_features = importance_df.sort_values(by='Importance', ascending=False).head(10).set_index('Feature')
        st.bar_chart(top_10_features)

# ==========================================
# TAB 2: BULK PREDICTOR
# ==========================================
with tab2:
    st.write("Upload a CSV file containing multiple employee records. The system will process all of them and return a downloadable sheet with predictions.")
    
    # 1. File Uploader Widget
    uploaded_file = st.file_uploader("Upload Employee Data (CSV format)", type=["csv"])
    
    if uploaded_file is not None:
        # Load the uploaded file
        bulk_df = pd.read_csv(uploaded_file)
        
        # Display preview of what was uploaded
        st.write(f"✅ Successfully loaded **{len(bulk_df)}** employee records. Generating predictions...")
        
        # We need to process the data exactly like we did for a single record!
        # Step A: Fill missing columns with dataset defaults so the model doesn't crash
        processed_bulk = bulk_df.copy()
        for col in defaults.keys():
            if col not in processed_bulk.columns:
                processed_bulk[col] = defaults[col]
                
        # Step B: Binary mappings
        binary_mappings = {'OverTime': {'Yes': 1, 'No': 0}, 'Gender': {'Male': 1, 'Female': 0}}
        for col, mapping in binary_mappings.items():
            if processed_bulk[col].dtype == 'object':
                processed_bulk[col] = processed_bulk[col].map(mapping)
                
        # Step C: One-hot encoding
        multi_cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
        bulk_encoded = pd.get_dummies(processed_bulk, columns=multi_cat_cols, drop_first=False)
        
        # Step D: Column alignment
        bulk_final = pd.DataFrame(columns=scaled_cols)
        for col in scaled_cols:
            if col in bulk_encoded.columns:
                bulk_final[col] = bulk_encoded[col]
            else:
                bulk_final[col] = 0
                
        bulk_final = bulk_final.astype(float)
        
        # Step E: Scaling
        numeric_cols = scaler.feature_names_in_
        bulk_final[numeric_cols] = scaler.transform(bulk_final[numeric_cols])
        
        # Step F: Run Prediction for all rows
        predictions = model.predict(bulk_final)
        probabilities = model.predict_proba(bulk_final)[:, 1] # Probability of Class 1 (Yes)
        
        # Step G: Append results to the original dataframe the user uploaded
        results_df = bulk_df.copy()
        
        # Add beautiful formatting to the final output
        results_df.insert(0, 'Attrition_Probability', [f"{prob*100:.1f}%" for prob in probabilities])
        results_df.insert(0, 'Attrition_Prediction', ['High Risk ⚠️' if p == 1 else 'Low Risk ✅' for p in predictions])
        
        # Display the results table
        st.markdown("### Batch Prediction Results")
        st.dataframe(results_df, use_container_width=True)
        
        # Provide a Download Button
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_data,
            file_name='employee_attrition_predictions.csv',
            mime='text/csv',
        )

