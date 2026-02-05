import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Page settings
st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")

st.title("Employee Attrition Prediction App")
st.write("Predict whether an employee is likely to leave the company")

# Load trained model
model = pickle.load(open("models/attrition_model.pkl", "rb"))

# Load dataset (used only to match feature structure)
df = pd.read_csv("data/employee_attrition.csv")

# Encode categorical columns (same logic as training)
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

st.subheader("Enter Employee Details")

# User inputs
age = st.number_input("Age", min_value=18, max_value=60, value=30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=30000)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=1)
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
overtime = st.selectbox("OverTime", ["No", "Yes"])

# Prediction
if st.button("Predict Attrition"):
    # Take one row to preserve all feature names
    input_data = df.drop("Attrition", axis=1).iloc[[0]].copy()

    # Replace selected fields
    input_data["Age"] = age
    input_data["MonthlyIncome"] = monthly_income
    input_data["YearsAtCompany"] = years_at_company
    input_data["JobLevel"] = job_level
    input_data["OverTime"] = 1 if overtime == "Yes" else 0

    # Predict
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Employee is likely to leave the company")
    else:
        st.success("✅ Employee is likely to stay in the company")

