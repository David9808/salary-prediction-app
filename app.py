import streamlit as st
import joblib
import pandas as pd

# Load the trained pipeline
pipeline = joblib.load('salary_prediction_pipeline.pkl')

df = pd.read_csv('salary_data.csv')

# Streamlit app
st.set_page_config(page_title="Salary Predictor", page_icon="💰")
st.title("Salary Predictor")
st.write("Enter the details below to predict your estimated salary:")

# inputs
st.header("Job Details")
job_simp = st.selectbox(
    "Job Title", 
    df['job_simp'].unique())

job_state = st.selectbox(
    "State",
    df['job_state'].unique()
)

seniority = st.selectbox(
    "Seniority Level",
    df.seniority.unique()
)

sector = st.selectbox(
    "Sector",
    df.Sector.unique()
)

type_of_ownership = st.selectbox(
    "Company Ownership",
    df['Type of ownership'].unique()
)

size = st.selectbox(
    "Company Size",
    df.Size.unique()
)

# Boolean inputs
st.header("Skills")
aws = st.checkbox("AWS")
excel = st.checkbox("Excel")
python_yn = st.checkbox("Python")
r_yn = st.checkbox("R")
spark = st.checkbox("Spark")
hourly = st.checkbox("Hourly Job")

if st.button("Predict Salary"):
    input_data = pd.DataFrame({
        'job_simp': [job_simp],
        'job_state': [job_state],
        'seniority': [seniority],
        'Sector': [sector],
        'Type of ownership': [type_of_ownership],
        'Size': [size],
        'aws': [int(aws)],
        'excel': [int(excel)],
        'python_yn': [int(python_yn)],
        'R_yn': [int(r_yn)],
        'spark': [int(spark)],
        'hourly': [int(hourly)],
    })

    # Predict salary
    try:
        prediction = pipeline.predict(input_data)
        st.success(f"Estimated Salary: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")