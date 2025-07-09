

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, encoders, and column order
model = joblib.load("best_model1.pkl")
label_encoders = joblib.load("label_encoders1.pkl")
feature_columns = [col.lower() for col in joblib.load("feature_columns1.pkl")]

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("Salary Prediction App")

mode = st.radio("Select Prediction Mode:", ["Individual Entry", "Bulk Upload"])

# Define inputs -> to be used in dropdowns or options
job_roles = ['Software Engineer', 'QA Engineer', 'Data Analyst', 'Backend Developer', 'ML Engineer',
             'Network Engineer', 'Financial Analyst', 'Risk Analyst', 'Underwriter', 'Product Manager',
             'UI/UX Designer', 'Academic Counselor', 'Research Associate', 'Mechanical Engineer',
             'Sales Executive', 'Operations Executive', 'Brand Manager', 'Travel Consultant',
             'Agronomist', 'Site Engineer', 'Video Editor', 'Journalist']

industries = ['IT Services', 'Banking', 'E-commerce', 'EdTech', 'Healthcare', 'Manufacturing', 'Retail',
              'Telecom', 'Logistics', 'FMCG', 'Travel & Tourism', 'Agriculture', 'Education',
              'Pharma', 'Real Estate', 'Insurance', 'Media', 'Entertainment']

locations = ['Mumbai', 'Bengaluru', 'Delhi', 'Hyderabad', 'Chennai', 'Pune', 'Noida',
             'Gurugram', 'Ahmedabad', 'Lucknow', 'Indore', 'Jaipur']

company_names = ["Infosys", "TCS", "Wipro", "HDFC Bank", "Flipkart", "BYJU'S", "Apollo Hospitals",
                 "Tata Motors", "Cognizant", "Reliance Retail", "Airtel", "Delhivery", "HUL",
                 "ICICI Bank", "MakeMyTrip", "Godrej Agrovet", "Aakash Institute", "Netflix India",
                 "Sun Pharma", "Prestige Group", "LIC India", "India Today"]

# Define company -> Autofil based on company
company_profiles = {
    "Infosys":         {"industry": "IT Services", "company size": "MNC", "company type": "Private"},
    "TCS":             {"industry": "IT Services", "company size": "MNC", "company type": "Public"},
    "Wipro":           {"industry": "IT Services", "company size": "MNC", "company type": "Private"},
    "HDFC Bank":       {"industry": "Banking", "company size": "MNC", "company type": "Private"},
    "Flipkart":        {"industry": "E-commerce", "company size": "Large", "company type": "Private"},
    "BYJU'S":          {"industry": "EdTech", "company size": "Large", "company type": "Private"},
    "Apollo Hospitals":{"industry": "Healthcare", "company size": "Large", "company type": "Private"},
    "Tata Motors":     {"industry": "Manufacturing", "company size": "MNC", "company type": "Public"},
    "Cognizant":       {"industry": "IT Services", "company size": "MNC", "company type": "Private"},
    "Reliance Retail": {"industry": "Retail", "company size": "MNC", "company type": "Private"},
    "Airtel":          {"industry": "Telecom", "company size": "MNC", "company type": "Private"},
    "Delhivery":       {"industry": "Logistics", "company size": "Large", "company type": "Private"},
    "HUL":             {"industry": "FMCG", "company size": "MNC", "company type": "Public"},
    "ICICI Bank":      {"industry": "Banking", "company size": "MNC", "company type": "Private"},
    "MakeMyTrip":      {"industry": "Travel & Tourism", "company size": "Large", "company type": "Private"},
    "Godrej Agrovet":  {"industry": "Agriculture", "company size": "Large", "company type": "Public"},
    "Aakash Institute":{"industry": "Education", "company size": "Medium", "company type": "Private"},
    "Netflix India":   {"industry": "Entertainment", "company size": "Large", "company type": "Private"},
    "Sun Pharma":      {"industry": "Pharma", "company size": "MNC", "company type": "Public"},
    "Prestige Group":  {"industry": "Real Estate", "company size": "Large", "company type": "Private"},
    "LIC India":       {"industry": "Insurance", "company size": "MNC", "company type": "Government"},
    "India Today":     {"industry": "Media", "company size": "Large", "company type": "Private"},
}

# input create for remainging columns
company_types = ['Public', 'Private', 'Government', 'NGO']
genders = ['Male', 'Female']
job_locations = ['On-site', 'Hybrid', 'Remote']
employment_statuses = ['Full-time', 'Part-time', 'Intern', 'Freelancer']

# Select the mode (Individual or Bulk)
# Individual BLock --------------------------------------------------------------------------------------------------------------------------------------
if mode == "Individual Entry":
    st.subheader("Enter Candidate Details")
    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox("Company Name", company_names)
        gender = st.selectbox("Gender", genders)
        location = st.selectbox("Location", locations)
        job_loc = st.selectbox("Job Location", job_locations)
        emp_status = st.selectbox("Employment Status", employment_statuses)
        working_hours = st.slider("Avg Working Hours", 30, 60, 40)
        rating = st.slider("Latest Rating", 1.0, 5.0, 4.0, 0.1)
    with col2:
        profile = company_profiles.get(company, {})
        industry = st.selectbox("Industry", [profile.get("industry")] if profile else industries)
        company_size = st.selectbox("Company Size", [profile.get("company size")] if profile else ["Start-up", "Small", "Medium", "Large", "MNC"])
        company_type = st.selectbox("Company Type", [profile.get("company type")] if profile else company_types)
        job_role = st.selectbox("Job Role", job_roles)
        age = st.slider("Age", 21, 60, 28)
        experience = st.slider("Experience (Years)", 0, 35, 2)
        certifications = st.number_input("Certifications", min_value=0, max_value=10, value=1)

    if st.button("Predict Salary"):
        input_df = pd.DataFrame([{
            "company name": company,
            "gender": gender,
            "location": location,
            "job location": job_loc,
            "employment status": emp_status,
            "avg working hours": working_hours,
            "latest rating": rating,
            "industry": industry,
            "company size": company_size,
            "company type": company_type,
            "job role": job_role,
            "age": age,
            "experience": experience,
            "certifications": certifications
        }])

        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str).apply(lambda x: x if x in le.classes_ else 'Unknown')
                if 'Unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'Unknown')
                input_df[col] = le.transform(input_df[col])

        input_df = input_df[feature_columns]
        salary_pred = model.predict(input_df)
        st.success(f"Estimated Salary: ₹{int(salary_pred[0]):,}/year")
# Bulk data predict ---------------------------------------------------------------------------------------------------------------------
else:
    st.subheader("Upload CSV File")
    compare_actual = st.checkbox("Compare with Actual Salary")

    st.markdown("### Download Sample Format")

# Define empty sample DataFrame -> if not present then you can download and use it
    empty_sample = pd.DataFrame(columns=[
        "company name", "gender", "location", "job location", "employment status",
        "avg working hours", "latest rating", "industry", "company size",
        "company type", "job role", "age", "experience", "certifications",
        "actual salary"  # Optional, only used if comparison is enabled
    ])

# CSV Download -> Sample
    csv_sample = empty_sample.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Empty Sample CSV",
        data=csv_sample,
        file_name="sample_salary_input.csv",
        mime="text/csv"
    )

# Optional preview -> Sample
    if st.checkbox("Preview Sample Format"):
        st.dataframe(empty_sample)
# Data Uploaded
    uploaded_file = st.file_uploader("Upload CSV with candidate records", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df_bulk = pd.read_csv(uploaded_file)
        else:
            df_bulk = pd.read_excel(uploaded_file)

        df_bulk.columns = df_bulk.columns.str.strip().str.lower()
        actual_salary_present = 'actual salary' in df_bulk.columns

        if not all(col in df_bulk.columns for col in feature_columns):
            st.error("CSV must contain all required feature columns:")
            st.write(feature_columns)
        elif compare_actual and not actual_salary_present:
            st.error("To compare with actual salary, please include an 'Actual Salary' column.")
        else:
            remark_rows = []
            for idx, row in df_bulk.iterrows():
                row_remarks = []
                for col in feature_columns:
                    if pd.isnull(row[col]):
                        if df_bulk[col].dtype == 'object':
                            fill_val = df_bulk[col].mode()[0]
                            df_bulk.at[idx, col] = fill_val
                            row_remarks.append(f"{col}: filled with '{fill_val}'")
                        else:
                            fill_val = round(df_bulk[col].mean(), 2)
                            df_bulk.at[idx, col] = fill_val
                            row_remarks.append(f"{col}: filled with {fill_val}")
                remark_rows.append("; ".join(row_remarks) if row_remarks else "No Missing Data")

            for col, le in label_encoders.items():
                if col in df_bulk.columns:
                    df_bulk[col] = df_bulk[col].astype(str).apply(lambda x: x if x in le.classes_ else 'Unknown')
                    if 'Unknown' not in le.classes_:
                        le.classes_ = np.append(le.classes_, 'Unknown')
                    df_bulk[col] = le.transform(df_bulk[col])

            df_model_input = df_bulk[feature_columns]
            df_bulk["Predicted Salary"] = model.predict(df_model_input).astype(int)

            decoded_df = df_bulk.copy()
            for col, le in label_encoders.items():
                if col in decoded_df.columns:
                    decoded_df[col] = le.inverse_transform(decoded_df[col])

            decoded_df["Remarks"] = remark_rows

            # Comparison Metrics
            if compare_actual:
                with st.expander("Click to View Model Evaluation Metrics"):
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                    actual = df_bulk["actual salary"].values
                    predicted = df_bulk["Predicted Salary"].values
                    mae = mean_absolute_error(actual, predicted)
                    rmse = mean_squared_error(actual, predicted, squared=False)
                    r2 = r2_score(actual, predicted)

                    st.write(f"**MAE**: ₹{mae:,.2f}")
                    st.write(f"**RMSE**: ₹{rmse:,.2f}")
                    st.write(f"**R² Score**: {r2:.4f}")

            st.success("Predictions complete. Preview below:")
            st.dataframe(decoded_df)

            csv = decoded_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predicted CSV", csv, "salary_predictions.csv", "text/csv")
