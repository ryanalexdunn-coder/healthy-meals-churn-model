import streamlit as st
import pandas as pd
import pickle


# -------------------------
# Load model and encoder
# -------------------------

with open("churn_rf_healthy_meals.pkl", "rb") as f:
    model = pickle.load(f)

with open("churn_encoder_healthy_meals.pkl", "rb") as f:
    encoder = pickle.load(f)


# -------------------------
# App title
# -------------------------

st.title("Healthy Meals Churn Prediction Model")

st.write(
    "Enter customer information to predict renewal probability and churn risk."
)


# -------------------------
# User Inputs
# -------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

total_num_sessions = st.number_input(
    "Total Number of Sessions",
    min_value=0,
    value=50
)

gross_total_session_length = st.number_input(
    "Gross Total Session Length",
    min_value=0,
    value=1000
)

active_days = st.number_input(
    "Active Days",
    min_value=0,
    value=30
)

active_quarters = st.number_input(
    "Active Quarters",
    min_value=0,
    value=4
)

avg_sessions_per_active_quarter = st.number_input(
    "Average Sessions Per Active Quarter",
    min_value=0.0,
    value=10.0
)

tech_comfort_score = st.number_input(
    "Technology Comfort Score",
    min_value=1,
    max_value=10,
    value=5
)


income_level = st.selectbox(
    "Income Level",
    ["High", "Low", "Medium", "Very High"]
)

education = st.selectbox(
    "Education",
    ["Graduate", "High School", "Other", "Post-Graduate"]
)

device_type = st.selectbox(
    "Device Type",
    ["Desktop-only", "Mobile-only", "Multi-device"]
)


# -------------------------
# Prediction
# -------------------------

if st.button("Predict Churn Risk"):

    # categorical variables
    raw = pd.DataFrame([{
        "EDUCATION": education,
        "INCOME_LEVEL": income_level,
        "DEVICE_TYPE": device_type
    }])


    # match encoder training order
    raw = raw[encoder.feature_names_in_]


    # encode categories
    encoded = encoder.transform(raw)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out()
    )


    # numeric variables
    numeric_df = pd.DataFrame([{
        "AGE": age,
        "TOTAL_NUM_SESSIONS": total_num_sessions,
        "GROSS_TOTAL_SESSION_LENGTH": gross_total_session_length,
        "ACTIVE_DAYS": active_days,
        "ACTIVE_QUARTERS": active_quarters,
        "AVG_SESSIONS_PER_ACTIVE_QUARTER": avg_sessions_per_active_quarter,
        "TECH_COMFORT_SCORE": tech_comfort_score
    }])


    # combine features
    input_df = pd.concat(
        [numeric_df, encoded_df],
        axis=1
    )


    # match exact model feature order
    input_df = input_df[model.feature_names_in_]


    # prediction
    renewal_probability = model.predict_proba(input_df)[0][1]

    churn_probability = 1 - renewal_probability


    # output
    st.success(
        f"Renewal Probability: {renewal_probability:.2%}"
    )

    st.warning(
        f"Churn Probability: {churn_probability:.2%}"
    )


    if churn_probability >= 0.60:
        risk = "High Churn Risk"
    elif churn_probability >= 0.40:
        risk = "Medium Churn Risk"
    else:
        risk = "Low Churn Risk"


    st.info(
        f"Risk Category: {risk}"
    )