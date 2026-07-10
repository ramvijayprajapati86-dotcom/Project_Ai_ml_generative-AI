import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Employee Retention Prediction",
    page_icon="👨‍💼",
    layout="centered"
)

st.title("👨‍💼 Employee Retention Prediction")
st.write("Predict whether an employee is likely to leave the company using Logistic Regression.")

# -----------------------------------
# Load Dataset
# -----------------------------------
@st.cache_data
def load_data():
    data_path = Path(__file__).parent / "HR_comma_sep.csv"

    # Debug information
    st.write("CSV Path:", data_path)
    st.write("File Exists:", data_path.exists())

    if not data_path.exists():
        st.error(f"CSV file not found!\nExpected location:\n{data_path}")
        st.stop()

    return pd.read_csv(data_path)

df = load_data()

# -----------------------------------
# Prepare Data
# -----------------------------------
features = [
    "satisfaction_level",
    "average_montly_hours",
    "promotion_last_5years",
    "salary"
]

subdf = df[features].copy()

X = pd.get_dummies(subdf, columns=["salary"], prefix="salary")
y = df["left"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.3,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------------
# User Input
# -----------------------------------
st.subheader("Enter Employee Details")

satisfaction_level = st.slider(
    "Satisfaction Level",
    0.0,
    1.0,
    0.5,
    0.01
)

average_montly_hours = st.number_input(
    "Average Monthly Hours",
    min_value=1,
    max_value=500,
    value=200
)

promotion_last_5years = st.selectbox(
    "Promotion in Last 5 Years",
    [0, 1],
    format_func=lambda x: "Yes" if x else "No"
)

salary = st.selectbox(
    "Salary Level",
    ["low", "medium", "high"]
)

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict Employee Retention"):

    input_data = pd.DataFrame([{
        "satisfaction_level": satisfaction_level,
        "average_montly_hours": average_montly_hours,
        "promotion_last_5years": promotion_last_5years,
        "salary": salary
    }])

    input_data = pd.get_dummies(
        input_data,
        columns=["salary"],
        prefix="salary"
    )

    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("Prediction: Employee is likely to leave the company.")
    else:
        st.success("Prediction: Employee is likely to stay in the company.")

    st.write(f"Probability of Leaving: {probability:.2%}")

# -----------------------------------
# Model Details
# -----------------------------------
st.subheader("Model Details")
st.write(f"Model Accuracy: {model.score(X_test, y_test):.2%}")
st.write("Algorithm: Logistic Regression")
