import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Google Play Store Analysis", layout="wide")

st.title("📱 Google Play Store Data Analysis")
st.write("Interactive dashboard using Streamlit")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("hourly_wages_Data_Camp.csv")

try:
    df = load_data()

    st.success("Dataset Loaded Successfully")
    st.write(df.head())

    st.subheader("Dataset Information")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if numeric_cols:
        col = st.selectbox("Select Numeric Column", numeric_cols)

        st.subheader(f"Histogram of {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader(f"Box Plot of {col}")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df[col], ax=ax2)
        st.pyplot(fig2)

    if "Rating" in df.columns:
        st.subheader("Rating Distribution")
        fig3 = px.histogram(df, x="Rating")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

except FileNotFoundError:
    st.error("Dataset file not found. Put hourly_wages_Data_Camp.csv in the same folder as app.py.")

except Exception as e:
    st.error(e)
