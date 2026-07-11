import streamlit as st
import pandas as pd
from transformers import pipeline

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="KGF 2 Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return classifier

classifier = load_model()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("About")

st.sidebar.info(
    """
    NLP Project using Hugging Face LLM

    ✔ Sentiment Analysis
    ✔ Movie Reviews
    ✔ Transformers
    ✔ Streamlit
    """
)

# ---------------------------------------------------
# Main Title
# ---------------------------------------------------
st.title("🎬 Analyzing Netflix Movie KGF Chapter 2 Reviews with LLMs")

st.markdown(
"""
This application predicts whether a movie review is **Positive** or **Negative**
using a pretrained DistilBERT model.
"""
)

# ---------------------------------------------------
# User Input
# ---------------------------------------------------
review = st.text_area(
    "Enter Movie Review",
    height=150,
    placeholder="Example: KGF 2 is an amazing movie with powerful action scenes."
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        result = classifier(review)

        sentiment = result[0]["label"]
        confidence = result[0]["score"]

        st.subheader("Prediction")

        if sentiment == "POSITIVE":
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.metric(
            "Confidence Score",
            f"{confidence*100:.2f}%"
        )

# ---------------------------------------------------
# Dataset Preview
# ---------------------------------------------------
st.divider()

st.header("Dataset Preview")

try:
    df = pd.read_csv("netflix movie KGF 2-2.csv", delimiter=";")
    st.dataframe(df.head())
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

except:
    st.info("Dataset not found. Upload the CSV to the project folder.")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.divider()

st.caption(
    "Built using Streamlit | Hugging Face Transformers | DistilBERT"
)
