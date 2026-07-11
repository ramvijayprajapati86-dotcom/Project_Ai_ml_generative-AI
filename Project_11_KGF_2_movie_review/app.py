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
# Custom CSS
# ---------------------------------------------------
st.markdown("""
<style>

body{
    background-color:#0E1117;
}

.main{
    background-color:#0E1117;
}

h1,h2,h3{
    color:#E50914;
}

.stButton>button{
    background-color:#E50914;
    color:white;
    border:none;
    border-radius:10px;
    height:3.2em;
    width:100%;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background-color:#B20710;
    color:white;
}

div[data-testid="metric-container"]{
    background:#1F2937;
    border-radius:10px;
    padding:15px;
}

textarea{
    font-size:18px !important;
}

hr{
    border:1px solid #333333;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Hugging Face Model
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
st.sidebar.image("poster.jpg", use_container_width=True)

st.sidebar.title("📌 About Project")

st.sidebar.markdown("""
### NLP Project

This project analyzes **Netflix Movie KGF Chapter 2 Reviews**
using **Large Language Models (LLMs)**.

### Technologies

- 🤗 Hugging Face Transformers
- 🐍 Python
- 📊 Pandas
- 🌐 Streamlit
- 🧠 DistilBERT

### Features

✅ Sentiment Analysis

✅ Positive/Negative Prediction

✅ Confidence Score

✅ Dataset Preview
""")

# ---------------------------------------------------
# Header Image
# ---------------------------------------------------
st.image("poster.jpg", use_container_width=True)

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.markdown(
"""
<h1 style='text-align:center;color:#E50914;'>
🎬 Analyzing Netflix Movie KGF Chapter 2 Reviews with LLMs
</h1>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div style="text-align:center;font-size:20px;">

This application predicts whether a **Netflix movie review**
is **Positive 😊** or **Negative 😞**
using the pretrained **DistilBERT Transformer Model**.

</div>
""",
unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------
# User Input
# ---------------------------------------------------
st.subheader("✍️ Enter Movie Review")

review = st.text_area(
    "",
    height=170,
    placeholder="Example: KGF 2 is an amazing movie with outstanding action scenes and performances."
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("🎯 Analyze Sentiment"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a movie review.")

    else:

        result = classifier(review)

        sentiment = result[0]["label"]
        confidence = result[0]["score"]

        st.divider()

        st.subheader("Prediction Result")

        if sentiment == "POSITIVE":

            st.success("😊 Positive Review")

        else:

            st.error("😞 Negative Review")

        st.metric(
            label="Confidence Score",
            value=f"{confidence*100:.2f}%"
        )

# ---------------------------------------------------
# Dataset Preview
# ---------------------------------------------------
st.divider()

st.header("📊 Dataset Preview")

try:

    df = pd.read_csv("netflix movie KGF 2-2.csv", delimiter=";")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"Total Rows : {df.shape[0]}")

    with col2:
        st.info(f"Total Columns : {df.shape[1]}")

except FileNotFoundError:

    st.warning("Dataset not found. Please place 'netflix movie KGF 2-2.csv' in the project folder.")

except Exception as e:

    st.error(f"Error loading dataset: {e}")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.divider()

st.markdown(
"""
<div style='text-align:center;color:gray;'>

### 🚀 Built with

🤗 Hugging Face Transformers |
🐍 Python |
📊 Pandas |
🌐 Streamlit |
🧠 DistilBERT

**NLP Project – KGF Chapter 2 Movie Review Sentiment Analysis**

</div>
""",
unsafe_allow_html=True
)
