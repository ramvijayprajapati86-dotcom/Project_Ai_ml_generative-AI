import streamlit as st

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Star Health Insurance AI",
    page_icon="🏥",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------

st.markdown("""
<style>

/* Background */

.stApp{
background:linear-gradient(135deg,#EAF6FF,#F8FBFF);
}

/* Hide Streamlit Menu */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

/* Title */

.title{
text-align:center;
font-size:42px;
font-weight:bold;
color:#0D6EFD;
}

.subtitle{
text-align:center;
font-size:18px;
color:#666;
margin-bottom:30px;
}

/* Sidebar */

[data-testid="stSidebar"]{
background:#0D6EFD;
}

[data-testid="stSidebar"] *{
color:white;
}

/* Buttons */

.stButton>button{

background:#0D6EFD;
color:white;
border-radius:12px;
height:50px;
font-size:16px;
font-weight:bold;
border:none;
}

.stButton>button:hover{

background:#084298;
color:white;
}

/* Cards */

.card{

background:white;
padding:25px;
border-radius:15px;
box-shadow:0px 5px 20px rgba(0,0,0,.15);
margin-bottom:25px;
}

/* Chat */

[data-testid="stChatMessage"]{

border-radius:15px;
padding:12px;
margin-bottom:10px;
}

/* Metric */

[data-testid="metric-container"]{

background:white;
padding:10px;
border-radius:15px;
box-shadow:0px 3px 12px rgba(0,0,0,.15);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------

st.markdown(
"<h1 class='title'>🏥 Star Health Insurance AI Assistant</h1>",
unsafe_allow_html=True
)

st.markdown(
"<p class='subtitle'>Retrieval Augmented Generation (RAG) Chatbot powered by OpenAI & LangChain</p>",
unsafe_allow_html=True
)

# -------------------------------
# METRICS
# -------------------------------

c1,c2,c3,c4=st.columns(4)

with c1:
    st.metric("📄 Knowledge Base","HTML")

with c2:
    st.metric("🤖 AI","GPT")

with c3:
    st.metric("⚡ Vector DB","FAISS")

with c4:
    st.metric("💬 Status","Ready")

st.write("")

# -------------------------------
# INFORMATION CARD
# -------------------------------

st.markdown("""

<div class="card">

<h3>📘 About This Chatbot</h3>

This chatbot answers questions related to

<ul>

<li>✅ Health Insurance Plans</li>

<li>✅ Maternity Insurance</li>

<li>✅ Family Health Insurance</li>

<li>✅ Senior Citizen Plans</li>

<li>✅ Policy Benefits</li>

<li>✅ Health Insurance Guidance</li>

</ul>

</div>

""",unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------

with st.sidebar:

    st.image(
        "https://www.starhealth.in/static/media/star-health-logo.png",
        width=180
    )

    st.markdown("## ⚙ Configuration")

    api_key=st.text_input(
        "OpenAI API Key",
        type="password"
    )

    st.divider()

    st.success("Knowledge Base Ready")

    st.info("Powered by\n\n• OpenAI\n\n• LangChain\n\n• FAISS")

    st.divider()

    st.markdown("### Sample Questions")

    st.write("• What are maternity plans?")

    st.write("• Benefits of Health Insurance")

    st.write("• Family Health Insurance")

    st.write("• Individual Policy")

    st.write("• Senior Citizen Plans")

# -------------------------------
# CHAT
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages=[]

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

prompt=st.chat_input("Ask your health insurance question...")

if prompt:

    st.session_state.messages.append(
        {"role":"user","content":prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Replace this with your RAG response
    answer="This is where your RAG chatbot answer will appear."

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role":"assistant","content":answer}
    )

