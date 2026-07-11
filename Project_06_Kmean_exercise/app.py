import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="K-Means Clustering", page_icon="🌸")

st.title("🌸 K-Means Clustering on Iris Dataset")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df = df[['petal length (cm)', 'petal width (cm)']]

st.subheader("Dataset")
st.dataframe(df.head())

k = st.slider("Select Number of Clusters", 2, 6, 3)

model = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = model.fit_predict(df)

st.subheader("Clustered Data")
st.dataframe(df.head())

centers = pd.DataFrame(
    model.cluster_centers_,
    columns=["petal length (cm)", "petal width (cm)"]
)

fig = px.scatter(
    df,
    x="petal length (cm)",
    y="petal width (cm)",
    color=df["Cluster"].astype(str),
    title="K-Means Clustering"
)

fig.add_scatter(
    x=centers["petal length (cm)"],
    y=centers["petal width (cm)"],
    mode="markers",
    marker=dict(symbol="x", size=14),
    name="Centroids"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Elbow Method")

sse = []
for i in range(1, 10):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(df[['petal length (cm)', 'petal width (cm)']])
    sse.append(km.inertia_)

elbow = pd.DataFrame({
    "Clusters": range(1, 10),
    "SSE": sse
})

fig2 = px.line(
    elbow,
    x="Clusters",
    y="SSE",
    markers=True,
    title="Elbow Method"
)

st.plotly_chart(fig2, use_container_width=True)
