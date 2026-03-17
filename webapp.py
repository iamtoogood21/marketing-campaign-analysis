import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Title
st.title("📊 Marketing Campaign Analysis App")

# Load Data
df = pd.read_csv("data/marketing_data.csv")

# Show Data
st.subheader("Dataset Preview")
st.dataframe(df)

# Charts
st.subheader("Response Distribution")
st.bar_chart(df["response"].value_counts())

st.subheader("Income Distribution")
st.line_chart(df["income"])

# Model Training
X = df[["age","income","previous_purchases","website_visits"]]
y = df["response"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# User Input
st.header("🔮 Predict Customer Response")

age = st.slider("Age", 18, 70, 30)
income = st.slider("Income", 20000, 100000, 50000)
purchases = st.slider("Previous Purchases", 0, 10, 2)
visits = st.slider("Website Visits", 0, 20, 5)

# Prediction
prediction = model.predict([[age, income, purchases, visits]])

if prediction[0] == 1:
    st.success("✅ Customer is likely to respond")
else:
    st.error("❌ Customer is unlikely to respond")