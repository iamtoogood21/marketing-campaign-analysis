import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("Marketing Campaign Predictor")

df = pd.read_csv("marketing_data.csv")

X = df[["age","income","previous_purchases"]]
y = df["response"]

model = LogisticRegression()
model.fit(X,y)

st.header("Enter Customer Information")

age = st.slider("Age",18,70)
income = st.slider("Income",20000,100000)
purchases = st.slider("Previous Purchases",0,10)

prediction = model.predict([[age,income,purchases]])

if prediction[0] == 1:
    st.success("Customer likely to respond to campaign")
else:
    st.error("Customer unlikely to respond")
    st.subheader("Dataset Overview")

st.write(df)

st.bar_chart(df["previous_purchases"])