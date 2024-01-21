import streamlit as st
from StreamLitDash import dash
from streamlit_home import func
from streamlit_prediction import predict
page = st.sidebar.selectbox("Choose Your Menu", ("Home Page","BI and Visualizations","Predict"))
if page == "Predict":
    predict()
if page == "Home Page":
    func()
if page == "BI and Visualizations":
    dash()
