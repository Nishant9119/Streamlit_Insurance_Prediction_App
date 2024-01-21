import streamlit as st
from StreamLitDash import dash
from streamlit_prediction import predict
def func():
    st.title("Welcome to CarSure")
    st.markdown("# A Claim Analytics App!")
    st.write(
        "Explore insightful visualizations on the Dashboard page and predict fraud claims in advance on the Predictions page.")
    st.markdown("## Key Features:")
    st.write("- Interactive BI and visualizations on the Dashboard.")
    st.write("- Advanced prediction model for detecting potential fraud claims.")
    st.markdown("## App Overview:")
    st.write("Our Claim Analytics App empowers you to:")
    st.write("- Gain insights into claim data through interactive visualizations.")
    st.write("- Predict potential fraud claims using advanced machine learning techniques.")
    st.markdown("---")
    st.write("Claim Analytics App v1.0 | © 2023 Nishant Sharma")



