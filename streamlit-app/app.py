import streamlit as st
import pandas as pd
import requests

response = requests.get("http://localhost:5000/predict")

st.markdown(
    "<h1 style='text-align: center' >Wine Quality Project</h1>", unsafe_allow_html=True
)

with st.form(key='my_form'):
    text_input = st.number_input(
        label='Alcohol', min_value=3.5, max_value=15.0)

    text_input = st.number_input(
        label='Chlorides', min_value=0.000, max_value=0.500)

    text_input = st.number_input(
        label='Citric acid', min_value=0.00, max_value=1.50)

    text_input = st.number_input(
        label='Density', min_value=0.00, max_value=1.20)

    text_input = st.number_input(
        label='Fixed acidity', min_value=0.0, max_value=17.0)

    text_input = st.number_input(
        label='Free sulfur dioxide', min_value=0.0, max_value=200.0)

    text_input = st.number_input(
        label='pH', min_value=0.00, max_value=14.00)

    text_input = st.number_input(
        label='Residual sugar', min_value=0.0, max_value=30.0)

    text_input = st.number_input(
        label='Sulphates', min_value=0.00, max_value=2.00)

    text_input = st.number_input(
        label='Total sulfur dioxide', min_value=0.0, max_value=300.0)

    st.selectbox('Select Type', ['red', 'white'], key=2)

    text_input = st.number_input(
        label='Volatile acidity', min_value=0.00, max_value=1.50)
    submit_button = st.form_submit_button(label='Submit')
