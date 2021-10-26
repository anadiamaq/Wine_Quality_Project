import streamlit as st
import pandas as pd
import requests
import json
import os
from dotenv import load_dotenv


load_dotenv()
BACKEND_URL = os.environ.get("BACKEND_URL")

st.markdown(
    "<h1 style='color:#56070C; text-align:center; margin-bottom:40px' >Wine Quality Project</h1>", unsafe_allow_html=True
)

st.image(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../images/st-wine.jpg')))

col1, col2 = st.columns(2)

with st.form(key='my_form'):

    with col1:

        alcohol = st.number_input(
            label='Alcohol (vol. %)', min_value=3.5, max_value=15.0, format="%.1f")

        chlorides = st.number_input(
            label='Chlorides (g(sodium chloride)/dm^3)', min_value=0.000, max_value=0.500, format="%.3f")

        citric_acid = st.number_input(
            label='Citric acid (g/dm^3)', min_value=0.00, max_value=1.50)

        density = st.number_input(
            label='Density (g/cm^3)', min_value=0.00, max_value=1.20)

        fixed_acidity = st.number_input(
            label='Fixed acidity (g(tartaric acid)/dm^3)', min_value=0.0, max_value=17.0, format="%.1f")

        free_sulfur_dioxide = st.number_input(
            label='Free sulfur dioxide (mg/dm^3)', min_value=0.0, max_value=200.0, format="%.1f")

    with col2:

        pH = st.number_input(
            label='pH', min_value=0.00, max_value=14.00)

        residual_sugar = st.number_input(
            label='Residual sugar (g/dm^3)', min_value=0.0, max_value=30.0, format="%.1f")

        sulphates = st.number_input(
            label='Sulphates (g(potassium sulphate)/dm^3)', min_value=0.00, max_value=2.00)

        total_sulfur_dioxide = st.number_input(
            label='Total sulfur dioxide (mg/dm^3)', min_value=0.0, max_value=300.0, format="%.1f")

        type = st.selectbox('Select Type', ['red', 'white'], key=2)

        volatile_acidity = st.number_input(
            label='Volatile acidity (g(acetic acid)/dm^3)', min_value=0.00, max_value=1.50)

    submit = st.form_submit_button(label='Predict')

    dict = {"alcohol": alcohol, "chlorides": chlorides, "citric_acid": citric_acid, "density": density, "fixed_acidity": fixed_acidity, "free_sulfur_dioxide": free_sulfur_dioxide,
            "pH": pH, "residual_sugar": residual_sugar, "sulphates": sulphates, "total_sulfur_dioxide": total_sulfur_dioxide, "type": type,  "volatile_acidity": volatile_acidity}
    data_json = json.dumps(dict, indent=1)

    print(data_json)

    x = requests.post(f'{BACKEND_URL}/predict', json=data_json)

    print(x.text)
    if submit == True:
        if int(x.text) < 10:
            st.write('The quality is:')
            x.text
        else:
            st.write(
                "Probably the physicochemical parameters are not real or that's not wine!!!")
