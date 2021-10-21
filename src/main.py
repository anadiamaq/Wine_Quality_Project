from flask import Flask
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

app = Flask("Wine_Quality_Project-api")


@app.route("/")
def title():
    return '<h1 align="center">Wine Quality Project</h1>'