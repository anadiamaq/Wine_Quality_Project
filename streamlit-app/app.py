import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
 
# load model
model = load_model('static/model.h5')
