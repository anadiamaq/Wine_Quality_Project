from app import app
import pandas as pd
from flask import request
from prep_function import preprocess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import deserialize as layer_from_config


@app.route("/")
def title():
    return '<h1 align="center">Wine Quality Project</h1>'


@app.route('/predict', methods=['POST'])
def predict_authentication():
    data = pd.DataFrame([request.json])
    pred = preprocess(data)
    model = Sequential()
    model.add(Dense(12, activation='relu',
              input_shape=(12,), kernel_regularizer='l2'))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae'])
    model.load_weights('weights.h5')
    print(model.get_weights())
    prediction = model.predict(pred)
    print(prediction)
    return {"result": int(prediction[0][0])}
