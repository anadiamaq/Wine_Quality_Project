from main import app
from flask import request
from prep_function import preprocessing
from tensorflow.keras.models import load_model

model =  load_model('model.h5')

@app.get('/predict')
def predict_authentication():
    alcohol = request.args.get('alcohol')
    chlorides = request.args.get('chlorides')
    citric_acid = request.args.get('citric_acid')
    density = request.args.get('density')
    fixed_acidity = request.args.get('fixed_acidity')
    free_sulfur_dioxide = request.args.get('free_sulfur_dioxide')
    pH = request.args.get('pH')
    residual_sugar = request.args.get('residual_sugar')
    sulphates = request.args.get('sulphates')
    total_sulfur_dioxide = request.args.get('total_sulfur_dioxide')
    type = request.args.get('type')
    volatile_acidity = request.args.get('volatile_acidity')
    pred = preprocessing(alcohol, chlorides, citric_acid, density, fixed_acidity, free_sulfur_dioxide,
    pH, residual_sugar, sulphates, total_sulfur_dioxide, type, volatile_acidity)
    prediction = model.predict(pred)
    return str(prediction)
