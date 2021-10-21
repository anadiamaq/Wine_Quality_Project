from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import pickle

def preprocessing(alcohol,chlorides,citric_acid,density,fixed_acidity,free_sulfur_dioxide,pH,
residual_sugar,sulphates,total_sulfur_dioxide,type,volatile_acidity):
    with open('scaler.pkl', 'rb') as file_sc:
        sc = pickle.load(file_sc)
        sc.transform(alcohol,chlorides,citric_acid,density,fixed_acidity,
                free_sulfur_dioxide,pH,residual_sugar,sulphates,total_sulfur_dioxide,
                volatile_acidity)
    with open('encoder.pkl','rb') as file_le:
        le = pickle.load(file_le)
        le.transform(type)
    return 

def preprocessing(*args):
    if args == int:
        with open('scaler.pkl', 'rb') as file_sc:
            sc = pickle.load(file_sc)
            sc.transform(args)
    if args == str:
        with open('encoder.pkl','rb') as file_le:
            le = pickle.load(file_le)
            le.transform(args)
    return 