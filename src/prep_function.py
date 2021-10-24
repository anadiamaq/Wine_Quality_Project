import pickle


def preprocess(data):  # data es un pd.DataFrame
    with open('scaler.pkl', 'rb') as file_sc:
        sc = pickle.load(file_sc)
    with open('encoder.pkl', 'rb') as file_le:
        le = pickle.load(file_le)
    to_st = ['alcohol', 'chlorides', 'citric_acid', 'density', 'fixed_acidity', 'free_sulfur_dioxide', 'pH',
             'residual_sugar', 'sulphates', 'total_sulfur_dioxide', 'volatile_acidity']
    to_le = ['type']
    data[to_st] = sc.transform(data[to_st])
    data[to_le] = le.transform(data[to_le])
    print(data)
    return data
