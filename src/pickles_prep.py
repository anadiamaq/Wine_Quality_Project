from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import pickle

wine_qa = pd.read_csv('data/wine-qa.csv')

X = wine_qa.drop(columns=['quality'])

le = LabelEncoder()
le.fit(X['type'])
with open('encoder.pkl','wb') as file:
    pickle.dump(le,file)

x = X.drop(columns=['type'])
sc = StandardScaler()
sc.fit(x)
with open('scaler.pkl','wb') as file:
    pickle.dump(sc,file)