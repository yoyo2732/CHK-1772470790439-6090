"""
Train Scheme Recommender Model
Run: python3 train_scheme_model.py
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import joblib, os

os.makedirs('models', exist_ok=True)

df = pd.read_csv('datasets/schemes.csv')
print("Loaded", len(df), "schemes")

cat_enc = LabelEncoder()
occ_enc = LabelEncoder()
df['Category_enc'] = cat_enc.fit_transform(df['Category'].astype(str))
df['Occ_enc']      = occ_enc.fit_transform(df['Occupation_Required'].astype(str))

feat_cols = ['Eligibility_Age_Min','Eligibility_Age_Max','Max_Income_Allowed',
             'Min_Land_Acres','Category_enc','Occ_enc']

X = df[feat_cols].fillna(0).values

model = NearestNeighbors(n_neighbors=min(8, len(df)), algorithm='ball_tree')
model.fit(X)
print("Scheme recommender trained on", len(df), "schemes")

joblib.dump(model,   'models/scheme_recommender.pkl')
joblib.dump(cat_enc, 'models/scheme_cat_encoder.pkl')
joblib.dump(occ_enc, 'models/scheme_occ_encoder.pkl')
print("Saved successfully!")
