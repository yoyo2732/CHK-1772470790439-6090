"""
Train Fraud Detection Model
Run: python3 train_fraud_model.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib, os

os.makedirs('models', exist_ok=True)

df = pd.read_csv('datasets/fraud.csv')
print("Loaded fraud data. Shape:", df.shape)

encoders = {}
for col in ['Gender', 'State', 'District', 'Profession']:
    le = LabelEncoder()
    df[col+'_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

df['Income_Per_Member']    = df['Annual_Income'] / df['Family_Size'].replace(0,1)
df['Benefit_Income_Ratio'] = df['Previous_Scheme_Benefits'] / df['Annual_Income'].replace(0,1)
df['App_Density']          = df['Applications_Submitted'] / df['Family_Size'].replace(0,1)

feat_cols  = ['Age','Gender_enc','State_enc','District_enc','Profession_enc',
              'Annual_Income','Family_Size','Bank_Account_Linked',
              'Previous_Scheme_Benefits','Applications_Submitted',
              'Income_Per_Member','Benefit_Income_Ratio','App_Density']

feat_names = ['Age','Gender','State','District','Profession',
              'Annual_Income','Family_Size','Bank_Account_Linked',
              'Previous_Scheme_Benefits','Applications_Submitted',
              'Income_Per_Member','Benefit_Income_Ratio','App_Density']

X = df[feat_cols].values
y = df['Fraud_Label'].values

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
model.fit(X, y)
print("Fraud model accuracy:", round(model.score(X,y)*100,1), "%")

joblib.dump(model,      'models/fraud_detector.pkl')
joblib.dump(encoders,   'models/fraud_encoders.pkl')
joblib.dump(feat_names, 'models/fraud_features.pkl')
print("Saved successfully!")
