"""
GovShield - ML Model Training Script
Trains both fraud detection and scheme recommendation models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

os.makedirs("models", exist_ok=True)

# ================================================================
# PART 1: FRAUD DETECTION MODEL
# ================================================================
print("Training Fraud Detection Model...")

df_fraud = pd.read_csv("datasets/fraud.csv")

# Feature engineering
df_fraud["Income_Per_Member"] = df_fraud["Annual_Income"] / df_fraud["Family_Size"].replace(0, 1)
df_fraud["Benefit_Income_Ratio"] = df_fraud["Previous_Scheme_Benefits"] / df_fraud["Annual_Income"].replace(0, 1)
df_fraud["App_Density"] = df_fraud["Applications_Submitted"] / df_fraud["Family_Size"].replace(0, 1)

# Encode categoricals
cat_cols = ["Gender", "State", "District", "Profession"]
fraud_encoders = {}
for col in cat_cols:
    if col in df_fraud.columns:
        le = LabelEncoder()
        df_fraud[col] = le.fit_transform(df_fraud[col].astype(str))
        fraud_encoders[col] = le

# Drop identifier columns
drop_cols = ["Person_ID", "Name", "Aadhaar_No", "Income_Cert_No", "PAN_No"]
for col in drop_cols:
    if col in df_fraud.columns:
        df_fraud.drop(columns=[col], inplace=True)

feature_cols = [c for c in df_fraud.columns if c != "Fraud_Label"]
X = df_fraud[feature_cols]
y = df_fraud["Fraud_Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
fraud_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
fraud_model.fit(X_train, y_train)

acc = accuracy_score(y_test, fraud_model.predict(X_test))
print(f"Fraud Model Accuracy: {acc:.2f}")

joblib.dump(fraud_model, "models/fraud_detector.pkl")
joblib.dump(fraud_encoders, "models/fraud_encoders.pkl")
joblib.dump(list(X.columns), "models/fraud_features.pkl")
print("Fraud model saved!")

# ================================================================
# PART 2: SCHEME RECOMMENDATION MODEL  
# ================================================================
print("Training Scheme Recommendation Model...")

df_schemes = pd.read_csv("datasets/schemes.csv")
df_schemes["Avg_Age"] = (df_schemes["Eligibility_Age_Min"] + df_schemes["Eligibility_Age_Max"]) / 2

cat_enc = LabelEncoder()
occ_enc = LabelEncoder()

df_schemes["Category_enc"] = cat_enc.fit_transform(df_schemes["Category"])
df_schemes["Occupation_enc"] = occ_enc.fit_transform(df_schemes["Occupation_Required"])

scheme_features = df_schemes[["Avg_Age", "Max_Income_Allowed", "Category_enc", "Occupation_enc", "Min_Land_Acres"]]
scheme_model = NearestNeighbors(n_neighbors=min(7, len(df_schemes)), metric="euclidean")
scheme_model.fit(scheme_features)

joblib.dump(scheme_model, "models/scheme_recommender.pkl")
joblib.dump(cat_enc, "models/scheme_cat_encoder.pkl")
joblib.dump(occ_enc, "models/scheme_occ_encoder.pkl")
print(f"Scheme model saved! ({len(df_schemes)} schemes)")
print("All models trained successfully!")
