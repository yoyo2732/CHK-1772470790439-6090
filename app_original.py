from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# ---------------- LOAD DATASET ----------------

try:
    schemes_df = pd.read_csv("datasets/schemes.csv")
except:
    print("⚠️ Scheme dataset not found")
    schemes_df = pd.DataFrame()

# ---------------- LOAD ML MODELS ----------------

try:
    scheme_model = joblib.load("models/scheme_recommender.pkl")
    cat_encoder = joblib.load("models/category_encoder.pkl")
    state_encoder = joblib.load("models/state_encoder.pkl")
except:
    scheme_model = None
    cat_encoder = None
    state_encoder = None
    print("⚠️ Scheme ML model not loaded")

try:
    fraud_model = joblib.load("models/fraud_detector_model.pkl")
except:
    fraud_model = None
    print("⚠️ Fraud detection model not loaded")


# ---------------- HOME PAGE ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- SCHEME RECOMMENDATION ----------------

def recommend_schemes(age, income, category, state):

    if scheme_model is not None:

        try:
            cat_encoded = cat_encoder.transform([category])[0]
        except:
            cat_encoded = 0

        try:
            state_encoded = state_encoder.transform([state])[0]
        except:
            state_encoded = 0

        user_vector = [[age, income, cat_encoded, state_encoded]]

        distances, indices = scheme_model.kneighbors(user_vector)

        recommended = schemes_df.iloc[indices[0]]

        return recommended.to_dict(orient="records")

    # -------- FALLBACK FILTER --------

    filtered = schemes_df[
        (schemes_df["Eligibility_Age_Min"] <= age) &
        (schemes_df["Eligibility_Age_Max"] >= age) &
        (schemes_df["Max_Income_Allowed"] >= income)
    ].copy()

    if category:
        filtered = filtered[
            filtered["Category"].str.lower().str.contains(category.lower(), na=False)
        ]

    if state:
        filtered = filtered[
            filtered["State_Availability"].str.lower().str.contains(state.lower(), na=False)
        ]

    return filtered.to_dict(orient="records")


# ---------------- SCHEMES PAGE ----------------

@app.route("/schemes")
def schemes_page():

    age = request.args.get("age")
    income = request.args.get("income")
    category = request.args.get("category")
    state = request.args.get("state")

    filtered = []

    for _, row in schemes_df.iterrows():

        # AGE FILTER
        if age:
            if not (int(row["Eligibility_Age_Min"]) <= int(age) <= int(row["Eligibility_Age_Max"])):
                continue

        # INCOME FILTER
        if income:
            if int(income) > int(row["Max_Income_Allowed"]):
                continue

        # CATEGORY FILTER
        if category and category != "":
            if category.lower() not in str(row["Category"]).lower():
                continue

        # STATE FILTER
        if state:
            if state.lower() not in str(row["State_Availability"]).lower():
                continue

        filtered.append(row.to_dict())

    return render_template("schemes.html", schemes=filtered)


# ---------------- FIND SCHEMES (FORM SUBMIT) ----------------

@app.route("/find_schemes", methods=["POST"])
def find_schemes():

    age = int(request.form["age"])
    income = int(request.form["income"])
    category = request.form["category"]
    state = request.form["state"]

    filtered = schemes_df[
        (schemes_df["Eligibility_Age_Min"] <= age) &
        (schemes_df["Eligibility_Age_Max"] >= age) &
        (schemes_df["Max_Income_Allowed"] >= income)
    ]

    if category:
        filtered = filtered[
            filtered["Category"].str.lower().str.contains(category.lower(), na=False)
        ]

    if state:
        filtered = filtered[
            filtered["State_Availability"].str.lower().str.contains(state.lower(), na=False)
        ]

    return render_template("schemes.html", schemes=filtered.to_dict(orient="records"))


# ---------------- FRAUD PAGE ----------------

@app.route("/fraud", methods=["GET","POST"])
def fraud_page():

    fraud_result = None

    if request.method == "POST":

        applications = int(request.form.get("Applications_Submitted",0))
        family_size = int(request.form.get("Family_Size",1))
        previous = int(request.form.get("Previous_Schemes",0))

        # Improved rule based fraud score
        score = 0

        if applications > 5:
            score += 2

        if previous > 4:
            score += 2

        if family_size > 6:
            score += 1

        fraud_probability = min(100, score * 25)

        fraud_detected = fraud_probability >= 50

        fraud_result = {
            "fraud_detected": fraud_detected,
            "probability": fraud_probability
        }

    return render_template("fraud.html", fraud_result=fraud_result)


# ---------------- ML FRAUD API ----------------

@app.route("/check_fraud", methods=["POST"])
def check_fraud():

    if fraud_model is None:
        return jsonify({"result": "Fraud model not loaded"})

    data = request.json

    applications = int(data["applications"])
    documents = 1 if data["documents"] == "Yes" else 0
    bank_linked = 1 if data["bank"] == "Yes" else 0
    location_match = 1 if data["location"] == "Yes" else 0

    features = np.array([[applications, documents, bank_linked, location_match]])

    prediction = fraud_model.predict(features)[0]

    if prediction == 1:
        result = "⚠️ Potential Fraud Detected"
    else:
        result = "✅ Application Looks Safe"

    return jsonify({"result": result})


# ---------------- HELP PAGE ----------------

@app.route("/help")
def help_page():
    return render_template("help.html")


# ---------------- RUN APP ----------------

if __name__ == "__main__":
    app.run(debug=True)