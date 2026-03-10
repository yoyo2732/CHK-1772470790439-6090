"""
GovShield - Government Scheme Awareness & Fraud Detection Platform
Flask Backend with ML Models (RandomForest + KNN)
Multilingual: English, Hindi, Marathi
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
import csv
from datetime import datetime
try:
    import anthropic
    _anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
except Exception:
    _anthropic_client = None

app = Flask(__name__)

# ================================================================
# LOAD DATASETS
# ================================================================
try:
    schemes_df = pd.read_csv("datasets/schemes.csv")
    print(f"Loaded {len(schemes_df)} schemes")
except Exception as e:
    schemes_df = pd.DataFrame()

try:
    fraud_df = pd.read_csv("datasets/fraud.csv")
    print(f"Loaded {len(fraud_df)} fraud records")
except Exception as e:
    fraud_df = pd.DataFrame()

# ================================================================
# LOAD ML MODELS
# ================================================================
try:
    fraud_model    = joblib.load("models/fraud_detector.pkl")
    fraud_encoders = joblib.load("models/fraud_encoders.pkl")
    fraud_features = joblib.load("models/fraud_features.pkl")
    print("Fraud detection model loaded")
except Exception as e:
    fraud_model = fraud_encoders = fraud_features = None
    print(f"Fraud model not loaded: {e}")

try:
    scheme_model     = joblib.load("models/scheme_recommender.pkl")
    scheme_cat_enc   = joblib.load("models/scheme_cat_encoder.pkl")
    scheme_occ_enc   = joblib.load("models/scheme_occ_encoder.pkl")
    print("Scheme recommendation model loaded")
except Exception as e:
    scheme_model = scheme_cat_enc = scheme_occ_enc = None
    print(f"Scheme model not loaded: {e}")

# ================================================================
# MULTILINGUAL TRANSLATIONS DICTIONARY
# ================================================================
TRANSLATIONS = {
    "en": {
        "app_title": "GovShield", "app_subtitle": "Scheme Awareness & Fraud Detection Platform",
        "nav_apply": "Apply", "nav_schemes": "Browse Schemes", "nav_admin": "Admin Panel",
        "form_title": "Applicant Profile", "sec_personal": "Personal Information",
        "sec_economic": "Economic Details", "sec_agriculture": "Agriculture Details",
        "sec_docs": "Documents & Scheme History",
        "f_name": "Full Name", "f_age": "Age", "f_gender": "Gender", "f_caste": "Caste Category",
        "f_state": "State", "f_district": "District", "f_income": "Annual Income (₹)",
        "f_occupation": "Occupation", "f_family": "Family Size", "f_land": "Land Size (acres)",
        "f_crop": "Crop Type", "f_bank": "Bank Account Linked?",
        "f_prev_schemes": "Previous Scheme Benefits Received (₹)",
        "f_apps_submitted": "Number of Previous Applications", "f_disabled": "Person with Disability?",
        "f_documents": "Documents Verified?",
        "btn_submit": "Analyze My Application", "btn_analyzing": "Analyzing...",
        "results_schemes": "Recommended Schemes", "results_fraud": "Fraud Risk Assessment",
        "fraud_score": "Fraud Risk Score", "fraud_low": "LOW RISK", "fraud_medium": "MEDIUM RISK",
        "fraud_high": "HIGH RISK", "eligible": "Eligible", "benefits": "Benefits",
        "apply_link": "Apply Online", "eligibility": "Eligibility", "how_to_apply": "How to Apply",
        "admin_title": "Admin Dashboard", "total_apps": "Total Applications",
        "flagged_apps": "High Risk Flagged", "approved_apps": "Approved", "under_review": "Under Review",
        "recent_apps": "Recent Applications", "col_id": "App ID", "col_name": "Name",
        "col_risk": "Risk Level", "col_schemes": "Schemes", "col_status": "Status",
        "col_flag": "Flag Reason", "col_date": "Date",
        "no_schemes": "No matching schemes found for your profile.",
        "err_name": "Name must be at least 2 characters.", "err_age": "Age must be between 1 and 120.",
        "err_income": "Income must be a positive number.", "err_family": "Family size must be at least 1.",
        "err_fix": "Please fix the errors before submitting.",
        "male": "Male", "female": "Female", "other": "Other", "yes": "Yes", "no": "No",
        "general": "General", "obc": "OBC", "sc": "SC", "st": "ST",
        "farmer": "Farmer", "laborer": "Daily Wage Laborer", "business": "Small Business Owner",
        "teacher": "Teacher/Govt Employee", "student": "Student", "retired": "Retired",
        "none_scheme": "None", "scheme_match": "Match", "ministry": "Ministry",
        "fraud_explanation": "Analysis Details", "flag_reasons": "Flagged Issues",
        "recommendations": "Verification Recommendations",
        "browse_title": "All Government Schemes", "filter_category": "Filter by Category",
        "all_categories": "All Categories", "footer": "GovShield — Making Government Schemes Accessible",
        "powered_by": "AI Powered", "view_details": "View Details", "match_score": "Match Score"
    },
    "hi": {
        "app_title": "गवशील्ड", "app_subtitle": "योजना जागरूकता और धोखाधड़ी पहचान मंच",
        "nav_apply": "आवेदन करें", "nav_schemes": "योजनाएं देखें", "nav_admin": "व्यवस्थापक",
        "form_title": "आवेदक प्रोफाइल", "sec_personal": "व्यक्तिगत जानकारी",
        "sec_economic": "आर्थिक विवरण", "sec_agriculture": "कृषि विवरण",
        "sec_docs": "दस्तावेज़ और योजना इतिहास",
        "f_name": "पूरा नाम", "f_age": "आयु", "f_gender": "लिंग", "f_caste": "जाति वर्ग",
        "f_state": "राज्य", "f_district": "जिला", "f_income": "वार्षिक आय (₹)",
        "f_occupation": "व्यवसाय", "f_family": "परिवार का आकार", "f_land": "भूमि का आकार (एकड़)",
        "f_crop": "फसल का प्रकार", "f_bank": "बैंक खाता जुड़ा हुआ?",
        "f_prev_schemes": "पूर्व योजना लाभ प्राप्त (₹)",
        "f_apps_submitted": "पिछले आवेदनों की संख्या", "f_disabled": "विकलांग व्यक्ति?",
        "f_documents": "दस्तावेज़ सत्यापित?",
        "btn_submit": "मेरे आवेदन का विश्लेषण करें", "btn_analyzing": "विश्लेषण हो रहा है...",
        "results_schemes": "अनुशंसित योजनाएं", "results_fraud": "धोखाधड़ी जोखिम मूल्यांकन",
        "fraud_score": "धोखाधड़ी जोखिम स्कोर", "fraud_low": "कम जोखिम",
        "fraud_medium": "मध्यम जोखिम", "fraud_high": "उच्च जोखिम",
        "eligible": "पात्र", "benefits": "लाभ", "apply_link": "ऑनलाइन आवेदन करें",
        "eligibility": "पात्रता", "how_to_apply": "आवेदन कैसे करें",
        "admin_title": "व्यवस्थापक डैशबोर्ड", "total_apps": "कुल आवेदन",
        "flagged_apps": "उच्च जोखिम चिह्नित", "approved_apps": "स्वीकृत", "under_review": "समीक्षाधीन",
        "recent_apps": "हाल के आवेदन", "col_id": "आवेदन ID", "col_name": "नाम",
        "col_risk": "जोखिम स्तर", "col_schemes": "योजनाएं", "col_status": "स्थिति",
        "col_flag": "चिह्नित कारण", "col_date": "तारीख",
        "no_schemes": "आपकी प्रोफाइल के लिए कोई योजना नहीं मिली।",
        "err_name": "नाम कम से कम 2 अक्षर होना चाहिए।", "err_age": "आयु 1 से 120 के बीच होनी चाहिए।",
        "err_income": "आय एक सकारात्मक संख्या होनी चाहिए।", "err_family": "परिवार कम से कम 1 होना चाहिए।",
        "err_fix": "सबमिट से पहले त्रुटियां ठीक करें।",
        "male": "पुरुष", "female": "महिला", "other": "अन्य", "yes": "हाँ", "no": "नहीं",
        "general": "सामान्य", "obc": "अन्य पिछड़ा वर्ग", "sc": "अनुसूचित जाति", "st": "अनुसूचित जनजाति",
        "farmer": "किसान", "laborer": "दैनिक मजदूर", "business": "लघु व्यवसाय स्वामी",
        "teacher": "शिक्षक/सरकारी कर्मचारी", "student": "छात्र", "retired": "सेवानिवृत्त",
        "none_scheme": "कोई नहीं", "scheme_match": "मिलान", "ministry": "मंत्रालय",
        "fraud_explanation": "विश्लेषण विवरण", "flag_reasons": "चिह्नित मुद्दे",
        "recommendations": "सत्यापन अनुशंसाएं",
        "browse_title": "सभी सरकारी योजनाएं", "filter_category": "श्रेणी के अनुसार फ़िल्टर",
        "all_categories": "सभी श्रेणियां", "footer": "गवशील्ड — सरकारी योजनाओं को सुलभ बनाना",
        "powered_by": "AI संचालित", "view_details": "विवरण देखें", "match_score": "मिलान स्कोर"
    },
    "mr": {
        "app_title": "गवशील्ड", "app_subtitle": "योजना जागरूकता आणि फसवणूक शोध प्लॅटफॉर्म",
        "nav_apply": "अर्ज करा", "nav_schemes": "योजना पहा", "nav_admin": "प्रशासक",
        "form_title": "अर्जदार प्रोफाइल", "sec_personal": "वैयक्तिक माहिती",
        "sec_economic": "आर्थिक तपशील", "sec_agriculture": "शेती तपशील",
        "sec_docs": "कागदपत्रे आणि योजना इतिहास",
        "f_name": "पूर्ण नाव", "f_age": "वय", "f_gender": "लिंग", "f_caste": "जात श्रेणी",
        "f_state": "राज्य", "f_district": "जिल्हा", "f_income": "वार्षिक उत्पन्न (₹)",
        "f_occupation": "व्यवसाय", "f_family": "कुटुंबाचा आकार", "f_land": "जमिनीचा आकार (एकर)",
        "f_crop": "पिकाचा प्रकार", "f_bank": "बँक खाते जोडलेले?",
        "f_prev_schemes": "मागील योजना लाभ मिळाले (₹)",
        "f_apps_submitted": "मागील अर्जांची संख्या", "f_disabled": "अपंग व्यक्ती?",
        "f_documents": "कागदपत्रे सत्यापित?",
        "btn_submit": "माझ्या अर्जाचे विश्लेषण करा", "btn_analyzing": "विश्लेषण होत आहे...",
        "results_schemes": "शिफारस केलेल्या योजना", "results_fraud": "फसवणूक जोखीम मूल्यांकन",
        "fraud_score": "फसवणूक जोखीम स्कोर", "fraud_low": "कमी जोखीम",
        "fraud_medium": "मध्यम जोखीम", "fraud_high": "उच्च जोखीम",
        "eligible": "पात्र", "benefits": "फायदे", "apply_link": "ऑनलाइन अर्ज करा",
        "eligibility": "पात्रता", "how_to_apply": "अर्ज कसा करावा",
        "admin_title": "प्रशासक डॅशबोर्ड", "total_apps": "एकूण अर्ज",
        "flagged_apps": "उच्च जोखीम चिन्हांकित", "approved_apps": "मंजूर", "under_review": "पुनरावलोकनाधीन",
        "recent_apps": "अलीकडील अर्ज", "col_id": "अर्ज ID", "col_name": "नाव",
        "col_risk": "जोखीम पातळी", "col_schemes": "योजना", "col_status": "स्थिती",
        "col_flag": "चिन्हांकित कारण", "col_date": "तारीख",
        "no_schemes": "आपल्या प्रोफाइलसाठी कोणतीही योजना आढळली नाही.",
        "err_name": "नाव किमान 2 अक्षरांचे असणे आवश्यक आहे.", "err_age": "वय 1 ते 120 च्या दरम्यान असणे आवश्यक आहे.",
        "err_income": "उत्पन्न सकारात्मक संख्या असणे आवश्यक आहे.", "err_family": "कुटुंब किमान 1 असणे आवश्यक आहे.",
        "err_fix": "सबमिट करण्यापूर्वी त्रुटी दुरुस्त करा.",
        "male": "पुरुष", "female": "महिला", "other": "इतर", "yes": "होय", "no": "नाही",
        "general": "सामान्य", "obc": "इतर मागासवर्गीय", "sc": "अनुसूचित जाती", "st": "अनुसूचित जमाती",
        "farmer": "शेतकरी", "laborer": "दैनिक मजूर", "business": "लघु व्यवसाय मालक",
        "teacher": "शिक्षक/सरकारी कर्मचारी", "student": "विद्यार्थी", "retired": "निवृत्त",
        "none_scheme": "काहीही नाही", "scheme_match": "जुळणी", "ministry": "मंत्रालय",
        "fraud_explanation": "विश्लेषण तपशील", "flag_reasons": "चिन्हांकित समस्या",
        "recommendations": "सत्यापन शिफारसी",
        "browse_title": "सर्व सरकारी योजना", "filter_category": "श्रेणीनुसार फिल्टर",
        "all_categories": "सर्व श्रेणी", "footer": "गवशील्ड — सरकारी योजना सुलभ करणे",
        "powered_by": "AI समर्थित", "view_details": "तपशील पहा", "match_score": "जुळणी स्कोर"
    }
}

# ================================================================
# FRAUD DETECTION LOGIC (Rule-based + ML Hybrid)
# ================================================================
def detect_fraud(data):
    flags = []
    score = 0
    income = float(data.get("income", 0))
    family_size = int(data.get("family_size", 1))
    prev_benefits = float(data.get("prev_benefits", 0))
    apps_submitted = int(data.get("apps_submitted", 0))
    bank_linked = data.get("bank_linked", "yes")
    documents = data.get("documents", "yes")
    land_size = float(data.get("land_size", 0))
    occupation = data.get("occupation", "farmer")

    if income > 0:
        ratio = prev_benefits / income
        if ratio > 0.8:
            flags.append("Previous benefits exceed 80% of declared income")
            score += 25
        elif ratio > 0.5:
            flags.append("High benefit-to-income ratio detected")
            score += 10

    if apps_submitted > 7:
        flags.append(f"Unusually high number of applications submitted ({apps_submitted})")
        score += 25
    elif apps_submitted > 4:
        flags.append("Above-average number of applications submitted")
        score += 10

    if family_size > 10:
        flags.append("Unusually large family size reported")
        score += 20
    elif family_size > 8 and income < 50000:
        flags.append("Large family size with very low income")
        score += 15

    if bank_linked == "no":
        flags.append("No bank account linked — DBT not possible")
        score += 15

    if documents == "no":
        flags.append("Supporting documents not verified")
        score += 20

    if occupation not in ["farmer"] and land_size > 10:
        flags.append("Large land holding claimed by non-farmer")
        score += 15

    if income > 800000:
        flags.append("Income exceeds threshold for most welfare schemes")
        score += 30

    # ML Model augmentation
    ml_prob = None
    if fraud_model is not None and fraud_features is not None:
        try:
            income_per_member = income / max(family_size, 1)
            benefit_ratio = prev_benefits / max(income, 1)
            app_density = apps_submitted / max(family_size, 1)
            feature_dict = {
                "Age": int(data.get("age", 30)),
                "Annual_Income": income,
                "Family_Size": family_size,
                "Bank_Account_Linked": 1 if bank_linked == "yes" else 0,
                "Previous_Scheme_Benefits": prev_benefits,
                "Applications_Submitted": apps_submitted,
                "Income_Per_Member": income_per_member,
                "Benefit_Income_Ratio": benefit_ratio,
                "App_Density": app_density,
            }
            for col, enc in fraud_encoders.items():
                val = data.get(col.lower(), "")
                try:
                    feature_dict[col] = enc.transform([str(val)])[0]
                except:
                    feature_dict[col] = 0
            feature_vector = [feature_dict.get(f, 0) for f in fraud_features]
            X_pred = np.array([feature_vector])
            ml_prob = fraud_model.predict_proba(X_pred)[0][1]
            score = int(score * 0.6 + ml_prob * 100 * 0.4)
        except Exception as e:
            print(f"ML prediction error: {e}")

    score = min(100, max(0, score))
    level = "HIGH" if score >= 67 else ("MEDIUM" if score >= 34 else "LOW")
    ml_str = f"ML confidence: {ml_prob*100:.0f}%" if ml_prob is not None else "Rule-based analysis"
    explanation = (
        f"AI analysis detected {len(flags)} suspicious pattern(s). "
        f"{ml_str}. Overall fraud risk score: {score}/100."
    )
    recommendations = "Verify: income certificate, Aadhaar linkage, land records, SECC database cross-check."
    return score, level, flags, explanation, recommendations


# ================================================================
# SCHEME RECOMMENDATION
# ================================================================
def recommend_schemes(age, income, occupation, land_size, gender, caste, disabled):
    recommended = []
    if schemes_df.empty:
        return []
    for _, scheme in schemes_df.iterrows():
        if not (int(scheme["Eligibility_Age_Min"]) <= age <= int(scheme["Eligibility_Age_Max"])):
            continue
        if income > int(scheme["Max_Income_Allowed"]):
            continue
        occ_req = str(scheme["Occupation_Required"]).lower()
        if occ_req != "all" and occupation.lower() not in occ_req:
            continue
        if float(scheme["Min_Land_Acres"]) > 0 and land_size < float(scheme["Min_Land_Acres"]):
            continue
        gender_req = str(scheme["Gender_Eligibility"]).lower()
        if gender_req not in ["all", gender.lower()]:
            continue

        match_score = 70
        if occ_req == "all" or occupation.lower() in occ_req:
            match_score += 15
        caste_req = str(scheme.get("Caste_Eligibility", "all")).lower()
        if caste_req == "all":
            match_score += 5
        elif caste.lower() in caste_req:
            match_score += 15
        if disabled == "yes" and "disability" in str(scheme["Category"]).lower():
            match_score += 20
        if gender == "female" and "girl" in str(scheme["Category"]).lower():
            match_score += 20
        scheme_dict = scheme.to_dict()
        scheme_dict["match_score"] = min(99, match_score)
        recommended.append(scheme_dict)

    recommended.sort(key=lambda x: x["match_score"], reverse=True)
    return recommended[:8]


# ================================================================
# DATABASE
# ================================================================
DB_FILE = "datasets/applications.csv"
DB_HEADERS = ["app_id","name","age","state","income","occupation","fraud_score","fraud_level","scheme_count","status","flag_reason","timestamp"]

def save_application(data, fraud_score, fraud_level, flags, scheme_count):
    is_new = not os.path.exists(DB_FILE)
    app_id = f"APP{int(datetime.now().timestamp()) % 100000:05d}"
    status = "Flagged" if fraud_level == "HIGH" else ("Under Review" if fraud_level == "MEDIUM" else "Approved")
    row = {
        "app_id": app_id, "name": data.get("name",""), "age": data.get("age",""),
        "state": data.get("state",""), "income": data.get("income",""),
        "occupation": data.get("occupation",""), "fraud_score": fraud_score,
        "fraud_level": fraud_level, "scheme_count": scheme_count, "status": status,
        "flag_reason": "; ".join(flags[:2]) if flags else "",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    with open(DB_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DB_HEADERS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return app_id, status


# ================================================================
# ROUTES
# ================================================================
@app.route("/")
def index():
    lang = request.args.get("lang", "en")
    t = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return render_template("index.html", t=t, lang=lang)

@app.route("/schemes")
def schemes_browse():
    lang = request.args.get("lang", "en")
    t = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    category_filter = request.args.get("category", "")
    categories = sorted(schemes_df["Category"].unique().tolist()) if not schemes_df.empty else []
    filtered = schemes_df.copy()
    if category_filter:
        filtered = filtered[filtered["Category"] == category_filter]
    return render_template("schemes.html", t=t, lang=lang,
                           schemes=filtered.to_dict(orient="records"),
                           categories=categories, selected_category=category_filter)

@app.route("/admin")
def admin():
    lang = request.args.get("lang", "en")
    t = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    apps = []
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            apps = list(reversed(list(csv.DictReader(f))))
    total = len(apps)
    flagged = sum(1 for a in apps if a.get("fraud_level") == "HIGH")
    approved = sum(1 for a in apps if a.get("status") == "Approved")
    under_review = sum(1 for a in apps if a.get("status") == "Under Review")
    return render_template("admin.html", t=t, lang=lang, apps=apps,
                           total=total, flagged=flagged, approved=approved, under_review=under_review)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    lang = data.get("lang", "en")
    try:
        age = int(data.get("age", 0))
        income = float(data.get("income", 0))
        land_size = float(data.get("land_size", 0))
        occupation = data.get("occupation", "farmer")
        gender = data.get("gender", "male")
        caste = data.get("caste", "general")
        disabled = data.get("disabled", "no")

        fraud_score, fraud_level, flags, explanation, recommendations = detect_fraud(data)
        schemes = recommend_schemes(age, income, occupation, land_size, gender, caste, disabled)
        app_id, status = save_application(data, fraud_score, fraud_level, flags, len(schemes))

        localized_schemes = []
        for s in schemes:
            name_key = "Scheme_Name_Hi" if lang == "hi" else ("Scheme_Name_Mr" if lang == "mr" else "Scheme_Name")
            desc_key = "Description_Hi" if lang == "hi" else ("Description_Mr" if lang == "mr" else "Description")
            ben_key = "Benefits_Hi" if lang == "hi" else ("Benefits_Mr" if lang == "mr" else "Benefits")
            localized_schemes.append({
                "name": s.get(name_key) or s.get("Scheme_Name",""),
                "ministry": s.get("Ministry",""),
                "category": s.get("Category",""),
                "description": s.get(desc_key) or s.get("Description",""),
                "eligibility": f"Age {s.get('Eligibility_Age_Min')}-{s.get('Eligibility_Age_Max')}, Income ≤ ₹{int(s.get('Max_Income_Allowed',0)):,}",
                "benefits": s.get(ben_key) or s.get("Benefits",""),
                "how_to_apply": f"Visit {s.get('Apply_URL','official portal')} or nearest CSC center",
                "apply_url": s.get("Apply_URL","#"),
                "match_score": s.get("match_score", 70),
            })

        fraud_explanations = {
            "en": explanation,
            "hi": f"AI विश्लेषण में {len(flags)} संदिग्ध पैटर्न मिले। धोखाधड़ी जोखिम: {fraud_score}/100।",
            "mr": f"AI विश्लेषणात {len(flags)} संशयास्पद नमुने। फसवणूक जोखीम: {fraud_score}/100।",
        }

        return jsonify({
            "success": True, "app_id": app_id, "status": status,
            "fraud": {
                "score": fraud_score, "level": fraud_level, "flags": flags,
                "explanation": fraud_explanations.get(lang, explanation),
                "recommendations": recommendations,
            },
            "schemes": localized_schemes, "scheme_count": len(localized_schemes),
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/translations")
def get_translations():
    return jsonify(TRANSLATIONS)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
