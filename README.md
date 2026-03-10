# GovShield - Government Scheme Awareness & Fraud Detection Platform

## Features
- **Multilingual**: English, Hindi, Marathi (full UI + scheme descriptions)
- **ML Fraud Detection**: RandomForest model (90% accuracy) with rule-based hybrid
- **Scheme Recommendation**: KNN-based matching across 15 government schemes
- **Admin Dashboard**: Real-time application log with fraud stats
- **Persistent Storage**: CSV-based application database

## Setup & Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train ML models (run once)
python train_models.py

# Start the app
python app.py
# → Open http://localhost:5000
```

## Project Structure
```
govshield/
├── app.py               # Flask backend (routes, ML logic, translations)
├── train_models.py      # Model training script
├── requirements.txt
├── datasets/
│   ├── fraud.csv        # Training data (50+ applicants with labels)
│   ├── schemes.csv      # 15 government schemes with multilingual info
│   └── applications.csv # Auto-created — stores submissions
├── models/
│   ├── fraud_detector.pkl
│   ├── fraud_encoders.pkl
│   ├── fraud_features.pkl
│   ├── scheme_recommender.pkl
│   ├── scheme_cat_encoder.pkl
│   └── scheme_occ_encoder.pkl
└── templates/
    ├── base.html        # Shared layout (header, nav, footer)
    ├── index.html       # Application form + live results
    ├── schemes.html     # Browse all schemes
    └── admin.html       # Admin dashboard
```

## Dataset Schema
### fraud.csv
| Column | Type | Description |
|--------|------|-------------|
| Person_ID | str | Unique ID |
| Age | int | Applicant age |
| Gender | str | Male/Female |
| State/District | str | Location |
| Profession | str | Occupation |
| Annual_Income | int | ₹ per year |
| Family_Size | int | Number of members |
| Bank_Account_Linked | 0/1 | DBT readiness |
| Previous_Scheme_Benefits | int | Total ₹ received |
| Applications_Submitted | int | Count of prior applications |
| Fraud_Label | 0/1 | Ground truth |

### schemes.csv
| Column | Description |
|--------|-------------|
| Scheme_Name / _Hi / _Mr | Name in 3 languages |
| Description / _Hi / _Mr | Description in 3 languages |
| Benefits / _Hi / _Mr | Benefits in 3 languages |
| Eligibility_Age_Min/Max | Age range |
| Max_Income_Allowed | Income ceiling ₹ |
| Min_Land_Acres | Minimum land required |
| Occupation_Required | all / farmer / laborer / etc |
| Gender_Eligibility | all / male / female |
| Apply_URL | Official application portal |

## Fraud Detection Logic
Hybrid Rule-based + ML approach:
- Benefit/income ratio > 80% → HIGH risk flag
- Applications > 7 → HIGH risk flag  
- No bank account linked → +15 points
- Unverified documents → +20 points
- ML RandomForest prediction blended at 40% weight
