from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# ---------------- LOAD MODELS ----------------
diabetes_model = joblib.load("../models/diabetes_model.pkl")

pcos_model = joblib.load("../models/pcos_model.pkl")
pcos_scaler = joblib.load("../models/scaler.pkl")

thyroid_model = joblib.load("../models/thyroid_model.pkl")

# ---------------- THYROID FEATURE LIST ----------------
thyroid_features = [
    'age','sex','on_thyroxine','query_on_thyroxine','on_antithyroid_medication',
    'sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid',
    'query_hyperthyroid','lithium','goitre','tumor','hypopituitary','psych',
    'TSH','TT4','T4U','FTI'
]

# ---------------- HELPER FUNCTION ----------------
def prepare_thyroid_features(data):
    sample = {col: 0 for col in thyroid_features}

    sample['age'] = float(data['age'])
    sample['sex'] = int(data['sex'])
    sample['TSH'] = float(data['TSH'])
    sample['TT4'] = float(data['TT4'])
    sample['T4U'] = float(data['T4U'])
    sample['FTI'] = float(data['FTI'])

    return np.array(list(sample.values())).reshape(1, -1)

# ---------------- HOME ----------------
@app.route("/")
def home():
    return "API is running successfully 🚀"

# ---------------- DIABETES ----------------
@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    data = request.json

    if not data or any(v is None for v in data.values()):
        return jsonify({"error": "Missing input"}), 400

    try:
        features = [
            data["Pregnancies"],
            data["Glucose"],
            data["BloodPressure"],
            data["SkinThickness"],
            data["Insulin"],
            data["BMI"],
            data["DiabetesPedigreeFunction"],
            data["Age"]
        ]

        features = np.array(features).reshape(1, -1)

        prediction = diabetes_model.predict(features)[0]
        probability = diabetes_model.predict_proba(features)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "risk_percentage": round(probability * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- PCOS ----------------
@app.route("/predict_pcos", methods=["POST"])
def predict_pcos():
    data = request.json

    if not data or any(v is None for v in data.values()):
        return jsonify({"error": "Missing input"}), 400

    try:
        features = [
            data["Follicle No. (R)"],
            data["Follicle No. (L)"],
            data["Skin darkening (Y/N)"],
            data["hair growth(Y/N)"],
            data["Weight gain(Y/N)"],
            data["AMH(ng/mL)"],
            data["Cycle(R/I)"],
            data["FSH/LH"],
            data["LH(mIU/mL)"],
            data["Fast food (Y/N)"]
        ]

        features = np.array(features).reshape(1, -1)

        # Apply scaler
        features = pcos_scaler.transform(features)

        prediction = pcos_model.predict(features)[0]
        probability = pcos_model.predict_proba(features)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "risk_percentage": round(probability * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- THYROID ----------------
@app.route("/predict_thyroid", methods=["POST"])
def predict_thyroid():
    data = request.json

    if not data or any(v is None for v in data.values()):
        return jsonify({"error": "Missing input"}), 400

    try:
        features = prepare_thyroid_features(data)

        prediction = thyroid_model.predict(features)[0]
        probability = thyroid_model.predict_proba(features)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "confidence": f"{round(probability * 100, 2)}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)