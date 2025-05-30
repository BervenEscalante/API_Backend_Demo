from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model stuff (optional now but keep if you want)
pipeline = joblib.load('trained_data/pass_fail_model.pkl')
label_encoder = joblib.load('trained_data/pass_fail_encoder.pkl')
feature_cols = joblib.load('trained_data/pass_fail_features.pkl')

def rule_based_predict(data):
    # Example criteria to FAIL:
    # - Study hours < 5
    # - Stress level > 7
    # - Sleep hours < 5
    # - Participation score < 50
    if (data['Study_Hours_per_Week'] < 5 or
        data['Stress_Level (1-10)'] > 7 or
        data['Sleep_Hours_per_Night'] < 5 or
        data['Participation_Score'] < 50):
        return 'Fail'
    else:
        return 'Pass'

@app.route('/')
def index():
    return "✅ Flask API running with rule-based pass/fail prediction."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Check required keys exist
        missing = [f for f in feature_cols if f not in data]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Use rule-based criteria to determine result
        result = rule_based_predict(data)

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
