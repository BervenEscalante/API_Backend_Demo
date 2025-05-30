import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import joblib
import os

os.makedirs("trained_data", exist_ok=True)

# Load your dataset (adjust path if needed)
df = pd.read_csv('csv/Students_Grading_Dataset.csv')

# Create Pass/Fail label
df['Result'] = df['Grade'].apply(lambda x: 'Pass' if x != 'F' else 'Fail')

# Encode target labels
label_encoder = LabelEncoder()
y_raw = label_encoder.fit_transform(df['Result'])  # Fail=0, Pass=1

selected_features = [
    'Study_Hours_per_Week',
    'Stress_Level (1-10)',
    'Sleep_Hours_per_Night',
    'Participation_Score'
]

X_raw = df[selected_features]

# Balance classes by upsampling minority (Fail)
data = pd.concat([X_raw, pd.Series(y_raw, name='Result')], axis=1)
pass_df = data[data['Result'] == 1]
fail_df = data[data['Result'] == 0]

fail_upsampled = resample(fail_df, replace=True, n_samples=len(pass_df), random_state=42)
balanced_data = pd.concat([pass_df, fail_upsampled]).sample(frac=1, random_state=42)

X = balanced_data[selected_features]
y = balanced_data['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, early_stopping=True, random_state=42))
])

pipeline.fit(X_train, y_train)

# Save artifacts
joblib.dump(pipeline, 'trained_data/pass_fail_model.pkl')
joblib.dump(label_encoder, 'trained_data/pass_fail_encoder.pkl')
joblib.dump(selected_features, 'trained_data/pass_fail_features.pkl')

print("✅ Training complete and model saved.")
