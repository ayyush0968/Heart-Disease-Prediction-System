import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# UI
st.title("Heart Disease Prediction (Advanced AI System)")
st.write("Includes multiple ML models, evaluation metrics & risk prediction")

# Load Data
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal','target']

    data = pd.read_csv(url, names=columns)
    data = data.replace('?', np.nan).dropna().astype(float)
    data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

    return data

data = load_data()

# Split
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
trained_models = {}

# Train & Evaluate
st.subheader("Model Evaluation")

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    results[name] = acc
    trained_models[name] = model

    st.write(f"{name}")
    st.write(f"Accuracy: {acc*100:.2f}%")
    st.write(f"ROC-AUC: {roc:.2f}")

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# Best Model
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

st.write(f"Best Model Selected: {best_model_name}")

# Feature Importance
st.subheader("Feature Importance")

if best_model_name in ["Random Forest", "Decision Tree"]:
    importance = best_model.feature_importances_
else:
    importance = np.abs(best_model.coef_[0])

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df)

# User Input
st.subheader("Enter Patient Details")

age = st.slider("Age", 1, 120, 63)
sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
cp = st.slider("Chest Pain Type", 0, 3, 3)
trestbps = st.slider("Blood Pressure", 80, 200, 145)
chol = st.slider("Cholesterol", 100, 600, 233)
fbs = st.selectbox("Fasting Sugar >120", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
restecg = st.slider("ECG", 0, 2, 0)
thalach = st.slider("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Angina", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
oldpeak = st.slider("ST Depression", 0.0, 10.0, 2.3)
slope = st.slider("Slope", 0, 2, 0)
ca = st.slider("Major Vessels", 0, 3, 0)
thal = st.slider("Thalassemia", 1, 3, 1)

# Prediction
if st.button("Predict Risk"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    input_scaled = scaler.transform(input_data)

    prediction = best_model.predict(input_scaled)[0]
    probability = best_model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    st.write(f"Risk Probability: {probability*100:.2f}%")

    if prediction == 1:
        st.write("High Risk of Heart Disease")
    else:
        st.write("Low Risk of Heart Disease")
