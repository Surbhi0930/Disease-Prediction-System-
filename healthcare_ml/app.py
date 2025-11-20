# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap

st.title("Disease Prediction Prototype (Demo only)")

# Load model
model = joblib.load("disease_model.pkl")

# Load feature names from training (assumes you saved columns somewhere)
# For simplicity, assume features are in features.txt (one per line)
with open("features.txt") as f:
    features = [line.strip() for line in f]

st.sidebar.header("Enter patient info / symptoms")
vals = {}
for feat in features:
    # if symptom: binary checkbox; else numeric input
    if feat.lower().startswith("symptom_") or feat.lower().startswith("has_"):
        vals[feat] = 1 if st.sidebar.checkbox(feat.replace("_"," ")) else 0
    else:
        vals[feat] = st.sidebar.number_input(feat, value=0.0)

input_df = pd.DataFrame([vals])

if st.button("Predict"):
    proba = model.predict_proba(input_df)[0]
    classes = model.named_steps['clf'].classes_
    top_idx = np.argsort(proba)[::-1][:5]
    st.write("Top predictions:")
    for i in top_idx:
        st.write(f"{classes[i]} — probability: {proba[i]:.3f}")
    # SHAP explanation (small)
    explainer = shap.TreeExplainer(model.named_steps['clf'])
    X_trans = model.named_steps['scaler'].transform(input_df)
    shap_vals = explainer.shap_values(X_trans)
    st.write("Feature influence (SHAP) for top class:")
    top_class = top_idx[0]
    shap.force_plot(explainer.expected_value[top_class], shap_vals[top_class], input_df, matplotlib=True, show=False)
    st.pyplot(bbox_inches='tight')
