import streamlit as st
import pandas as pd

import sys
import os

# Add project root to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.predict import predict_single
from src.symptom_keywords import rough_predict

st.title("Skin Disease Diagnostic System")

st.subheader("Stage 1: Rough Disease Screening (Based on Your Description)")

user_symptoms = st.text_area(
    "Describe your skin problem in your own words (e.g., 'itchy red patches on elbow, dryness, scaling')",
    height=150
)

rough_results = None

if st.button("Get Rough Diagnosis"):
    if not user_symptoms.strip():
        st.warning("Please enter a description.")
    else:
        rough_results = rough_predict(user_symptoms)
        st.info("Top possible matches based on your description:")
        for disease, score in rough_results:
            st.write(f"**{disease}** (keyword matches: {score})")
