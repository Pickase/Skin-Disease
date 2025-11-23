# Hybrid Skin Disease Diagnosis System (AI + Machine Learning)

This project is a complete end-to-end Skin Disease Diagnostic System that combines:

### **1. NLP-Based Symptom Keyword Screening (Rough Diagnosis)**  
The user enters a natural-language description of their symptoms (e.g., “itchy red patches on elbow, dryness”).  
A custom keyword-matching AI maps this text to likely diseases such as Psoriasis, Eczema, Lichen Planus, etc.

### **2. ML-Based Structured Diagnosis (Final Prediction)**  
For a precise diagnosis, the system uses a trained **XGBoost model** built on the Dermatology dataset.  
The model uses **34 clinical + histopathological features** to identify the exact skin disease category with ~98% test accuracy.

This hybrid approach provides both:
- an **easy-to-use AI assistant**, and  
- a **high-accuracy medical ML model**,  
similar to real-world clinical triage systems.

---

# Project Structure

skin-disease/
│
├── data/
│ ├── raw/
│ │ └── dataset_dermatology.csv
│ └── processed/
│ ├── train.csv
│ └── test.csv
│
├── models/
│ ├── xgb_model.pkl
│ └── label_encoder.pkl
│
├── src/
│ ├── config.py
│ ├── utils.py
│ ├── data_prep.py
│ ├── features.py
│ ├── pipelines.py
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py
│ └── symptom_keywords.py
│
├── app/
│ └── app.py
│
├── requirements.txt
└── README.md


---

# How the System Works

## **🔹 Stage 1 — Rough NLP Screening**
- User enters a symptom description in plain English  
- A custom dictionary of medical keywords maps the text to possible diseases  
- The app returns the **Top 3 predicted conditions** based on keyword match score  
- Helps guide the user before entering numerical symptoms

### Example Input:
> "itchy red patches and dryness on elbows"

### Example Output:
- Psoriasis (3 keyword matches)  
- Eczema (2 matches)  
- Dermatitis (1 match)

---

## **🔹 Stage 2 — Precise ML Diagnosis**
The system uses the processed dataset to train a model on:

- 34 numeric clinical features  
- 1 target column (`class`)

The workflow:

1. Data cleaning  
2. Replace “?” values  
3. Label encoding  
4. Train-test split  
5. SMOTEENN (oversampling + noise cleaning)  
6. Train **XGBoostClassifier with tuned parameters**  
7. Save model + label encoder  
8. Evaluate accuracy  

### Best model performance:
- **Train Accuracy:** ~1.00  
- **Test Accuracy:** ~0.982  
- **Top 3 models:** XGBoost, ANN, Gradient Boosting  
- **Chosen model:** **XGBoost (best balance of speed + accuracy + stability)**

---

# Running the Project

## 1️⃣ Create a virtual environment

python3 -m venv venv
source venv/bin/activate


## 2️⃣ Install dependencies

pip install -r requirements.txt


## 3️⃣ Prepare the data

python -m src.data_prep


## 4️⃣ Train the model

python -m src.train


## 5️⃣ Evaluate the model

python -m src.evaluate


## 6️⃣ Run the Streamlit App

streamlit run app/app.py


---

# Features
 
### XGBoost tuned model  
### SMOTEENN oversampling  
### Clean modular Python package  
### Reproducible ML pipeline  
### Streamlit UI for diagnosis  
### Ready for deployment & GitHub  

---

# Technologies Used

- **Python 3.12**  
- **pandas**  
- **numpy**  
- **scikit-learn**  
- **XGBoost**  
- **imblearn (SMOTEENN)**  
- **joblib**  
- **Streamlit**  


---

# requirements.txt

pandas
numpy
scikit-learn
xgboost
imblearn
joblib
streamlit


---

# 👤 Author  
Pranav Joshi  
ML Skin Disease Diagnostic System  

---

