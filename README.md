
# Skin Disease Classification System (Keyword Filter + XGBoost Model)

This project is a complete machine learning system for predicting skin diseases using the Dermatology dataset.  
It includes:

### 1. Keyword-Based Symptom Filter (Rough Screening)
Users can type a short symptom description (e.g., “itchy red patches on knee”), and the system performs a simple keyword match against predefined lists.  
This step provides a **rough idea** of which disease categories are likely based on the words used.  
(Important: This is *not* NLP or AI — just basic keyword matching.)

### 2. XGBoost Machine Learning Model (Final Prediction)
Users then enter 34 clinical features, and the trained XGBoost model predicts the exact disease class.  
This model was trained after preprocessing steps including cleaning, label encoding, train-test split, and SMOTEENN oversampling.

---

# Project Structure

skin-disease/
│
├── data/
│   ├── raw/
│   │   └── dataset_dermatology.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
│
├── models/
│   ├── xgb_model.pkl
│   └── label_encoder.pkl
│
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── data_prep.py
│   ├── features.py
│   ├── pipelines.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│   └── symptom_keywords.py
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md

---

# How the System Works

## **1️⃣ Keyword-Based Filter (Simple Matching)**  
A dictionary of symptom keywords is used to match user text against diseases.  
This is **not** machine learning / NLP — just straightforward string matching.

It helps users:
- understand possible directions  
- identify common-sense connections  
- prepare for the detailed ML input step  

Example Output:
- Psoriasis (3 keyword matches)  
- Eczema (2 matches)  
- Dermatitis (1 match)

This step is optional and only for guidance.

---

## **2️⃣ Machine Learning Prediction (XGBoost)**

The dataset includes:
- 34 numeric features (clinical + histopathological)
- “class” as the target label

### Preprocessing steps:
- Replace "?" values  
- Convert all features to numeric  
- Remove duplicates  
- Label encoding  
- Train-test split  
- SMOTEENN oversampling (to balance classes)

### Model used:
`XGBClassifier` with tuned parameters.

### Achieved Accuracy (From Notebook):
- **Training accuracy:** ~1.00  
- **Test accuracy:** ~0.982  

This model was chosen because it provided the best combination of:
- High accuracy  
- Stability  
- Fast performance  
- Good generalization  

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



## 6️⃣ Run Streamlit App



streamlit run app/app.py



---

# Requirements


pandas
numpy
scikit-learn
xgboost
imblearn
joblib
streamlit



---

# Author  
Pranav Joshi  
Skin Disease Classification — Machine Learning Project

