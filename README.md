# Skin Disease Classification System (Keyword Filter + XGBoost Model)

This project is a complete machine learning pipeline for predicting skin diseases using the Dermatology dataset.

It includes:

1. **Keyword-Based Symptom Filter (basic string matching)**  
   Users describe symptoms in text (e.g., “itchy red patches”).  
   The system checks for matching words from predefined symptom lists to suggest possible diseases.  
   This is NOT NLP or AI — just simple keyword matching for rough guidance.

2. **XGBoost Machine Learning Model (final prediction)**  
   The main ML model uses 34 clinical features to classify the skin disease with ~98% accuracy.

---

## 📂 Project Structure

```
skin-disease/
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
│   ├── predict.py
│   └── symptom_keywords.py
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## How It Works

### 1️⃣ Keyword-Based Screening  
- Uses simple string matching  
- Suggests top disease categories based on keywords  
- Helps guide the user before entering detailed numeric features  

### 2️⃣ ML Classification (XGBoost)
Dataset: Dermatology dataset with 34 features + 1 target class.

Pipeline steps:
- Replace "?" values  
- Convert columns to numeric  
- Remove duplicates  
- Label Encoding  
- Train-test split  
- SMOTEENN oversampling  
- Train tuned XGBoost model  

Performance:  
- **Train Accuracy:** ~100%  
- **Test Accuracy:** ~98.2%  

---

## Running the Project

### Create virtual environment  
```
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies  
```
pip install -r requirements.txt
```

### Prepare dataset  
```
python -m src.data_prep
```

### Train model  
```
python -m src.train
```

### Evaluate model  
```
python -m src.evaluate
```

### Run Streamlit App  
```
streamlit run app/app.py
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
imblearn
joblib
streamlit
```

---

## Author  
Pranav Joshi  
Skin Disease Classification — Machine Learning Project
