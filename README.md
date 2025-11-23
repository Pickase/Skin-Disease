# Skin Disease Classification System

This project implements a complete machine learning pipeline for predicting skin diseases using the Dermatology dataset. It combines a simple keyword-based symptom filter with a trained XGBoost model for final classification.

A live demo of the project is available here:  
https://skin-disease-1.streamlit.app/

---

## Project Structure

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

## How the System Works

### 1. Keyword-Based Screening
Users can describe their symptoms in plain English.  
The system performs keyword matching to suggest possible disease categories.  
This step acts as a rough screening tool to guide the user before entering detailed clinical features.

### 2. Machine Learning Classification (XGBoost)
The primary model uses 34 clinical and histopathological features to determine the exact disease class.

Pipeline steps:
- Replace invalid entries such as “?”
- Convert features to numeric format
- Remove duplicates
- Label encode the target variable
- Stratified train-test split
- Apply SMOTEENN oversampling to balance classes
- Train a tuned XGBoostClassifier

Performance from experimentation:
- Train Accuracy: ~100%
- Test Accuracy: ~98.2%

---

## Running the Project

### 1. Create and activate a virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Prepare the dataset
```
python -m src.data_prep
```

### 4. Train the model
```
python -m src.train
```

### 5. Evaluate the model
```
python -m src.evaluate
```

### 6. Launch the Streamlit Application
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

## Project Link

Streamlit App:  
https://skin-disease-1.streamlit.app/

---

## Author
Pranav Joshi  
Skin Disease Classification – Machine Learning Project
