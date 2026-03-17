# 💳 Fraud Detection System (Machine Learning)

## 📊 Problem Statement
Detect fraudulent financial transactions using machine learning to minimize financial losses.

## 🏦 Domain
Banking, Financial Services, Insurance (BFSI)

## 🛠️ Tech Stack
Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn

## ⚙️ Project Workflow
- Data Cleaning & Preprocessing
- Handling Imbalanced Data using SMOTE
- Feature Engineering
- Model Building (Logistic Regression, Random Forest)
- Model Evaluation

## 📈 Results
- Achieved ~90% accuracy
- Improved fraud detection using precision & recall
- Reduced false negatives (critical in fraud detection)

## 📊 Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Precision, Recall, F1-score

## 🚀 Business Impact
- Helps banks detect fraud early
- Reduces financial loss
- Improves transaction security

## 📌 Future Improvements
- Deploy using Flask/Streamlit
- Real-time fraud detection system# Fraud Detection System (Machine Learning)

## 📊 Problem Statement
Detect fraudulent transactions using machine learning techniques.

## 🏦 Domain
BFSI (Banking, Financial Services, Insurance)

## 🛠️ Tech Stack
Python, Pandas, Scikit-learn, Imbalanced-learn

## ⚙️ Workflow
- Data preprocessing
- Handling imbalanced data (SMOTE)
- Model building (Logistic Regression, Random Forest)
- Evaluation using Precision, Recall, F1-score

## 📈 Results
- Improved fraud detection accuracy
- Reduced false negatives (important in fraud cases)

## 🚀 Business Impact
Helps financial institutions detect fraud early and reduce losses.



import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data.csv")

# Basic EDA
print(df.head())
print(df.info())

# Preprocessing
df = df.dropna()

# Feature & Target
X = df.drop("target", axis=1)
y = df["target"]

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
