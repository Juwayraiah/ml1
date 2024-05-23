#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pickle
import streamlit as st

# Load the dataset
df = pd.read_csv(r"E:\Junior year uni\semester 2\ML & DM\Juwayraiah\heart disease dataset project.csv")

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Handle outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Feature Engineering and Scaling
df['cp'] = df['cp'] ** 2
scaler = StandardScaler()
df[['age', 'trestbps', 'chol', 'thalach']] = scaler.fit_transform(df[['age', 'trestbps', 'chol', 'thalach']])
df = pd.get_dummies(df, columns=['sex', 'cp', 'restecg', 'slope', 'thal'], drop_first=True)

# Split the dataset into training and testing sets
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and train a model
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# Evaluate the model's performance
y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Test Accuracy:", accuracy_rf * 100, "%")

# Train and evaluate other models
model_svm = SVC(kernel='linear')
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Test Accuracy:", accuracy_svm * 100, "%")

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Test Accuracy:", accuracy_lr * 100, "%")

# Save the best model
pickle.dump(model_rf, open("heart_disease_model.sav", "wb"))

# Streamlit app
st.title("Heart Disease Prediction")
st.info("Please fill out the sections below")

# Load the model
model = pickle.load(open("heart_disease_model.sav", 'rb'))

# Input fields
age = st.text_input('Age')
sex = st.selectbox('Sex', options=[0, 1])
cp = st.text_input('Chest Pain Type')
trestbps = st.text_input('Resting Blood Pressure')
chol = st.text_input('Serum Cholestoral in mg/dl')
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
restecg = st.text_input('Resting Electrocardiographic Results')
thalach = st.text_input('Maximum Heart Rate Achieved')
exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
oldpeak = st.text_input('ST Depression Induced by Exercise')
slope = st.text_input('Slope of the Peak Exercise ST Segment')
ca = st.text_input('Number of Major Vessels Colored by Flourosopy')
thal = st.text_input('Thalassemia')

# Make prediction
if st.button('Predict'):
    user_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    user_data = scaler.transform(user_data)  # Ensure the same scaling as training data
    prediction = model.predict(user_data)
    st.write('Prediction:', 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease')
