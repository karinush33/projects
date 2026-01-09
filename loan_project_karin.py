"""
Loan Approval Prediction Model Training
Author: Karin Rosental
Date: 9.1.26
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

# --- Step 1: Load the data ---
df = pd.read_csv('train.csv')

# --- Step 2: Select Features and Target ---
# Choosing 6 key features as required by the project guidelines

features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
X = df[features]
y = df['Loan_Status']

# --- Step 3: Data Cleaning ---
# Drop rows with missing values (NaN) to ensure a clean dataset for training

X = X.dropna()
y = y[X.index]

# --- Step 4: Target Encoding ---
# Convert labels 'Y' and 'N' to numerical values 1 and 0

y = (y == 'Y').astype(int)

# --- Step 5: Train-Test Split ---
# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 6: Build the Pipeline ---
# Define the pipeline steps: Scaling the data followed by the SVM classifier
model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", probability=True))
])

# --- Step 7: Model Training ---
# Fit the pipeline model on the training data
model_pipeline.fit(X_train, y_train)

# --- Step 8: Save the Model ---
# Export the trained pipeline to a file for use in the Streamlit app
joblib.dump(model_pipeline, 'loan_model.joblib')

print("The model has been trained and saved successfully!")
