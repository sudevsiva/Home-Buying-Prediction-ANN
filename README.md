# Home-Buying-Prediction-ANN

# Home Affordability Prediction using Machine Learning

This project predicts whether a person can afford to buy a home based on
income, savings, and expenses using an Artificial Neural Network (ANN).

## Features
- Data preprocessing pipeline
- Feature engineering
- ANN classification model
- Model persistence using Joblib
- Prediction pipeline

## Technologies
Python, Pandas, Scikit-Learn, TensorFlow

## Model Architecture
Input Layer → Dense(64) → Dense(32) → Sigmoid Output

## Run the Project

Install dependencies

pip install -r requirements.txt

Train model

python src/train_model.py

Run prediction

python src/predict.py
