# -*- coding: utf-8 -*-
"""
Training script for advertising sales prediction model
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_data_from_s3, setup_s3_client

def train_model():
    """
    Train the advertising sales prediction model
    """
    print("Starting model training...")

    # Setup S3 client
    s3 = setup_s3_client()

    # Load data
    print("Loading data from S3...")
    advertising_data = load_data_from_s3(s3, 'teamolducsyersproject', 'processed/advertising.csv')

    if advertising_data is None:
        print("Failed to load data. Exiting...")
        return None

    print("Data loaded successfully:")
    print(advertising_data.head())
    print(f"Data shape: {advertising_data.shape}")

    # Prepare features and target
    X = advertising_data[['TV', 'Radio', 'Newspaper']]  # Features
    y = advertising_data['Sales']  # Target variable

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Train model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Save model
    model_filename = 'advertising_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved as '{model_filename}'")

    # Save feature names for later use
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'feature_names.joblib')
    print(f"Feature names saved: {feature_names}")

    return model, mse, r2

if __name__ == "__main__":
    train_model() 
