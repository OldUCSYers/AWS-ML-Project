# -*- coding: utf-8 -*-
"""
Prediction script for advertising sales prediction model
"""

import pandas as pd
import joblib
import numpy as np
from utils import validate_input_data

def load_model(model_path='advertising_model.joblib'):
    """
    Load the trained model
    
    Args:
        model_path (str): Path to the saved model file
    
    Returns:
        model: Loaded scikit-learn model
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run train.py first to create the model.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_feature_names(feature_path='feature_names.joblib'):
    """
    Load the feature names used during training
    
    Args:
        feature_path (str): Path to the saved feature names file
    
    Returns:
        list: List of feature names
    """
    try:
        feature_names = joblib.load(feature_path)
        return feature_names
    except FileNotFoundError:
        print(f"Warning: Feature names file '{feature_path}' not found.")
        return ['TV', 'Radio', 'Newspaper']  # Default feature names

def predict_single(model, tv, radio, newspaper):
    """
    Make a single prediction
    
    Args:
        model: Trained model
        tv (float): TV advertising spend
        radio (float): Radio advertising spend
        newspaper (float): Newspaper advertising spend
    
    Returns:
        float: Predicted sales
    """
    # Create input array
    input_data = np.array([[tv, radio, newspaper]])
    
    # Validate input
    if not validate_input_data(input_data):
        return None
    
    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]

def predict_batch(model, input_data):
    """
    Make predictions for multiple samples
    
    Args:
        model: Trained model
        input_data (pd.DataFrame or np.array): Input data with features
    
    Returns:
        np.array: Array of predictions
    """
    # Convert to DataFrame if numpy array
    if isinstance(input_data, np.ndarray):
        feature_names = load_feature_names()
        input_data = pd.DataFrame(input_data, columns=feature_names)
    
    # Validate input
    expected_features = ['TV', 'Radio', 'Newspaper']
    if not all(feature in input_data.columns for feature in expected_features):
        print(f"Error: Input data must contain columns: {expected_features}")
        return None
    
    # Select and order features correctly
    X = input_data[expected_features]
    
    # Validate input
    if not validate_input_data(X.values):
        return None
    
    # Make predictions
    predictions = model.predict(X)
    return predictions

def predict_from_csv(model, csv_path):
    """
    Make predictions from a CSV file
    
    Args:
        model: Trained model
        csv_path (str): Path to CSV file with input data
    
    Returns:
        pd.DataFrame: DataFrame with input data and predictions
    """
    try:
        # Load data
        data = pd.read_csv(csv_path)
        print(f"Loaded {len(data)} samples from {csv_path}")
        
        # Make predictions
        predictions = predict_batch(model, data)
        
        if predictions is not None:
            # Add predictions to dataframe
            data['Predicted_Sales'] = predictions
            return data
        else:
            return None
            
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def main():
    """
    Main function for interactive predictions
    """
    # Load model
    model = load_model()
    if model is None:
        return
    
    print("\n" + "="*50)
    print("ADVERTISING SALES PREDICTION")
    print("="*50)
    
    while True:
        print("\nChoose an option:")
        print("1. Single prediction")
        print("2. Batch prediction from CSV")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            try:
                print("\nEnter advertising spend values:")
                tv = float(input("TV advertising spend: "))
                radio = float(input("Radio advertising spend: "))
                newspaper = float(input("Newspaper advertising spend: "))
                
                prediction = predict_single(model, tv, radio, newspaper)
                if prediction is not None:
                    print(f"\nPredicted Sales: {prediction:.2f}")
                
            except ValueError:
                print("Error: Please enter valid numeric values.")
                
        elif choice == '2':
            csv_path = input("Enter path to CSV file: ").strip()
            result = predict_from_csv(model, csv_path)
            
            if result is not None:
                print("\nPredictions:")
                print(result)
                
                # Ask if user wants to save results
                save_choice = input("\nSave results to CSV? (y/n): ").strip().lower()
                if save_choice == 'y':
                    output_path = input("Enter output file path: ").strip()
                    result.to_csv(output_path, index=False)
                    print(f"Results saved to {output_path}")
                    
        elif choice == '3':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 
