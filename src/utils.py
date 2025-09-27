# -*- coding: utf-8 -*-
"""
Utility functions for advertising sales prediction model
"""

import boto3
import pandas as pd
import numpy as np
from io import StringIO
import logging

def setup_s3_client():
    """
    Setup and return S3 client
    
    Returns:
        boto3.client: S3 client
    """
    try:
        s3 = boto3.client('s3')
        return s3
    except Exception as e:
        print(f"Error setting up S3 client: {e}")
        print("Make sure your AWS credentials are configured properly.")
        return None

def load_data_from_s3(s3_client, bucket_name, key):
    """
    Load CSV data from S3 bucket
    
    Args:
        s3_client: boto3 S3 client
        bucket_name (str): S3 bucket name
        key (str): S3 object key (file path)
    
    Returns:
        pd.DataFrame: Loaded data or None if error
    """
    try:
        print(f"Loading data from s3://{bucket_name}/{key}")
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        data = pd.read_csv(response['Body'])
        print(f"Successfully loaded {len(data)} rows and {len(data.columns)} columns")
        return data
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        return None

def validate_data_columns(data, required_columns):
    """
    Validate that data contains required columns
    
    Args:
        data (pd.DataFrame): Input data
        required_columns (list): List of required column names
    
    Returns:
        bool: True if all required columns present, False otherwise
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(data.columns)}")
        return False
    
    return True

def validate_input_data(input_array):
    """
    Validate input data for predictions
    
    Args:
        input_array (np.array): Input data array
    
    Returns:
        bool: True if valid, False otherwise
    """
    # Check for NaN values
    if np.isnan(input_array).any():
        print("Error: Input data contains NaN values")
        return False
    
    # Check for negative values (assuming advertising spend can't be negative)
    if (input_array < 0).any():
        print("Warning: Input data contains negative values")
        print("Advertising spend values should typically be non-negative")
    
    return True

def print_data_summary(data):
    """
    Print summary statistics of the data
    
    Args:
        data (pd.DataFrame): Input data
    """
    print("\n" + "="*40)
    print("DATA SUMMARY")
    print("="*40)
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nBasic statistics:")
    print(data.describe())
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found.")

def save_predictions_to_s3(s3_client, predictions_df, bucket_name, key):
    """
    Save predictions DataFrame to S3
    
    Args:
        s3_client: boto3 S3 client
        predictions_df (pd.DataFrame): DataFrame with predictions
        bucket_name (str): S3 bucket name
        key (str): S3 object key (file path)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert DataFrame to CSV string
        csv_buffer = StringIO()
        predictions_df.to_csv(csv_buffer, index=False)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        print(f"Predictions saved to s3://{bucket_name}/{key}")
        return True
        
    except Exception as e:
        print(f"Error saving predictions to S3: {e}")
        return False

def create_sample_data(n_samples=10):
    """
    Create sample data for testing
    
    Args:
        n_samples (int): Number of samples to create
    
    Returns:
        pd.DataFrame: Sample data
    """
    np.random.seed(42)
    
    # Generate random data within reasonable ranges
    tv_spend = np.random.uniform(0, 300, n_samples)
    radio_spend = np.random.uniform(0, 50, n_samples)
    newspaper_spend = np.random.uniform(0, 100, n_samples)
    
    sample_data = pd.DataFrame({
        'TV': tv_spend,
        'Radio': radio_spend,
        'Newspaper': newspaper_spend
    })
    
    print(f"Created sample data with {n_samples} rows")
    return sample_data

def setup_logging(log_level=logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('advertising_ml.log'),
            logging.StreamHandler()
        ]
    )

def calculate_prediction_intervals(model, X, confidence_level=0.95):
    """
    Calculate prediction intervals (simplified version)
    Note: This is a basic implementation. For more accurate intervals,
    consider using statistical methods specific to your model.
    
    Args:
        model: Trained model
        X: Input features
        confidence_level (float): Confidence level for intervals
    
    Returns:
        tuple: (predictions, lower_bounds, upper_bounds)
    """
    predictions = model.predict(X)
    
    # Simple estimation using residuals (this is simplified)
    # In practice, you'd want to use proper statistical methods
    residual_std = 2.0  # This should be calculated from training residuals
    margin = 1.96 * residual_std  # For 95% confidence interval
    
    lower_bounds = predictions - margin
    upper_bounds = predictions + margin
    
    return predictions, lower_bounds, upper_bounds 
 
