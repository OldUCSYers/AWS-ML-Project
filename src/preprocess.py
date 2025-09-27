# -*- coding: utf-8 -*-
"""
Data preprocessing pipeline for advertising sales prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvertisingDataPreprocessor:
    """
    Comprehensive data preprocessing class for advertising sales data
    """
    
    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.preprocessor = None
        self.feature_names = None
        self.outlier_bounds = {}
        self.original_columns = None
        
    def load_and_explore_data(self, data):
        """
        Initial data exploration and summary
        
        Args:
            data (pd.DataFrame): Raw advertising data
            
        Returns:
            pd.DataFrame: Data with basic info
        """
        print("="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Basic info
        print("\nData types:")
        print(data.dtypes)
        
        # Missing values
        print("\nMissing values:")
        missing_counts = data.isnull().sum()
        missing_percent = (missing_counts / len(data)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_counts,
            'Missing Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(data.describe())
        
        return data
    
    def handle_missing_values(self, data, strategy='mean'):
        """
        Handle missing values in the dataset
        
        Args:
            data (pd.DataFrame): Input data
            strategy (str): Imputation strategy ('mean', 'median', 'mode', 'knn')
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        print("\n" + "="*30)
        print("HANDLING MISSING VALUES")
        print("="*30)
        
        if data.isnull().sum().sum() == 0:
            print("No missing values found!")
            return data
        
        # Separate numerical and categorical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        data_processed = data.copy()
        
        # Handle numerical missing values
        if numerical_cols:
            if strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                data_processed[numerical_cols] = imputer.fit_transform(data_processed[numerical_cols])
            else:
                imputer = SimpleImputer(strategy=strategy)
                data_processed[numerical_cols] = imputer.fit_transform(data_processed[numerical_cols])
            
            self.imputer = imputer
            print(f"Numerical missing values handled using {strategy} imputation")
        
        # Handle categorical missing values
        if categorical_cols:
            for col in categorical_cols:
                if data_processed[col].isnull().sum() > 0:
                    mode_value = data_processed[col].mode()[0] if not data_processed[col].mode().empty else 'Unknown'
                    data_processed[col].fillna(mode_value, inplace=True)
            print("Categorical missing values handled using mode imputation")
        
        return data_processed
    
    def detect_and_handle_outliers(self, data, columns=None, method='iqr', action='cap'):
        """
        Detect and handle outliers in numerical columns
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to check for outliers (default: all numerical)
            method (str): Detection method ('iqr', 'zscore', 'modified_zscore')
            action (str): Action to take ('remove', 'cap', 'log_transform')
            
        Returns:
            pd.DataFrame: Data with outliers handled
        """
        print("\n" + "="*30)
        print("OUTLIER DETECTION & HANDLING")
        print("="*30)
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        data_processed = data.copy()
        outliers_info = {}
        
        for col in columns:
            if col not in data_processed.columns:
                continue
                
            original_count = len(data_processed)
            
            if method == 'iqr':
                Q1 = data_processed[col].quantile(0.25)
                Q3 = data_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (data_processed[col] < lower_bound) | (data_processed[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data_processed[col]))
                outlier_mask = z_scores > 3
                lower_bound = data_processed[col].mean() - 3 * data_processed[col].std()
                upper_bound = data_processed[col].mean() + 3 * data_processed[col].std()
                
            elif method == 'modified_zscore':
                median = data_processed[col].median()
                mad = np.median(np.abs(data_processed[col] - median))
                modified_z_scores = 0.6745 * (data_processed[col] - median) / mad
                outlier_mask = np.abs(modified_z_scores) > 3.5
                lower_bound = median - 3.5 * mad / 0.6745
                upper_bound = median + 3.5 * mad / 0.6745
            
            outlier_count = outlier_mask.sum()
            outliers_info[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / original_count) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            if outlier_count > 0:
                if action == 'remove':
                    data_processed = data_processed[~outlier_mask]
                elif action == 'cap':
                    data_processed.loc[data_processed[col] < lower_bound, col] = lower_bound
                    data_processed.loc[data_processed[col] > upper_bound, col] = upper_bound
                elif action == 'log_transform':
                    # Apply log transformation (add 1 to handle zeros)
                    data_processed[col] = np.log1p(data_processed[col])
        
        # Store outlier bounds for future use
        self.outlier_bounds = outliers_info
        
        print("Outliers detected:")
        for col, info in outliers_info.items():
            print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")
        
        return data_processed
    
    def feature_engineering(self, data):
        """
        Create new features from existing ones
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        print("\n" + "="*30)
        print("FEATURE ENGINEERING")
        print("="*30)
        
        data_processed = data.copy()
        
        # For advertising dataset, create interaction and derived features
        if all(col in data_processed.columns for col in ['TV', 'Radio', 'Newspaper']):
            # Total advertising spend
            data_processed['Total_Advertising'] = (
                data_processed['TV'] + 
                data_processed['Radio'] + 
                data_processed['Newspaper']
            )
            
            # Advertising ratios
            data_processed['TV_Ratio'] = data_processed['TV'] / (data_processed['Total_Advertising'] + 1e-8)
            data_processed['Radio_Ratio'] = data_processed['Radio'] / (data_processed['Total_Advertising'] + 1e-8)
            data_processed['Newspaper_Ratio'] = data_processed['Newspaper'] / (data_processed['Total_Advertising'] + 1e-8)
            
            # Interaction features
            data_processed['TV_Radio_Interaction'] = data_processed['TV'] * data_processed['Radio']
            data_processed['TV_Newspaper_Interaction'] = data_processed['TV'] * data_processed['Newspaper']
            data_processed['Radio_Newspaper_Interaction'] = data_processed['Radio'] * data_processed['Newspaper']
            
            # Polynomial features (squared terms)
            data_processed['TV_Squared'] = data_processed['TV'] ** 2
            data_processed['Radio_Squared'] = data_processed['Radio'] ** 2
            data_processed['Newspaper_Squared'] = data_processed['Newspaper'] ** 2
            
            # Advertising efficiency (if Sales column exists)
            if 'Sales' in data_processed.columns:
                data_processed['TV_Efficiency'] = data_processed['Sales'] / (data_processed['TV'] + 1e-8)
                data_processed['Radio_Efficiency'] = data_processed['Sales'] / (data_processed['Radio'] + 1e-8)
                data_processed['Newspaper_Efficiency'] = data_processed['Sales'] / (data_processed['Newspaper'] + 1e-8)
                data_processed['Overall_Efficiency'] = data_processed['Sales'] / (data_processed['Total_Advertising'] + 1e-8)
            
            print("Created advertising-specific features:")
            print("- Total_Advertising, TV/Radio/Newspaper_Ratio")
            print("- Interaction features (TV_Radio, TV_Newspaper, Radio_Newspaper)")
            print("- Polynomial features (squared terms)")
            if 'Sales' in data_processed.columns:
                print("- Efficiency metrics")
        
        return data_processed
    
    def scale_features(self, data, scaling_method='standard', exclude_columns=None):
        """
        Scale numerical features
        
        Args:
            data (pd.DataFrame): Input data
            scaling_method (str): 'standard', 'minmax', 'robust'
            exclude_columns (list): Columns to exclude from scaling
            
        Returns:
            pd.DataFrame: Data with scaled features
        """
        print("\n" + "="*30)
        print("FEATURE SCALING")
        print("="*30)
        
        if exclude_columns is None:
            exclude_columns = []
        
        # Add target column to exclusions if it exists
        if 'Sales' in data.columns:
            exclude_columns.append('Sales')
        
        # Get numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [col for col in numerical_cols if col not in exclude_columns]
        
        if not cols_to_scale:
            print("No columns to scale!")
            return data
        
        data_processed = data.copy()
        
        # Choose scaler
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaling method. Choose 'standard', 'minmax', or 'robust'")
        
        # Fit and transform
        data_processed[cols_to_scale] = scaler.fit_transform(data_processed[cols_to_scale])
        
        self.scaler = scaler
        print(f"Applied {scaling_method} scaling to {len(cols_to_scale)} columns")
        
        return data_processed
    
    def select_features(self, data, target_column='Sales', k=10, method='f_regression'):
        """
        Select best features using statistical methods
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column name
            k (int): Number of features to select
            method (str): Selection method
            
        Returns:
            pd.DataFrame: Data with selected features
        """
        print("\n" + "="*30)
        print("FEATURE SELECTION")
        print("="*30)
        
        if target_column not in data.columns:
            print(f"Target column '{target_column}' not found. Skipping feature selection.")
            return data
        
        # Prepare features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Select only numerical features for f_regression
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numerical = X[numerical_features]
        
        if len(numerical_features) == 0:
            print("No numerical features found for selection!")
            return data
        
        # Adjust k if necessary
        k = min(k, len(numerical_features))
        
        # Feature selection
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        else:
            raise ValueError("Invalid selection method")
        
        X_selected = selector.fit_transform(X_numerical, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = [numerical_features[i] for i in selected_indices]
        
        # Create output dataframe
        data_selected = pd.DataFrame(X_selected, columns=selected_features)
        data_selected[target_column] = y.values
        
        self.feature_selector = selector
        self.feature_names = selected_features
        
        print(f"Selected {len(selected_features)} best features:")
        feature_scores = selector.scores_[selected_indices]
        for feature, score in zip(selected_features, feature_scores):
            print(f"  {feature}: {score:.4f}")
        
        return data_selected
    
    def create_preprocessing_pipeline(self, numerical_features=None, categorical_features=None):
        """
        Create a complete preprocessing pipeline
        
        Args:
            numerical_features (list): List of numerical feature names
            categorical_features (list): List of categorical feature names
            
        Returns:
            sklearn.compose.ColumnTransformer: Complete preprocessing pipeline
        """
        print("\n" + "="*30)
        print("CREATING PREPROCESSING PIPELINE")
        print("="*30)
        
        transformers = []
        
        if numerical_features:
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_transformer, numerical_features))
            print(f"Numerical pipeline: imputation + scaling for {len(numerical_features)} features")
        
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
            print(f"Categorical pipeline: imputation + one-hot encoding for {len(categorical_features)} features")
        
        if transformers:
            preprocessor = ColumnTransformer(transformers=transformers)
            self.preprocessor = preprocessor
            return preprocessor
        else:
            print("No features specified for pipeline!")
            return None
    
    def full_preprocessing_pipeline(self, data, target_column='Sales', 
                                  outlier_method='iqr', outlier_action='cap',
                                  scaling_method='standard', feature_selection=True, 
                                  k_features=10):
        """
        Complete preprocessing pipeline
        
        Args:
            data (pd.DataFrame): Raw data
            target_column (str): Target column name
            outlier_method (str): Outlier detection method
            outlier_action (str): Outlier handling action
            scaling_method (str): Feature scaling method
            feature_selection (bool): Whether to perform feature selection
            k_features (int): Number of features to select
            
        Returns:
            pd.DataFrame: Fully preprocessed data
        """
        print("="*60)
        print("FULL PREPROCESSING PIPELINE")
        print("="*60)
        
        # Store original columns
        self.original_columns = data.columns.tolist()
        
        # Step 1: Data exploration
        data = self.load_and_explore_data(data)
        
        # Step 2: Handle missing values
        data = self.handle_missing_values(data, strategy='mean')
        
        # Step 3: Feature engineering
        data = self.feature_engineering(data)
        
        # Step 4: Handle outliers
        numerical_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                         if col != target_column]
        data = self.detect_and_handle_outliers(data, columns=numerical_cols, 
                                             method=outlier_method, action=outlier_action)
        
        # Step 5: Feature scaling
        data = self.scale_features(data, scaling_method=scaling_method, 
                                 exclude_columns=[target_column])
        
        # Step 6: Feature selection (optional)
        if feature_selection and target_column in data.columns:
            data = self.select_features(data, target_column=target_column, k=k_features)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED")
        print("="*60)
        print(f"Final dataset shape: {data.shape}")
        print(f"Final columns: {list(data.columns)}")
        
        return data
    
    def save_preprocessor(self, filepath='preprocessor.joblib'):
        """
        Save the preprocessor components
        
        Args:
            filepath (str): Path to save the preprocessor
        """
        preprocessor_components = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_selector': self.feature_selector,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'outlier_bounds': self.outlier_bounds,
            'original_columns': self.original_columns
        }
        
        joblib.dump(preprocessor_components, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='preprocessor.joblib'):
        """
        Load preprocessor components
        
        Args:
            filepath (str): Path to load the preprocessor from
        """
        try:
            components = joblib.load(filepath)
            self.scaler = components.get('scaler')
            self.imputer = components.get('imputer')
            self.feature_selector = components.get('feature_selector')
            self.preprocessor = components.get('preprocessor')
            self.feature_names = components.get('feature_names')
            self.outlier_bounds = components.get('outlier_bounds', {})
            self.original_columns = components.get('original_columns')
            print(f"Preprocessor loaded from {filepath}")
        except FileNotFoundError:
            print(f"Preprocessor file {filepath} not found!")

def preprocess_advertising_data(data, target_column='Sales', save_preprocessor=True):
    """
    Convenience function for preprocessing advertising data
    
    Args:
        data (pd.DataFrame): Raw advertising data
        target_column (str): Target column name
        save_preprocessor (bool): Whether to save the preprocessor
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    preprocessor = AdvertisingDataPreprocessor()
    
    processed_data = preprocessor.full_preprocessing_pipeline(
        data=data,
        target_column=target_column,
        outlier_method='iqr',
        outlier_action='cap',
        scaling_method='standard',
        feature_selection=True,
        k_features=10
    )
    
    if save_preprocessor:
        preprocessor.save_preprocessor('advertising_preprocessor.joblib')
    
    return processed_data, preprocessor

# Example usage
if __name__ == "__main__":
    # This would be called with your actual data
    print("Preprocessing module loaded successfully!")
    print("Use preprocess_advertising_data(your_data) to preprocess your advertising dataset.") 
