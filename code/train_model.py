"""
Simple Linear Regression model training script for Boston House Prices dataset.
This script can be run locally or as part of an Azure ML job.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import argparse
from azureml.core import Run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/boston_house_prices.csv', 
                        help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs', 
                        help='Directory to save the trained model')
    args = parser.parse_args()
    
    # Get the Azure ML run context
    run = Run.get_context()
    
    # Load the dataset
    print(f"Loading data from: {args.data_path}")
    data = pd.read_csv(args.data_path)
    
    # Prepare features and target
    X = data.drop('MEDV', axis=1)  # Features (all columns except MEDV)
    y = data['MEDV']  # Target (median home value)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {X.columns.tolist()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Model Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE: {mse:.4f}")
    
    # Log metrics to Azure ML
    run.log("R2_Score", r2)
    run.log("RMSE", rmse)
    run.log("MSE", mse)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(args.output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_path}")
    
    # Save feature names for later use
    feature_names_path = os.path.join(args.output_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(X.columns.tolist()))
    
    print(f"Feature names saved to: {feature_names_path}")

if __name__ == "__main__":
    main()
