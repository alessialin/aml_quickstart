"""
Local test script to verify the training code works without Azure ML dependencies.
This is useful for debugging and development.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def test_local_training():
    """Test the training logic locally without Azure ML"""
    
    print("🧪 Testing local training...")
    
    # Load the dataset
    data_path = "../data/boston_house_prices.csv"
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        return False
    
    data = pd.read_csv(data_path)
    print(f"✅ Dataset loaded: {data.shape}")
    
    # Prepare features and target
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"✅ Model trained successfully!")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    
    # Test that we can save the model
    os.makedirs("test_outputs", exist_ok=True)
    model_path = "test_outputs/test_model.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved to {model_path}")
    
    # Clean up
    os.remove(model_path)
    os.rmdir("test_outputs")
    
    return True

def test_dependencies():
    """Test that all required dependencies are available"""
    
    print("📦 Testing dependencies...")
    
    try:
        import pandas
        print(f"✅ pandas: {pandas.__version__}")
    except ImportError:
        print("❌ pandas not found")
        return False
    
    try:
        import sklearn
        print(f"✅ scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn not found")
        return False
    
    try:
        import numpy
        print(f"✅ numpy: {numpy.__version__}")
    except ImportError:
        print("❌ numpy not found")
        return False
    
    # Test Azure ML packages (optional for local testing)
    try:
        import azure.ai.ml
        print(f"✅ azure-ai-ml: {azure.ai.ml.__version__}")
    except ImportError:
        print("⚠️  azure-ai-ml not found (required for Azure ML jobs)")
    
    try:
        import azure.identity
        print(f"✅ azure-identity available")
    except ImportError:
        print("⚠️  azure-identity not found (required for Azure ML jobs)")
    
    return True

def main():
    """Run all tests"""
    
    print("🧪 Azure ML Quickstart - Local Test Suite")
    print("=========================================")
    
    # Test dependencies
    deps_ok = test_dependencies()
    if not deps_ok:
        print("❌ Dependency test failed")
        return False
    
    print()
    
    # Test training
    training_ok = test_local_training()
    if not training_ok:
        print("❌ Training test failed")
        return False
    
    print()
    print("🎉 All tests passed! Your environment is ready.")
    print("Next steps:")
    print("   1. Configure your .env file")
    print("   2. Run: cd code && python upload_data_asset.py")
    print("   3. Run: cd jobs && python submit_training_job.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
