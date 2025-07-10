"""
Script to upload the Boston House Prices dataset as a data asset to Azure Machine Learning.
"""

import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Get Azure ML workspace details from environment variables
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_ML_WORKSPACE_NAME")
    
    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError(
            "Please set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, and "
            "AZURE_ML_WORKSPACE_NAME in your .env file"
        )
    
    # Initialize Azure ML client
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    
    print(f"Connected to workspace: {workspace_name}")
    
    # Define the data asset
    data_asset = Data(
        name="boston-house-prices",
        version="1",
        description="Boston House Prices dataset for regression modeling",
        path="../data/boston_house_prices.csv",
        type=AssetTypes.URI_FILE,
        tags={
            "source": "sklearn.datasets",
            "task": "regression"
        }
    )
    
    # Upload the data asset
    print("Uploading data asset to Azure ML...")
    uploaded_data_asset = ml_client.data.create_or_update(data_asset)
    
    print(f"Data asset uploaded successfully!")
    print(f"Name: {uploaded_data_asset.name}")
    print(f"Version: {uploaded_data_asset.version}")
    print(f"ID: {uploaded_data_asset.id}")
    
    # List all data assets to verify
    print("\nExisting data assets in the workspace:")
    data_assets = ml_client.data.list()
    for asset in data_assets:
        print(f"  - {asset.name} (v{asset.version})")

if __name__ == "__main__":
    main()
