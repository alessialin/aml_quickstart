"""
Script to submit a training job to Azure Machine Learning.
This script creates and submits a command job that runs the training script on Azure ML compute.
"""

import os
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment, AmlCompute
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Get Azure ML workspace details from environment variables
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_ML_WORKSPACE_NAME")
    compute_name = os.getenv("AZURE_ML_COMPUTE_NAME", "cpu-cluster")
    
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
    
    # Check if compute cluster exists, create if not
    try:
        compute_target = ml_client.compute.get(compute_name)
        print(f"Using existing compute cluster: {compute_name}")
    except Exception:
        print(f"Creating new compute cluster: {compute_name}")
        compute_target = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size="Standard_DS3_v2",  # Choose an appropriate VM size
            min_instances=0,  # Scale down to 0 when not in use
            max_instances=1,  # Keep it small for cost efficiency
            idle_time_before_scale_down=120,  # Scale down after 2 minutes of inactivity
            tier="Dedicated"  # Use dedicated tier for better performance
        )
        ml_client.compute.begin_create_or_update(compute_target).wait()
        print(f"Compute cluster '{compute_name}' created successfully")
    
    # Create or get environment
    env_name = "sklearn-training-env"
    try:
        env = ml_client.environments.get(env_name, version="1")
        print(f"Using existing environment: {env_name}")
    except Exception:
        print(f"Creating new environment: {env_name}")
        env = Environment(
            name=env_name,
            description="Environment for scikit-learn training",
            conda_file="conda_env.yml",  # Use conda environment file
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            version="1"
        )
        env = ml_client.environments.create_or_update(env)
    
    # Define the data input (using the uploaded data asset)
    data_input = Input(
        type="uri_file",
        path="azureml:boston-house-prices:1"  # Reference to uploaded data asset
    )
    
    # Define the training job
    job = command(
        experiment_name="boston-house-prices-training",
        display_name="Linear Regression Training",
        description="Train a linear regression model on Boston house prices data",
        code="../code",  # Path to the code directory
        command="python train_model.py --data_path ${{inputs.data}} --output_dir ${{outputs.model_output}}",
        environment=env,
        inputs={
            "data": data_input
        },
        outputs={
            "model_output": {
                "type": "uri_folder",
                "mode": "rw_mount"
            }
        },
        compute=compute_name,
        tags={
            "model_type": "linear_regression",
            "dataset": "boston_house_prices",
            "framework": "scikit-learn"
        }
    )
    
    # Submit the job
    print("Submitting training job...")
    submitted_job = ml_client.jobs.create_or_update(job)
    
    print(f"Job submitted successfully!")
    print(f"Job name: {submitted_job.name}")
    print(f"Job status: {submitted_job.status}")
    print(f"Studio URL: {submitted_job.studio_url}")
    
    # Wait for job completion (optional)
    print("\nTo monitor the job, you can:")
    print(f"1. Visit the Studio URL: {submitted_job.studio_url}")
    print(f"2. Run: az ml job show --name {submitted_job.name}")
    print(f"3. Stream logs: az ml job stream --name {submitted_job.name}")

if __name__ == "__main__":
    main()
