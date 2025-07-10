# Azure Machine Learning Quickstart Tutorial

This quickstart tutorial demonstrates how to use Azure Machine Learning to train a simple linear regression model on the Boston House Prices dataset.

## Project Structure

```
aml_quickstart_tutorial/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.template            # Environment variables template
├── data/
│   └── boston_house_prices.csv  # Sample dataset
├── code/
│   ├── train_model.py       # Model training script
│   └── upload_data_asset.py # Script to upload data to Azure ML
└── jobs/
    └── submit_training_job.py   # Script to submit training job to Azure ML
```

## Prerequisites

1. An Azure subscription
2. An Azure Machine Learning workspace
3. Python 3.8 or later
4. Azure CLI (optional, for additional commands)

## Setup Instructions

### 0. Set up everything
To set up the environment automatically run:
```
cd aml_quickstart_tutorial
./setup.sh
```

or you can also do it step by step:

### 1. Install Dependencies

```bash
cd aml_quickstart_tutorial
pip install -r requirements.txt
```

### 2. Configure Environment Variables

1. Copy the environment template:
   ```bash
   cp .env.template .env
   ```

2. Edit `.env` and fill in your Azure details:
   ```
   AZURE_SUBSCRIPTION_ID=your_subscription_id_here
   AZURE_RESOURCE_GROUP=your_resource_group_here
   AZURE_ML_WORKSPACE_NAME=your_workspace_name_here
   AZURE_ML_COMPUTE_NAME=cpu-cluster
   ```

### 3. Authenticate with Azure

Make sure you're authenticated with Azure. You can use one of these methods:

- **Azure CLI**: `az login`
- **VS Code**: Use the Azure extension
- **Environment variables**: Set `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`

## Usage Guide

### Step 1: Test Local Training (Optional)

Before running on Azure ML, you can test the training script locally:

```bash
cd code
python train_model.py --data_path ../data/boston_house_prices.csv --output_dir ./local_outputs
```

### Step 2: Upload Data Asset to Azure ML

Upload the dataset as a data asset to your Azure ML workspace:

```bash
cd code
python upload_data_asset.py
```

This creates a data asset named "boston-house-prices" that can be reused across experiments.

### Step 3: Submit Training Job to Azure ML

Submit the training job to run on Azure ML compute:

```bash
cd jobs
python submit_training_job.py
```

This will:
- Create a compute environment with the required dependencies
- Submit a training job using the uploaded data asset
- Train a linear regression model
- Save the trained model and metrics

### Step 4: Monitor the Job

After submitting the job, you can monitor it:

1. **Azure ML Studio**: Use the Studio URL provided in the output
2. **Azure CLI**: `az ml job show --name <job_name>`
3. **Stream logs**: `az ml job stream --name <job_name>`


## What the Training Script Does

The training script (`train_model.py`):

1. Loads the Boston House Prices dataset
2. Splits the data into training and test sets
3. Trains a linear regression model using scikit-learn
4. Evaluates the model and logs metrics to Azure ML
5. Saves the trained model and feature names

## Key Features

- **Data Asset Management**: The dataset is uploaded as a versioned data asset
- **Environment Management**: Automatic creation of conda environment with dependencies
- **Experiment Tracking**: Metrics are logged to Azure ML for comparison
- **Model Artifacts**: Trained models are saved and can be registered for deployment
- **Scalable Compute**: Jobs run on Azure ML compute clusters

## Metrics Tracked

The training script logs the following metrics:
- **R² Score**: Coefficient of determination (how well the model explains variance)
- **RMSE**: Root Mean Square Error
- **MSE**: Mean Square Error

## Next Steps

After completing this quickstart, you can:

1. **Register the Model**: Register the trained model for deployment
2. **Create Endpoints**: Deploy the model as a real-time or batch endpoint
3. **Experiment with Different Models**: Try other algorithms like Random Forest
4. **Add Data Drift Monitoring**: Monitor your deployed model for data drift
5. **Set up MLOps Pipelines**: Automate training and deployment with Azure DevOps or GitHub Actions

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Make sure you're logged in with `az login`
2. **Compute Not Found**: Ensure your compute cluster exists or update the compute name in `.env`
3. **Environment Creation Fails**: Check that all dependencies in `requirements.txt` are available
4. **Data Asset Not Found**: Make sure you've run the upload script first

### Getting Help

- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Azure ML Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/)
- [Azure ML CLI](https://docs.microsoft.com/en-us/azure/machine-learning/reference-azure-machine-learning-cli)

## Dataset Information

The Boston House Prices dataset contains:
- **Features**: 13 attributes including crime rate, property tax, pupil-teacher ratio, etc.
- **Target**: Median home value (MEDV)
- **Samples**: ~500 housing records
- **Task**: Regression (predicting continuous values)

This is a classic dataset for learning regression techniques, though note that it's considered outdated for real-world applications due to ethical concerns with some features.
