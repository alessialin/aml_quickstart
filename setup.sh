#!/bin/bash

# Azure ML Quickstart Setup Script
# This script helps you set up the environment and run the quickstart tutorial

set -e

echo "🚀 Azure ML Quickstart Tutorial Setup"
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8 or later."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Dependencies installed successfully!"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️  Setting up environment configuration..."
    cp .env.template .env
    echo "📝 Please edit .env file with your Azure details:"
    echo "   - AZURE_SUBSCRIPTION_ID"
    echo "   - AZURE_RESOURCE_GROUP" 
    echo "   - AZURE_ML_WORKSPACE_NAME"
    echo ""
    echo "💡 You can find these values in the Azure portal or by running:"
    echo "   az account show --query '{subscriptionId:id, name:name}'"
    echo "   az ml workspace list"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🎯 Next Steps:"
echo "=============="
echo "1. Edit .env file with your Azure ML workspace details"
echo "2. Authenticate with Azure: az login"
echo "3. Test local training: cd code && python train_model.py"
echo "4. Upload data asset: cd code && python upload_data_asset.py"
echo "5. Submit training job: cd jobs && python submit_training_job.py"
echo ""
echo "📚 For detailed instructions, see README.md"
echo "🎉 Setup complete! Happy machine learning!"
