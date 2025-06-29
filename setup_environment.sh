#!/bin/bash
# Setup script for Human Pose Detection project

set -e  # Exit on any error

echo "🚀 Setting up Human Pose Detection Environment"
echo "=" * 50

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "📦 Using conda for environment management..."
    
    # Create conda environment
    echo "Creating conda environment 'pose-detection'..."
    conda env create -f environment.yml -y
    
    echo "✅ Conda environment created successfully!"
    echo "To activate the environment, run:"
    echo "  conda activate pose-detection"
    
elif command -v python3 &> /dev/null; then
    echo "🐍 Using pip for environment management..."
    
    # Create virtual environment
    echo "Creating virtual environment..."
    python3 -m venv pose_detection_env
    
    # Activate virtual environment
    source pose_detection_env/bin/activate
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install requirements
    echo "Installing requirements..."
    pip install -r requirements.txt
    
    echo "✅ Virtual environment created successfully!"
    echo "To activate the environment, run:"
    echo "  source pose_detection_env/bin/activate"
    
else
    echo "❌ Error: Neither conda nor python3 found!"
    echo "Please install Python 3.8+ or Anaconda/Miniconda"
    exit 1
fi

echo ""
echo "📋 Next steps:"
echo "1. Activate the environment (see above)"
echo "2. Run: python pose_detection_professional.py --input your_video.mp4"
echo ""
echo "�� Setup complete!" 