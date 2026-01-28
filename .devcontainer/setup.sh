#!/bin/bash
set -e

echo "=================================="
echo "Starting cuRobo setup..."
echo "=================================="

# Update and install git-lfs (already included via feature, but just in case)
apt-get update
apt-get install -y git-lfs

# Initialize git-lfs
git lfs install

# Install common Python packages first
echo "Installing common Python packages..."
pip install --upgrade pip
pip install ipykernel matplotlib pandas numpy scikit-learn tensorboard

# Clone cuRobo repository
echo "Cloning cuRobo repository..."
cd /workspace
if [ ! -d "curobo" ]; then
    git clone https://github.com/NVlabs/curobo.git
fi

# Install cuRobo
echo "Installing cuRobo (this will take ~20 minutes)..."
cd curobo
pip install -e . --no-build-isolation

# Run tests to verify installation
echo "Running cuRobo tests..."
python3 -m pytest . || echo "Some tests failed, but installation may still be functional"

echo "=================================="
echo "Setup complete!"
echo "cuRobo is installed at /workspace/curobo"
echo "Examples are available in /workspace/curobo/examples/"
echo "=================================="