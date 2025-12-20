#!/bin/bash
# setup_env.sh - Automated conda environment setup for INN_final project
# Usage: ./setup_env.sh

set -e  # Exit immediately if a command exits with a non-zero status

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  INN_final Environment Setup"
echo "=========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found conda: $(conda --version)"
echo ""

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo -e "${RED}Error: environment.yml not found in current directory${NC}"
    exit 1
fi

# Check if environment already exists
ENV_NAME="inn_final"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Warning: Environment '${ENV_NAME}' already exists${NC}"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Exiting without changes."
        exit 0
    fi
fi

# Create the conda environment
echo -e "${GREEN}Creating conda environment from environment.yml...${NC}"
conda env create -f environment.yml

echo ""
echo -e "${GREEN}✓ Environment created successfully!${NC}"
echo ""

# Ask if user wants to generate training data
echo "=========================================="
echo "  Data Generation"
echo "=========================================="
echo ""
read -p "Do you want to generate the training dataset now? This will create inverse kinematics data (~5-10 min). (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Generating training data...${NC}"
    echo "This will create: ./data/inverse_data_4_100_1000.pickle"

    # Activate environment and run data generation
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ${ENV_NAME}

    # Create temporary script with correct parameters
    python -c "
import sys
sys.path.insert(0, '.')
from robot_arm_2d import RobotArm2d

# Generate data with parameters matching expected filename
# Expected: inverse_data_4_100_1000.pickle
# Format: inverse_data_{num_joints}_{num_forward}_{num_inverse_each}.pickle
arm = RobotArm2d(lengths=[0.5, 0.5, 1], sigmas=[0.25, 0.5, 0.5, 0.5])
num_forward = 100  # Number of forward configurations
num_inverse_each = 1000  # Inverse samples per configuration

print('Generating dataset...')
print(f'  - Forward configurations: {num_forward}')
print(f'  - Inverse samples per config: {num_inverse_each}')
print(f'  - Total samples: {num_forward * num_inverse_each}')
arm.generate_data(arm.sample_priors(num_forward), num_inverse_each)
print('Dataset generation complete!')
"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Training data generated successfully!${NC}"
    else
        echo -e "${RED}✗ Data generation failed${NC}"
        echo "You can generate it later by running:"
        echo "  conda activate ${ENV_NAME}"
        echo "  python robot_arm_2d.py"
    fi

    conda deactivate
fi

echo ""
echo "=========================================="
echo "  Next Steps"
echo "=========================================="
echo ""
echo "1. Activate the environment:"
echo -e "   ${GREEN}conda activate ${ENV_NAME}${NC}"
echo ""
echo "2. Verify the installation:"
echo -e "   ${GREEN}python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')\"${NC}"
echo ""
echo "3. If you skipped data generation, run:"
echo -e "   ${GREEN}python robot_arm_2d.py${NC}"
echo ""
echo "4. To deactivate when done:"
echo -e "   ${GREEN}conda deactivate${NC}"
echo ""
echo "=========================================="
