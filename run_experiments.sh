#!/bin/bash

# Set experiment name
EXPERIMENT_NAME="gemma_icl_experiments"

# Make sure the script is executable
# chmod +x run_experiments.sh

# Create directories if they don't exist
mkdir -p results
mkdir -p records
mkdir -p logs

echo "Starting In-Context Learning Experiments with Gemma 1.1 2B IT"

# Zero-shot experiments with different prompt types using Gemma 1.1 2B IT
echo "Running zero-shot experiments with Gemma 1.1 2B IT..."
python prompting.py --shot 0 --ptype 0 --model google/gemma-1.1-2b-it --force_cpu --template_only --use_auth_token --experiment_name "${EXPERIMENT_NAME}_gemma_basic"

echo "Running additional zero-shot experiments with different prompt types..."
python prompting.py --shot 0 --ptype 1 --model google/gemma-1.1-2b-it --force_cpu --template_only --use_auth_token --experiment_name "${EXPERIMENT_NAME}_gemma_detailed"
python prompting.py --shot 0 --ptype 2 --model google/gemma-1.1-2b-it --force_cpu --template_only --use_auth_token --experiment_name "${EXPERIMENT_NAME}_gemma_cot"

# One-shot experiments with different example selection strategies using Gemma 1.1 2B IT
echo "Running one-shot experiments with Gemma 1.1 2B IT..."
python prompting.py --shot 1 --ptype 0 --model google/gemma-1.1-2b-it --example_selection random --force_cpu --template_only --use_auth_token --experiment_name "${EXPERIMENT_NAME}_gemma_basic"
python prompting.py --shot 1 --ptype 0 --model google/gemma-1.1-2b-it --example_selection similar --force_cpu --template_only --use_auth_token --experiment_name "${EXPERIMENT_NAME}_gemma_basic"
python prompting.py --shot 1 --ptype 0 --model google/gemma-1.1-2b-it --example_selection diverse --force_cpu --template_only --use_auth_token --experiment_name "${EXPERIMENT_NAME}_gemma_basic"

# Three-shot experiments with Gemma 1.1 2B IT
echo "Running three-shot experiments with Gemma 1.1 2B IT..."
python prompting.py --shot 3 --ptype 0 --model google/gemma-1.1-2b-it --example_selection random --force_cpu --template_only --use_auth_token --experiment_name "${EXPERIMENT_NAME}_gemma_basic"
python prompting.py --shot 3 --ptype 0 --model google/gemma-1.1-2b-it --example_selection similar --force_cpu --template_only --use_auth_token --experiment_name "${EXPERIMENT_NAME}_gemma_basic"
python prompting.py --shot 3 --ptype 0 --model google/gemma-1.1-2b-it --example_selection diverse --force_cpu --template_only --use_auth_token --experiment_name "${EXPERIMENT_NAME}_gemma_basic"

echo "Experiments completed!" 