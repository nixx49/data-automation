#!/bin/bash
set -e  # Exit on any error

# Activate Conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate myapp_env

if [ $# -eq 0 ]; then
    # If no arguments provided, run the default script
    echo "No command provided. Running default script: /app/scripts/run.sh"
    /app/scripts/run.sh
else
    # If arguments are provided, run them as a command
    echo "Running custom command: python src/run.py $@"
    python src/run.py "$@"
fi
