#!/bin/bash
set -e  # Exit on any error

# Run process_data
echo "Running process_data..."
python src/run.py process_data --cfg config/cfg.yaml --dataset news --dirout ztmp/data

# Run process_data_all
echo "Running process_data_all..."
python src/run.py process_data_all --cfg config/cfg.yaml --dataset news --dirout ztmp/data

echo "All tasks completed!"
