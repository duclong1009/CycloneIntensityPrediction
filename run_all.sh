#!/bin/bash
#SBATCH --job-name=longnd           # Job name
#SBATCH --output=error/run_2days.txt  # Output file
#SBATCH --error=log/run_2days.txt     # Error file
#SBATCH --ntasks=5                   # Number of tasks (one per script)
#SBATCH --gpus=2                     # Number of GPUs (if you need 1 GPU for all tasks, adjust this if you need more)

# Run scripts in parallel
sh script/2days/run.sh