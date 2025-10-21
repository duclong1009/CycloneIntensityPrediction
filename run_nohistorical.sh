#!/bin/bash
#SBATCH --job-name=run_ablation # Job name
#SBATCH --output=error/run_ablation.txt      # Output file
#SBATCH --error=log/run_ablation.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1               # Number of GPUs per node

sh script/1days/prompt6_nohistorical.sh & sh script/3days/prompt6_nohistorical.sh