#!/bin/bash
#SBATCH --job-name=longnd # Job name
#SBATCH --output=error/multiinput4.txt      # Output file
#SBATCH --error=log/multiinput4.txt        # Error file
#SBATCH --ntasks=1              # Number of tasks (processes)
#SBATCH --gpus=1     

sh script/multi_input3/run1.sh & sh script/multi_input3/run2.sh