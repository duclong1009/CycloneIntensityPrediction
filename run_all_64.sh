#!/bin/bash
#SBATCH --job-name=longnd # Job name
#SBATCH --output=error/multiinput.txt      # Output file
#SBATCH --error=log/multiinput.txt        # Error file
#SBATCH --ntasks=1              # Number of tasks (processes)
#SBATCH --gpus=1     

sh multi_input/run1.sh & sh multi_input/run2.sh