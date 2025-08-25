#!/bin/bash
#SBATCH --job-name=longnd # Job name
#SBATCH --output=error/multiinput3.txt      # Output file
#SBATCH --error=log/multiinput3.txt        # Error file
#SBATCH --ntasks=1              # Number of tasks (processes)
#SBATCH --gpus=1     

sh script/multi_input2/run1.sh & sh script/multi_input2/run2.sh