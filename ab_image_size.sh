#!/bin/bash
#SBATCH --job-name=longnd # Job name
#SBATCH --output=error/image_size_all.txt      # Output file
#SBATCH --error=log/image_size_all.txt        # Error file
#SBATCH --ntasks=5               # Number of tasks (processes)
#SBATCH --gpus=1       

sh script2/basedyear_data/data0/image_size.sh & sh script2/basedyear_data/data1/image_size.sh & sh script2/basedyear_data/data2/image_size.sh & sh script2/basedyear_data/data3/image_size.sh & sh script2/basedyear_data/data4/image_size.sh


wait
