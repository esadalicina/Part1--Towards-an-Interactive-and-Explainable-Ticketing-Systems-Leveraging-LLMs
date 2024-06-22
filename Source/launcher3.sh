#!/bin/bash -l
#SBATCH --job-name=Simple-LLM
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -G 4
#SBATCH -c 12
#SBATCH -t 240
#SBATCH -N 1                  
#SBATCH --export=ALL


print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"

# Load necessary modules
module load lang/Python
 
# Activate the virtual environment
source /home/users/elicina/.virtualenvs/Master-Thesis/bin/activate || print_error_and_exit "Failed to activate virtual environment"
 
# Run your Python script
python /home/users/elicina/Master-Thesis/Source/Ticket-Classification/LLM-models/XLNet.py || print_error_and_exit "Python script execution failed"
 
