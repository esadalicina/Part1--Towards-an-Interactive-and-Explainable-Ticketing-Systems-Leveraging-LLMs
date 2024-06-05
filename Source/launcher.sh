#!/bin/bash -l
#SBATCH --job-name=Simple-LLM
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --mem=0


print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"

# Load necessary modules
module load lang/Python
 
# Activate the virtual environment
source /home/users/elicina/.virtualenvs/Master-Thesis/bin/activate || print_error_and_exit "Failed to activate virtual environment"
 
# Run your Python script
# python /home/users/elicina/Master-Thesis/Source/Ticket-Classification/RB.py || print_error_and_exit "Python script execution failed"
python /home/users/elicina/Master-Thesis/Source/Ticket-Classification/LLM-models/SimpleLMM.py || print_error_and_exit "Python script execution failed"
 
#python /home/users/elicina/Master-Thesis/Source/Ticket-Classification/LLM-models/LLM-models.py || print_error_and_exit "Python script execution failed"
 
