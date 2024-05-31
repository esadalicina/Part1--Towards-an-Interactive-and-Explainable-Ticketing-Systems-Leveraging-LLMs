#!/bin/bash -l
#SBATCH --job-name=Simple-LLM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH --partition=bigmem
#SBATCH --mem=0
 
print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"
 
# Python 3.X by default (also on system)
module load lang/Python
 
# Activate the virtual environment
source /home/users/elicina/.virtualenvs/Master-Thesis/bin/activate || print_error_and_exit "Failed to activate virtual environment"
 
# Run your Python script
# python /home/users/elicina/Master-Thesis/Source/Ticket-Classification/RB.py || print_error_and_exit "Python script execution failed"
# python /home/users/elicina/Master-Thesis/Source/Ticket-Classification/SimpleLMM.py || print_error_and_exit "Python script execution failed"
 
python /home/users/elicina/Master-Thesis/Source/Ticket-Classification/LLM-models/LLM-models.py || print_error_and_exit "Python script execution failed"
 
