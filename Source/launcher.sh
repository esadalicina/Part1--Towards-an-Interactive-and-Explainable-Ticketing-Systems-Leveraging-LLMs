#!/bin/bash -l
#SBATCH -J Simple-LLM    
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 112              
#SBATCH --t 1:00:00        
#SBATCH --qos normal       
#SBATCH -p bigmem           
#SBATCH --mem=3072    
#SBATCH --ntasks-per-socket=4

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge  print_error_and_exit "No 'module' command"
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Python 3.X by default (also on system)
module load lang/Python

# Activate the virtual environment
source /home/users/elicina/.virtualenvs/Master-Thesis/bin/activate || print_error_and_exit "Failed to activate virtual environment"

# Run your Python script
# python /home/users/elicina/Master-Thesis/Source/Ticket-Classification/RB.py || print_error_and_exit "Python script execution failed"

python /home/users/elicina/Master-Thesis/Source/Ticket-Classification/SimpleLMM.py || print_error_and_exit "Python script execution failed"

# python /home/users/elicina/Master-Thesis/Source/Ticket-Classification/LLM-models.py || print_error_and_exit "Python script execution failed"

#python /home/users/elicina/Master-Thesis/Source/Ticket-Classification/test.py || print_error_and_exit "Python script execution failed"
