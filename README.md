# Master-Thesis
LLM-based ticketing system 

## Acces Iris Cluster

- (base) esada@Air-von-Esada ~ % conda deactivate
- esada@Air-von-Esada ~ % cd ~/.ssh
- esada@Air-von-Esada .ssh % open congif (Open a file)
- esada@Air-von-Esada .ssh % ssh iris-cluster
- 128 [elicina@access2 ~]$ git clone https://github.com/esadalicina/Master-Thesis.git
  (Add user name and the created token as password)
- 0 [elicina@access2 ~]$ ls (See files)
- 0 [elicina@access2 ~]$ rm -r ... (Remove files/directories)
- 0 [elicina@access2 ~]$ si (enter Node for exucation of code)
- 0 [elicina@access2 ]$ exit (Get out of cluster)


Connect with HPC in VSC:
- Host Connection
- Iris-host


Submit job:
- create .sh file 
- run sbatch file path

Submit real time jon:
- si-gpu -G 1 -c 8 -t 120


Activate env:
- conda activate myenv
- source /home/users/elicina/.virtualenvs/Master-Thesis/bin/activate


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

