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

Creating requirements file:
- pip freeze > requirements.txt


PYTHON Tutorial: https://github.com/ULHPC/tutorials/tree/devel/python/basics


DATASET CODE:
- https://www.kaggle.com/code/nezukokamaado/customer-dashboard-beginner-project
- https://www.kaggle.com/code/huynhquochiep/support-ticket-sentiment-eda

BASELINE CODE:
- https://github.com/Kavitha-Kothandaraman/Automatic-IT-Ticket-Assignment-NLP
- https://github.com/IBM/support-ticket-classification
- https://monkeylearn.com/blog/ticket-classification-with-ai/
- https://github.com/sukhijapiyush/NLP-Case-Study---Automatic-Ticket-Classification
- https://github.com/yipinlyu/IT-Support-Tickets-Optimization-with-Machine-Learning
- https://medium.com/@karthikkumar_57917/it-support-ticket-classification-using-machine-learning-and-ml-model-deployment-ba694c01e416
- https://www.kaggle.com/models

WEBSITE CODE:
- https://github.com/evereux/flicket
- https://github.com/django-helpdesk/django-helpdesk
- https://www.udemy.com/course/wayscript-ticketing-application/
- https://www.freeprojectz.com/python-django-mysql-project-download/helpdesk-ticketing-system
- https://dev.to/wayscript/tutorial-build-an-employee-help-desk-ticketing-system-2lbb
- FRAG BING COPOLIT

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




IDEA FOR WEBSITE
- Apply grammary for word correction to avpid incorrect spelling