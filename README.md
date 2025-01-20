# Master Thesis: Towards an Interactive and Explainable Ticketing System Leveraging Large Language Models (LLMs)

This repository contains the code, data, and resources for Part 1 of the master thesis titled "Towards an Interactive and Explainable Ticketing System Leveraging Large Language Models (LLMs)". 
The project focuses on data preprocessing, model training, and optimization for classifying ticketing system data. 
It compares a variety of machine learning models, including traditional algorithms and modern LLMs to improve system performance.

Public dataset: https://www.consumerfinance.gov/data-research/consumer-complaints/


## The repository consists of two main sections:

- Data Preprocessing and Analysis: Steps to clean, preprocess, and transform ticketing data for model training.
- Model Training and Comparison: Implementation of multiple classification models, including traditional approaches (e.g., SVM, Random Forest, CNN) and Large Language Models (e.g., BERT, RoBerta, XNet), with a focus on hyperparameter optimization and explainability techniques.

## Models Implemented

### Traditional Machine Learning Models
- Support Vector Machine (SVM)
- Random Forest
- Logistic Regression
  
### Traditional Deep Learning Models

### Large Language Models (LLMs)
- BERT (Bidirectional Encoder Representations from Transformers)

## Model Optimization
- Grid Search & Random Search: Hyperparameter tuning techniques used for optimizing traditional models.
- HuggingFace: Pretrained LLM models were used on our dataset from the HuggingFace company (https://huggingface.co/models) 

### The performance of each model is evaluated based on:

- Accuracy: Correctly classified tickets.
- F1-Score: Balance between precision and recall.
- Explainability: Techniques such as SHAP (SHapley Additive exPlanations) are used to explain model decisions for ticket classification.
The results and comparison charts are stored in the results/ directory.


## Repository Contents

- data/: Contains the raw and preprocessed ticketing datasets.
- notebooks/: Jupyter notebooks with exploratory data analysis (EDA), data preprocessing, and model comparisons.
- models/: Saved models and scripts for training and testing various classifiers.
- scripts/: Bash and Python scripts for data preprocessing, model training, and evaluation.
- results/: Performance metrics, charts, and results from model comparisons.
- README.md: Instructions for accessing and using the code on the Iris Cluster.

## Prerequisites

- Python (>= 3.8)
- Conda (optional, for environment management)
- HPC Access: Iris Cluster for running high-performance jobs
- Git: To clone and manage the repository

## Installing Dependencies: 

You can create a virtual environment with the required packages using Conda or virtualenv.

- Using Conda:

-      conda create --name thesis-env python=3.8
-      conda activate thesis-env
-      pip install -r requirements.txt

- Using Virtualenv:

-      python3 -m venv venv
-      source venv/bin/activate
-      pip install -r requirements.txt


## Accessing the Iris Cluster for Data Processing and Model Training

To efficiently handle large datasets and run resource-intensive models, the Iris Cluster is used. 
Below are instructions on how to connect to the cluster and submit jobs for model training.

### Accessing the Cluster

- Deactivate Conda (if active):
-      conda deactivate
- Navigate to the SSH configuration:
-      cd ~/.ssh
- Connect to the Iris Cluster:
-      ssh iris-cluster
- Clone the repository on the Iris Cluster:
-      git clone https://github.com/esadalicina/Master-Thesis.git
- Navigate to the repository directory:
-      cd Master-Thesis


### Submitting Jobs on the Cluster

Create a shell script (train.sh) to train various models:

```
#!/bin/bash
#SBATCH --job-name=model-training       # Job name
#SBATCH --output=logs/train_out.txt     # Output log file
#SBATCH --error=logs/train_err.txt      # Error log file
#SBATCH --time=04:00:00                 # Wall time (4 hours)
#SBATCH --cpus-per-task=16              # Number of CPUs
#SBATCH --mem=64GB                      # Memory required
#SBATCH --partition=gpu                 # Use GPU partition

# Load necessary modules (e.g., TensorFlow, PyTorch)
module load python/3.8
module load tensorflow/2.4.1            # Example for TensorFlow
```

# Activate the environment
-      source /home/users/elicina/.virtualenvs/Master-Thesis/bin/activate

# Run the model training script
-      python scripts/train_model.py --model BERT --epochs 5


### To submit the training job:

-     sbatch train.sh

### (Optional) Real-Time GPU Session (Interactive Mode)

- For debugging or running models in real-time with GPU resources, you can request an interactive session:

-      si-gpu -G 1 -c 8 -t 120
  This will give you an interactive environment where you can manually run scripts and test models.



