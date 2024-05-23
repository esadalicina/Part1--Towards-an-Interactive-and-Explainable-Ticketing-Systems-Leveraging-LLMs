from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Tokenization import *






# Prepare the training and test data
#train_X, test_X, train_y, test_y = train_test_split(X_res, y_res, test_size=0.2, random_state=40)




