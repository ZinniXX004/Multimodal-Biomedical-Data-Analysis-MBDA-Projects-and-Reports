import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv("BreastCancerData (4).csv")

data_columns = data.columns
data_columns

len(data_columns)
data.drop(labels=data_columns[32], axis=1, inplace=True)


data.info()
data.isnull().sum().sort_values(ascending=False)

