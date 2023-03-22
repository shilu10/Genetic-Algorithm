import numpy as np 
from sklearn.metrics import *
from math import sqrt


def get_metrics_value(y_true, y_pred, regression): 
    if regression: 
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        adj_r2 = r2_score(y_true, y_pred)
        return rmse, adj_r2
    
    else: 
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1_score = f1_score(y_true, y_pred)

        return accuracy, precision, recall, f1_score