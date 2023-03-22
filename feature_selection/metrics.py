import numpy as np 
from sklearn.metrics import *
from math import sqrt


def get_metrics_value(y_true, y_pred, regression): 
    """
        this function is used to calculate the different metrics score for both regression and classification.
        Params:
            y_true: test y value.
            y_pred: prediction for test x from model.
            regression: bool value to represent whether it is regression problem or not.
    """
    if regression: 
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        adj_r2 = r2_score(y_true, y_pred)
        return rmse, adj_r2
    
    else: 
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return accuracy, precision, recall, f1