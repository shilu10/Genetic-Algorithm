import sklearn
import numpy as np 
import pandas as pd 
from sklearn.svm import SVC , SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class BuildModel:
    '''
        BuildModel: Supports three models SVM, LogisiticRegression, RandomForest
    ''' 
    def __init__(self, model_name: str, is_regression): 
        self.model_name = model_name

    def build(self): 
        if self.model_name == "svm": 
            if is_regression:
                return SVR()
            return SVC()
        
        if self.model_name == "lg": 
            if is_regression: 
                return LogisticRegression()
            return LogisticRegression()
        
        else: 
            if is_regression: 
                return RandomForestRegressor
        return RandomForestClassifier()


            
        