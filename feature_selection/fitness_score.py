import random 
import numpy as np 
import sklearn
from metrics import *
from neural_net import *
from tensorflow.keras.optimizers import Adam
import tensorflow as tf 
import math 

class FitnessFunction: 
    def __init__(self, population, train_X, train_y, test_X, test_y, feature_names, scoring_criteria): 
        '''
            Params:
                population: Population array from generate_pop method.
                model: Model from build model method.
                train_X, train_y: training data for our model.
                test_X, test_y: testing data for our model.
                scoring_criteria: On what basis scoring, needs to happend eg: Accuracy, f1-score, recall, precision.
        '''

        self.population = population
        self.scores = []
        self.train_X = train_X 
        self.train_y = train_y
        self.test_X = test_X 
        self.test_y = test_y
        self.scoring_criteria = scoring_criteria
        self.feature_names = feature_names
    
    def compile_model(self, fc1, fc2, fc3, input_dims, out_dims): 
        model = ANN(fc1, fc2, fc3, input_dims, out_dims)
        model.compile(optimizer=Adam(), loss="binary_crossentropy")
        return model

    def convert_to_nparray(self): 
        train_X = np.asarray(train_X[col_names]).astype(np.float64)
        train_y = np.asarray(train_y).astype(np.float64)
        test_X = np.asarray(test_X[col_names]).astype(np.float64)
        test_y = np.asarray(test_y).astype(np.float64)
        return train_X, train_y, test_X, test_y

    def train_model(self, genotype, train_X, train_y):
        col_numbers = [_ for _ in range(len(genotype)) if genotype[_] == 1]
        col_names = [self.feature_names[i] for i in range(len(self.feature_names)) if i in col_numbers]

        model = compile_model(16, 32, 64, len(col_names), 1)
        model.fit(train_X, train_y, epochs=100)

        col_numbers = []
        return model, col_names

    def test_model(self, model, test_X, train_X, selected_cols): 
        pred_y = model.predict(test_X[selected_cols])
        pred_train_y = model.predict(train_X[selected_cols])
        
        return pred_y, pred_train_y

    def get_metrics_score(self, y_true, y_pred, regression): 

        results = get_metrics_value(y_true, y_pred, regression)

        if not regression:
            acc, precision, recall, f1_score = results 

        else: rmse, adj_r2 = results

        if self.scoring_criteria == "rmse" and regression: 
            return ((0.9 * rmse) + (0.10 * adj_r2) )

        if self.scoring_criteria == "adj_r2" and regression: 
            return ((0.9 * adj_r2) + (0.1 * rmse))

        if self.scoring_criteria == "acc": 
            return ((0.88 * acc) + (0.03 * recall) + (0.03 * precision) + (0.03 * f1_score))
        
        if self.scoring_criteria == "recall": 
            return ((0.03 * acc) + (0.88 * recall) + (0.03 * precision) + (0.03 * f1_score))
        
        if self.scoring_criteria == "precision": 
            return ((0.03 * acc) + (0.03 * recall) + (0.88 * precision) + (0.03 * f1_score))

        if self.scoring_criteria == "f1": 
            return ((0.03 * acc) + (0.03 * recall) + (0.03 * precision) + (0.88 * f1_score))
        

    def get_fitness_score(self,  regression): 

        for individual in self.population:
            if np.any(individual): 
                train_X, train_y, test_X, test_y = self.convert_to_nparray()
                model, selected_cols = self.train_model(individual, train_X, train_y)
                y_pred, y_pred_for_train = self.test_model(model, test_X, train_X, selected_cols)
                score = self.get_metrics_score(test_y, y_pred, regression)

                score = math.round(score, 5)
                self.scores.append(score)

        return self.scores