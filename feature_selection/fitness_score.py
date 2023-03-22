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
        """
            This method is used to compile the neural network model using adam and binary crossentropy.
            Params:
                fc1          : Number of hidden unit for fully connected layer 1.
                fc2          : Number of hidden unit for fully connected layer 2.
                fc3          : Number of hidden unit for fully connected layer 3.
                input_dims   : Input Dimension of the Neural Network model, which number of features in the dataset.
                out_dims     : Output Dimension of the Neural Network model, by default it is 1.
        """
        model = ANN(fc1, fc2, fc3, input_dims, out_dims)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy")
        return model

    def convert_to_nparray(self, genotype): 
        """
            This method, will convert the data of type pandas dataframe into numpy array, to train a tensorflow model.
            Params: 
                genotype     : Genotype is the encoding of the features(individual in population).
        """
        col_numbers = [_ for _ in range(len(genotype)) if genotype[_] == 1]
        col_names = [self.feature_names[i] for i in range(len(self.feature_names)) if i in col_numbers]
        
        train_X = np.asarray(self.train_X[col_names])
        train_y = np.asarray(self.train_y)
        test_X = np.asarray(self.test_X[col_names])
        test_y = np.asarray(self.test_y)
        return train_X, train_y, test_X, test_y

    def train_model(self, train_X, train_y, input_dims):
        """
            This method, will train a neural net model, with the specific individual.
            Params:
                train_X       : Independent variable to the model.
                train_y       : Dependent variable to the model.
                input_dims    : Input dimension of the model, len(individual).
        """
        model = self.compile_model(32, 16, 16, input_dims, 1)
        model.fit(train_X, train_y, epochs=10, verbose=False)
        col_numbers = []
        return model

    def test_model(self, model, test_X, train_X): 
        """
            This method, will return a prediction for the prediction data.
            Params:
                test_X        : Independent Variable of testing data.
                model         : fitted model returned by the train_model method.
                train_X       : Independent Variable of training data.
        """
        pred_y = model.predict(test_X, verbose=False)
        pred_train_y = model.predict(train_X, verbose=False)
        pred_y = np.where(pred_y >= 0.5, 1, 0)
        return pred_y
    
    def get_metrics_score(self, y_true, y_pred, regression): 
        """
            This method, will calculate the metrics value for the prediction done by the model.
            Params: 
                y_true        : Ground Truth value of the testing data.
                y_pred        : Prediction data of the model.
                regression    : To specify, whether it is regression task or classfication task.
        """
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
        

    def get_fitness_score(self, regression, verbose=True): 
        """
            This method calculates the fitness score.
            Params:
                regression       : To specify, whether it is regression task or classfication task.
        """
        for individual in self.population:
            if np.any(individual): 
                train_X, train_y, test_X, test_y = self.convert_to_nparray(individual)
                model = self.train_model(train_X, train_y, len(individual))
                y_pred = self.test_model(model, test_X, train_X)
                score = self.get_metrics_score(test_y, y_pred, regression)

                score = round(score, 5)
                print("Score: ", score) if verbose else None
                self.scores.append(score)

        return self.scores / sum(self.scores), self.scores