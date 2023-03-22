import pandas as pd 
import numpy as np 
import sklearn
from train_preprocessing import *
from neural_net import *
from generate_population import *
from fitness_score import *

dataframe = pd.read_csv("/home/adminuser/ArtificialIntelligence/GA_feature_selection/clean_dataframe.csv")

def get_uniques(dataframe):
    for feature in dataframe.columns:

        print(f"Feature {feature}: \n{list(pd.Series(dataframe[feature]).unique())}")

def get_missing_values(dataframe):
    for col in dataframe.isnull():
        all = list()
        print(f'Feature {col}:\n\n')
        for index, instance in enumerate(dataframe.isnull()[col]):
            if instance is True:
                all.append(index)
        print(f'Missing instances: {all}')
        print('Number of missing instances: ', len(all))

# GET UNIQUE VALUES
#print(get_uniques(dataframe))

# GET MISSING VALUE
#print(get_missing_values(dataframe))

train_X, train_y, test_X, test_y = get_training_testing_data(dataframe, "Churn", 0.8)

num_cols = len(train_X.columns)

generator = GenPopulation()

population = generator.generate(num_cols, pow(num_cols, 2), 0)

fitness_function = FitnessFunction(
                            population,
                            train_X,
                            train_y,
                            test_X,
                            test_y,
                            train_X.columns,
                            "acc"
                        )

fitness_function.get_fitness_score(regression=False)
