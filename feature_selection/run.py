import pandas as pd 
import numpy as np 
import sklearn
#from train_preprocessing import *
#from neural_net import *
#from generate_population import *
#from fitness_score import *

def main(dataframe_path, number_of_generation=10, mutation_rate=0.1, n_individual=100, max_features=25): 
    """
        This method is the combines all the genetic algo pieces into one, and evolve over generation,
        to provide a fittest population.
        Params:
            dataframe_path         : Path for the csv file that contains the dataset.
            number_of_generation   : Number of generation to evolve.
            mutation_rate          : Mutation Rate, decides whether to mutate the children or not, based on probability.
            n_individual           : Number of individuals, needed to be created in a population, if the value is 0, 
                                     it will create a power(len(fearures), 2)
            max_features           : Subset size, if value is 0, then the subset size will be len(features).
    """
    
    dataframe = pd.read_csv(dataframe_path)
    dataframe = preprocess_dataframe(dataframe)
    
    train_X, train_y, test_X, test_y = get_training_testing_data(dataframe, "Churn", 0.8)
    print("Before Resampling", train_X.shape)

    train_X, train_y = handle_imbalance(train_X, train_y)
    print("After Resampling", train_X.shape)

    num_cols = len(train_X.columns)
    generator = GenPopulation()
    population = generator.generate(num_cols, n_individual, max_features)
    
    for i in range(number_of_generation): 
        fitness_function = FitnessFunction(
                                    population,
                                    train_X,
                                    train_y,
                                    test_X,
                                    test_y,
                                    train_X.columns,
                                    "acc"
                                )
        fitness_score , _= fitness_function.get_fitness_score(regression=False, verbose=False)
        
        new_generation_population = get_next_generation_population(population, fitness_score, mutation_rate)
        population = new_generation_population
        print(f"Generation: {i}, Max Fitness Score: {max(_)}")
            
    return population, fitness_score, _

df_path = "/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
pop, fitness_score, _ = main(df_path, 20, 0.1, 50, 10)