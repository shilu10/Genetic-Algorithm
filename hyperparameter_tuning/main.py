import tensorflow.keras as keras 
import tensorflow as tf 
import numpy as np 
from population_generator import * 
from next_population_generator import * 
from preprocessing import * 
from fitness_function import *

from tensorflow.keras.layers import (Conv2D, BatchNormalization,
                                        MaxPool2D, ReLU,
                                        ELU, LeakyReLU, Flatten,
                                        Dense, AveragePooling2D
                                    )

# In each generation, no new layer will be added to the model, only the crossover and mutation, will happen
# Only in the phase, the new layer will be added with the prev_layers(best layer from the individual).
survival_rate = 0.10
mutation_rate = 0.10 
n_phase = 5
n_generation = 5 
n_population = 10

params = {

    "phase0" : {
        'a_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
        'a_include_BN': [True, False],
        'a_output_channels': [8, 16, 32, 64, 128, 256, 512],
        'activation_type': [ReLU, ELU, LeakyReLU],
        'b_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
        'b_include_BN': [True, False],
        'b_output_channels': [8, 16, 32, 64, 128, 256, 512],
        'include_pool': [True, False],
        'pool_type': [MaxPool2D, AveragePooling2D],
        'include_skip': [True, False]
    },

    "rest_phases" : {
        'include_layer': [True, False],
        'a_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
        'a_include_BN': [True, False],
        'a_output_channels': [8, 16, 32, 64, 128, 256, 512],
        'b_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
        'b_include_BN': [True, False],
        'b_output_channels': [8, 16, 32, 64, 128, 256, 512],
        'include_pool': [True, False],
        'pool_type': [MaxPool2D, AveragePooling2D],
        'include_skip': [True, False]
    }

}


# load the data and converting it to the tensorflow object, for better performace.
training_data, testing_data = tf.keras.datasets.cifar10.load_data()
train_ds, test_ds, test_X, test_y = get_training_data(training_data, testing_data)


def main(n_phase, n_generation, n_population, params): 
    """
        Main Function, that contains all the component of Variable Length Genetic Algorithm.
        Params:
            n_phase        : phase, directly proportional to number of layers.
            n_generation   : how many generation do we need to evolve our initial pop in each phase.
            n_population   : size of the population.
            params         : HyperParameter Dictionary. 
    """
    prev_phase_best = None 
    population = []
    generator = GenPopulation()
    fitness_function = FitnessFunction()

    for phase in range(n_phase): 
        print(f"phase: {phase}")
        population = generator.generate(
            params, 
            n_population,
            prev_phase_best,
            phase
        )
        for generation in range(n_generation):         
            print(f"    Generation: {generation}")
            fitness_scores, prediction_scores = fitness_function.get_fitness_score(
                population,
                train_ds,
                test_ds,
                test_X,
                test_y
            )
            print("prediction scores: ", prediction_scores)
            mating_pool_index = [_ for _ in range(len(fitness_scores)) if fitness_scores[_] >= survival_rate]
            mating_pool = population[mating_pool_index]
            mating_pool_fitness_scores = fitness_scores[mating_pool_index]
            
            if phase == 0: 
                param = params.get("phase0")
            else: 
                param = params.get("rest_phases")
            next_population = get_next_generation_population(mating_pool,
                                                                 mating_pool_fitness_scores,
                                                                 mutation_rate, 
                                                                 param
                                                            )
            population = next_population
            population = np.array(population)

        prev_phase_best_index = fitness_scores.argmax()
        prev_phase_best = population[prev_phase_best_index]
    
    return population, fitness_scores, prev_phase_best, prediction_scores

if __name__ == "__main__": 
    population, fitness_score, prev_phase_best, prediction_scores = main(n_phase, n_generation, n_population, params)