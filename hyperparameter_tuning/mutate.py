import random 
import numpy as np 


def mutate(individual, param):
    """
        This function is used to mutate the childrens based randomly.
        Params:
            mutation_rate     : It is probability value, that decides whether to mutate or not.
    """
    individual_chromosome = individual.chromosome
    mutation_index = random.randint(0, len(individual_chromosome)-1)
    
    for i, key in enumerate(individual_chromosome): 
        if i == mutation_index: 
            individual.chromosome[key] = random.choice(param[key])
            
    individual.chromosome = individual_chromosome
    return individual