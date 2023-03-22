import random 
import numpy as np

def get_next_generation_population(population, fitness_score, mutation_rate): 
    """
        This function used to generate a new individauls by combining genes of the parensts from old population.
        Params:
            population      : Array of individuals created from GenPopulation Class.
            fitness_score   : Score calculated for each individual in the population by the fitness class.
            mutation_rate   : Amount of mutation needed to be done.
    """
    new_generation_population = []
    while len(new_generation_population) <= len(population)-1: 
        parent_1, parent_2 = pick_parents(population, fitness_score)
        print(parent_1, parent_2, "parents")
        child_1, child_2 = reproduce(parent_1, parent_2)
        mutated_child1 = mutate(child_1, mutation_rate)
        mutated_child2 = mutate(child_2, mutation_rate)
        
        new_generation_population.append(mutated_child1)
        new_generation_population.append(mutated_child2)
        
    return new_generation_population
    