from typing import *
import itertools
import random
import numpy as np 

class GenPopulation:
    def __init__(self): 
        self.population = []

    def generate(self, number_of_features: int, max_pop_size: int, max_features=0, verbose=False) -> np.array: 
        """
            Params: 
                number_of_features: is used to encode the actual data into genotype.
                max_pop_size:       is used to restrict number of individual generation.
                max_featrures:      how many features needed to be in the subset. 
                                      if max_features is 0, then maximum subset size is number of number 
        """
        n = max_features or number_of_features
        n_individuals = max_pop_size if max_pop_size < pow(n, 2) else pow(n, 2)
        binary_encodings = [np.array(i) for i in itertools.product([0, 1], repeat=n)]
        self.population += [random.choice(binary_encodings) for _ in range(n_individuals)]
        print(self.population) if verbose else None
        binary_encodings = []

        return np.array(self.population)