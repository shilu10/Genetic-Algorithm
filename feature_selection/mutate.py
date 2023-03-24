def mutate(individual, mutation_rate):
    """
        This function is used to mutate the childrens based randomly.
        Params:
            mutation_rate     : It is probability value, that decides whether to mutate or not.
    """
    random_index = random.randint(0, len(individual)-1)
    
    if random.random() < mutation_rate: 
        random_index_val = individual[random_index]
        inverse_random_index_val = int(not random_index_val)
        individual[random_index] = inverse_random_index_val
        
    return individual