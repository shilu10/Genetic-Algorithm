def pick_parents(population, fitness_score): 
    """
        This function, will pick two parent chromosomes from the population.
        Params:
            population      : Array of individuals created from GenPopulation Class.
            fitness_score   : Score calculated for each individual in the population by the fitness class.
    """
    parent_1, parent_2 = random.choices(population, fitness_score, k=2)
    return parent_1, parent_2
        
    
def reproduce(parent_1, parent_2):
    """
        This function will generate a new childrens by combining two parents gene.
        Params:
            parent_1       : parent_1 array that is picked by the pick_parent function.
            parent_2       : parent_2 array that is picked by the pick_parent function.
    """
    crosspoint = random.randint(1, len(parent_1)-1) 
    child_1 = np.append(parent_1[:crosspoint], parent_2[crosspoint:])
    child_2 = np.append(parent_2[:crosspoint], parent_1[crosspoint:])
    return child_1, child_2
