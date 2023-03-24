def get_next_generation_population(population, fitness_score,
                                           mutation_rate, param): 
    """
        This function used to generate a new individauls by combining genes of the parensts from old population.
        Params:
            population      : Array of individuals created from GenPopulation Class.
            fitness_score   : Score calculated for each individual in the population by the fitness class.
            mutation_rate   : Amount of mutation needed to be done.
            param           : Hyperparameter dictionary, that will be used by the mutate function.
    """
    new_generation_population = []
    
    for _ in range(2): 
        index = fitness_score.argmax()
        new_generation_population.append(population[index])
        
    for _ in range(len(population)//2-1): 
        parent_1, parent_2 = pick_parents(population, fitness_score)
        # Mutation.
        if random.random() < mutation_rate: 
            parent_1 = mutate(parent_1, param)
        
        if random.random() < mutation_rate: 
            parent_2 = mutate(parent_2, param)
        
        # Mating.
        child1_chromosome, child2_chromosome = reproduce(parent_1, parent_2)
        
        # Creating new organsim for next generation.
        child1_organism = Organism(parent_1.params, parent_1.prev_phase_best, parent_1.phase)
        child1_organism.chromosome = child1_chromosome
        child1_organism.build_model()
        
        child2_organism = Organism(parent_1.params, parent_1.prev_phase_best, parent_1.phase)
        child2_organism.chromosome = child2_chromosome
        child2_organism.build_model()
      
        new_generation_population.append(child1_organism)
        new_generation_population.append(child2_organism)
    
    print("[+] Successfully generated the next generation's population.")
    return new_generation_population