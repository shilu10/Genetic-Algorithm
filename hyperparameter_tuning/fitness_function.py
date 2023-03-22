import numpy as np 


def train_model(individual, train_ds): 
    individual.fit(
        train_ds,
        epochs=5,
        verbose=0
    )
    return individual

def get_prediction_score(individual, test_ds): 
    prediction_score = individual.evaluate(test_ds,
                                         verbose=0
                                    )

    return prediction_score


def get_fitness_score(population, train_ds, test_ds): 
    fitness_scores = []

    for individual in population: 
        model = individual.build_model()
        trained_model = train_model(model, train_ds)
        model_pred_score = get_prediction_score(trained_model, test_ds)

        fitness_scores.append(model_pred_score)
    fitness_scores = np.array(fitness_scores)
    
    return fitness_scores / sum(fitness_scores)

