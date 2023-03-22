from organism import *
from fitness_function import *
import tensorflow.keras as keras 
import tensorflow as tf 
import numpy as np 


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train = X_train.astype('float32')
X_train = X_train/255.

X_test = X_test.astype('float32')
X_test = X_test/255.

y_train = tf.reshape(tf.one_hot(y_train, 10), shape=(-1, 10))
y_test = tf.reshape(tf.one_hot(y_test, 10), shape=(-1, 10))

BATCH_SIZE = 256
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1024).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

params: {

    phase0 : {
        'a_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
        'a_include_BN': [True, False],
        'a_output_channels': [8, 16, 32, 64, 128, 256, 512],
        'activation_type': [ReLU, ELU, LeakyReLU],
        'b_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
        'b_include_BN': [True, False],
        'b_output_channels': [8, 16, 32, 64, 128, 256, 512],
        'include_pool': [True, False],
        'pool_type': [MaxPool2D, AveragePooling2D]
    }

    rest_phases : {
        'include_layer': [True, False],
        'a_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
        'a_include_BN': [True, False],
        'a_output_channels': [8, 16, 32, 64, 128, 256, 512],
        'activation_type': [ReLU, ELU, LeakyReLU],
        'b_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
        'b_include_BN': [True, False],
        'b_output_channels': [8, 16, 32, 64, 128, 256, 512],
        'include_pool': [True, False],
        'pool_type': [MaxPool2D, AveragePooling2D]
    }

}



if __name__ = "__main__": 
    n_phases = 5 
    n_generation = 5 
    n_population = 10 

    prev_phase_best = None 
    population = []

    for phase in range(n_phases): 
        for generation in range(n_generation):
            for individual in range(n-n_population):
                _organism = Organism(
                    params = params,
                    prev_phase_best = prev_phase_best,
                    phase = phase 
                )
                population.append(-_organism)
            
            fitness_scores = get_fitness_score(
                population,
                train_ds,
                test_ds
            )

    """
            nature_selection()
            mutation()
            cross_over()
            generate_next_population()

    These functions are needed to be created.
    """
