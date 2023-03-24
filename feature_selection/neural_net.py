from tensorflow.keras.layers import Dense, Input
import tensorflow as tf 
import tensorflow.keras as keras  


class ANN(keras.Model): 
    """
        SubClassing way of building the Keras Model.
    """
    def __init__(self, fc1, fc2, fc3, in_dims, out_dims): 
        super(ANN, self).__init__()
        
        self.fc1 = Dense(fc1,  activation="relu", input_shape=(in_dims, ))
        self.fc2 = Dense(fc1,  activation="relu")
        self.fc3 = Dense(fc1,  activation="relu")

        self.output_sigmoid = Dense(out_dims, activation="sigmoid")
        self.output_softmax = Dense(out_dims, activation="softmax")
        self.out_dims = out_dims
    
    def call(self, input_): 
        x = self.fc1(input_)
        x = self.fc2(x)
        x = self.fc3(x)
        if self.out_dims > 1: 
            x = self.output_softmax(x)
            return x 
        
        x = self.output_sigmoid(x)
        return x
