from tensorflow.keras.layers import Dense, Input
import tensorflow as tf 
import tensorflow.keras as keras  


class ANN(keras.Model): 
    def __init__(self, fc1, fc2, fc3, in_dims, out_dims): 
        super(self, ANN).__init__ 
        
        self.input = Input(input_shape=(in_dims, ))
        self.fc1 = Dense(fc1,  activation="relu")
        self.fc2 = Dense(fc1,  activation="relu")
        self.fc3 = Dense(fc1,  activation="relu")

        self.output_sigmoid = Dense(out_dims, activation="sigmpid")
        self.output_softmax = Dense(out_dims, , activation="softmax")
    
    def call(self, input): 
        x = self.input(input)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        if out_dims > 1: 
            x = self.output_softmax(x)
            return x 
        
        x = self.output_sigmoid(x)
        return x
