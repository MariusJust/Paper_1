from tensorflow.keras.initializers import he_normal, Zeros
from tensorflow.keras.layers import Input, Dense, Add

def create_hidden_layer(self, node):
    """ Create a hidden layer with the specified input and layer number. """
    kernel_initializer = he_normal()
    bias_initializer = Zeros()
    hidden_layer = Dense(node, activation='swish', use_bias=True,
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    return hidden_layer


def create_output_layer(self, input_tensor):
    """ Create the output layer with the specified input tensor. """
    kernel_initializer = he_normal()
    self.output_layer = Dense(1, activation='linear', use_bias=False, kernel_initializer=kernel_initializer)
    return self.output_layer(input_tensor)
