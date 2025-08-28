import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Concatenate, Lambda
from tensorflow.keras.initializers import Zeros

def create_fixed_effects(self, Delta1, Delta2):
    """ Create country and time fixed effect layers. """
    self.country_FE_layer = Dense(1, activation='linear', use_bias=False, kernel_initializer=Zeros())
    self.time_FE_layer = Dense(1, activation='linear', use_bias=False, kernel_initializer=Zeros())
    country_FE = self.country_FE_layer(Delta1)
    time_FE = self.time_FE_layer(Delta2)
    
    return country_FE, time_FE, self.country_FE_layer, self.time_FE_layer

def create_country_trends(self, linear_trend, quadratic_trend):
     
    self.linear_trend_layer=Dense(1, activation='linear', use_bias=False, kernel_initializer=Zeros())
    self.quadratic_trend_layer=Dense(1, activation='linear', use_bias=False, kernel_initializer=Zeros())
    linear_trend = self.linear_trend_layer(linear_trend)
    quadratic_trend = self.quadratic_trend_layer(quadratic_trend)
    
    return linear_trend, quadratic_trend, self.linear_trend_layer, self.quadratic_trend_layer