import numpy as np
import keras
import tensorflow as tf
from keras import Model
from keras import layers
from gymnasium.core import ObsType, ActType

HID_SIZE = 128
 
class A2CModel(Model):
    
    def __init__(self, obs_shape, act_shape):
        super().__init__()
        self.flatten = layers.Flatten()

        self.base = keras.Sequential([
            layers.Input((obs_shape[0] * obs_shape[1], )),
            layers.Dense(HID_SIZE, activation="relu"),
        ])
        
        self.mu = keras.Sequential([
            layers.Dense(act_shape, activation="tanh")
        ])
        self.var = keras.Sequential([
            layers.Dense(act_shape, activation="softplus")
        ])
        self.value = layers.Dense(1)

    def call(self, inputs):
        inputs = self.flatten(inputs)
        base_out = self.base(inputs)
        return self.mu(base_out), self.var(base_out), self.value(base_out)
    
    
class A2CAgent:
    
    def __init__(self, obs_size, act_shape) -> None:
        self.net = A2CModel(obs_size, act_shape)
        
    def get_action_probs(self, state: ObsType):
        return self.net(state)
    
    def get_actions(self, mu, var) -> ActType:
        sigma = tf.math.sqrt(var).numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)[0]     
        
        return actions