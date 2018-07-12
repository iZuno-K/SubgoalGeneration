import numpy as np
import tensorflow as tf


class GMM(object):
    """Gaussian Mixture Model"""
    def __init__(self, num_gauss, state_dim, action_dim):
        self.num_gauss = num_gauss


