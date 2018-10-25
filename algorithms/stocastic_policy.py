import numpy as np
import tensorflow as tf


class GMM(object):
    """Gaussian Mixture Model"""
    def __init__(self, num_gauss, state_dim, action_dim):
        self.num_gauss = num_gauss

class P_knack_policy(object):
    """
    - 現在目ざすコツのstateはどこか
    - 現在のstateのコツ値はいくらか
    - コツ正規化パラメータ(直近のmin max)
    -
    sub_goal = knack_state
    func: calc_kanck_of_current_state
    if not knack --> P control
    if knack --> Policy
    """




