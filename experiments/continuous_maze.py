import numpy as np
import gym
import environments  # import my environments
from sac.algos import SAC
from sac.policies import GMMPolicy

def main():
    env = gym.make('ContinuousSpaceMaze-v0')

    # define value function
    layer_size = 30
    qf = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(layer_size, layer_size))
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(layer_size, layer_size))


    # use GMM policy
    policy = GMMPolicy(
        env_spec=env.spec,
        K=policy_params['K'],
        hidden_layer_sizes=(M, M),
        qf=qf,
        reg=1e-3,
    )

    sac = SAC()