import gym
import environments  # to register
# from value_function import Q_learning, Value_function
from algorithms.sugoal_generation import SubgoalGeneration
from misc.plotter.maze_plotter import maze_plot
import random
import numpy as np


if __name__ == '__main__':
    # env = gym.make('FrozenLake-v0')
    env = gym.make('FrozenLakeDeterministic-v0')
    a_dim = env.action_space.n
    s_dim = env.observation_space.n
    # total_timesteps = 1000
    total_timesteps = 100000
    # alg = SubgoalGeneration(state_dim=s_dim, action_dim=a_dim, gamma=0.99, alpha=0.3, epsilon=0.3)
    alg = SubgoalGeneration(state_dim=s_dim, action_dim=a_dim, gamma=0.99, alpha=0.3, epsilon=0.3, total_timesteps_for_decay=total_timesteps)  # total_timesteps
    # alg = SubgoalGeneration(state_dim=s_dim, action_dim=a_dim, gamma=0.99, alpha=0.3, epsilon=0.9, total_timesteps_for_decay=total_timesteps)
    # alg = SubgoalGeneration(state_dim=s_dim, action_dim=a_dim, gamma=0.99, alpha=0.3, epsilon=0.9, total_timesteps_for_decay=total_timesteps)
    seed = 1
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    s0 = env.reset()
    env.render()
    traj = []
    count = 0
    for i in range(total_timesteps):
        a = alg.act(state=s0, exploration=True)
        s1, r, done, _ = env.step(action=a)
        # env.render()

        traj.append([s0, a, r, s1])
        s0 = s1

        if done:
            # achieve goal
            if s1 == 15:
                # print("success")
                # traj[-1][2] = 100
                count += 1
            # else:
            #     traj[-1][2] = -10

            alg.update(trajectory=traj)
            traj = []
            s0 = env.reset()
            done = False

    print(alg.q_table)
    print(alg.v_table)
    print(count)
    print(alg.subgoals)

    maze_plot(map=env.unwrapped.desc, v_table=alg.v_table.reshape(4, 4), state_importance=alg.state_importance.reshape(4, 4))