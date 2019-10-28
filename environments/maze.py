import gym
import environments  # to register
# from value_function import Q_learning, Value_function
from algorithms.sugoal_generation import KnackBasedQlearnig
from misc.plotter.maze_plotter import maze_plot
import random
import numpy as np
import misc.baselines_logger as logger

# logger 取るべきは？
# ゴールドまで到達するのにかかったstep数
# episode単位で取るか

# evaluation

if __name__ == '__main__':
    # env = gym.make('FrozenLake-v0')
    env = gym.make('CliffMazeDeterministic-v0')
    eval_env = gym.make('CliffMazeDeterministic-v0')
    # env = gym.make('FrozenLakeDeterministic-v0')
    logger.configure(dir='/tmp/CliffMaze', enable_std_out=False)
    a_dim = env.action_space.n
    s_dim = env.observation_space.n
    # total_timesteps = 1000
    total_timesteps = 200000
    metric = "large_variance"
    # alg = SubgoalGeneration(state_dim=s_dim, action_dim=a_dim, gamma=0.99, alpha=0.3, epsilon=0.3)
    # alg = KnackBasedQlearnig(state_dim=s_dim, action_dim=a_dim, gamma=0.99, alpha=0.3, epsilon=0.3, total_timesteps_for_decay=total_timesteps)  # total_timesteps
    alg = KnackBasedQlearnig(state_dim=s_dim, action_dim=a_dim, gamma=0.99, alpha=0.3, epsilon=0.5, total_timesteps_for_decay=total_timesteps, metric=metric, exploitation_ratio=0.2)  # total_timesteps

    # alg = SubgoalGeneration(state_dim=s_dim, action_dim=a_dim, gamma=0.99, alpha=0.3, epsilon=0.9, total_timesteps_for_decay=total_timesteps)
    # alg = SubgoalGeneration(state_dim=s_dim, action_dim=a_dim, gamma=0.99, alpha=0.3, epsilon=0.9, total_timesteps_for_decay=total_timesteps)
    seed = 1
    env.seed(seed)
    eval_env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    s0 = env.reset()
    env.render()
    traj = []
    count = 0
    steps = 0
    for i in range(total_timesteps):
        a = alg.act(state=s0, exploration=True)
        s1, r, done, _ = env.step(action=a)
        steps += 1
        # env.render()

        traj.append([s0, a, r, s1])
        s0 = s1

        if done:
            # achieve goal
            if s1 == 80:
                # print("success")
                # traj[-1][2] = 100
                count += 1
            # else:
            #     traj[-1][2] = -10
            alg.update(trajectory=traj)

            traj = []
            s0 = env.reset()
            done = False
            steps = 0

    print(alg.q_table)
    print(count)

    maze_plot(map=env.unwrapped.desc, v_table=alg.v_table.reshape(9, 9), state_importance=alg.state_importance.reshape(9, 9), metric=metric)