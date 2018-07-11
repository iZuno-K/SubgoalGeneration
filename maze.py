import numpy as np
import gym
# from value_function import Q_learning, Value_function
from sugoal_generation import SubgoalGeneration
from plotter import maze_plot

if __name__ == '__main__':
    # env = gym.make('FrozenLake-v0')
    env = gym.make('FrozenLakeDeterministic-v0')
    a_dim = env.action_space.n
    s_dim = env.observation_space.n
    alg = SubgoalGeneration(state_dim=s_dim, action_dim=a_dim, gamma=0.9, alpha=0.3, epsilon=0.01)

    s0 = env.reset()
    env.render()
    traj = []
    count = 0
    for i in range(100000):
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
    maze_plot(map=env.unwrapped.desc, values=alg.v_table.reshape(4, 4))