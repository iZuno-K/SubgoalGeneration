import gym
import environments  # to register
# from value_function import Q_learning, Value_function
from algorithms.sugoal_generation import KnackBasedQlearnig, Qlearning
from misc.plotter.maze_plotter import maze_plot
import random
import numpy as np
import misc.baselines_logger as logger
import argparse

# logger 取るべきは？
# ゴールドまで到達するのにかかったstep数
# episode単位で取るか

# evaluation
def evaluation(env, alg):
    """
    ゴールに到達するのに必要だったステップ数を出力
    ゴールにたどり着かなかったらnanを返す
    :param env:
    :param alg:
    :return:
    """
    steps = 0
    s0 = env.reset()
    eval_episodes = 1
    goal_steps = np.nan
    for i in range(eval_episodes):
        s0 = env.reset()
        done = False
        steps = 0
        while not done:
            a = alg.act(state=s0, exploration=False)
            s1, r, done, _ = env.step(action=a)
            steps += 1
            s0 = s1
            # env.render()
            if done:
                # achieve goal
                # if s1 == 80:
                if r > 0:
                    goal_steps = steps

    return goal_steps


def main(seed=1, alg_type="Bottlenck"):
    env = gym.make('CliffMazeDeterministic-v0')
    eval_env = gym.make('CliffMazeDeterministic-v0')
    map_size = (env.nrow, env.ncol)
    optimal_steps = env.nrow + env.ncol - 2
    # env = gym.make('FrozenLakeDeterministic-v0')
    logger.configure(dir='/tmp/CliffMaze/{}'.format(alg_type), log_suffix='seed{}'.format(seed), enable_std_out=False)
    a_dim = env.action_space.n
    s_dim = env.observation_space.n
    print(a_dim, s_dim)
    total_timesteps = 200000
    metric = "large_variance"
    if alg_type == "Bottleneck":
        alg = KnackBasedQlearnig(state_dim=s_dim, action_dim=a_dim, gamma=0.99, alpha=0.3, epsilon=0.3, metric=metric, exploitation_ratio=0.01)
    elif alg_type == "EpsGreedy":
        alg = Qlearning(state_dim=s_dim, action_dim=a_dim, gamma=0.99, alpha=0.3, epsilon=0.3)
    else:
        raise NotImplementedError

    env.seed(seed)
    eval_env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    s0 = env.reset()
    env.render()
    traj = []
    count = 0
    steps = 0
    eval_per_steps = 200
    eval_goal_steps = []
    for i in range(total_timesteps):
        a = alg.act(state=s0, exploration=True)
        s1, r, done, _ = env.step(action=a)
        steps += 1
        # env.render()

        traj.append([s0, a, r, s1])
        s0 = s1

        if done:
            # achieve goal
            # if s1 == 80:
            if r > 0:
                count += 1
            alg.update(trajectory=traj)

            traj = []
            s0 = env.reset()
            done = False
            steps = 0

        if i % eval_per_steps == 0:
            goal_steps = evaluation(eval_env, alg)
            logger.record_tabular('total_steps', i)
            logger.record_tabular('eval_goal_steps', goal_steps)
            logger.dump_tabular()
            if goal_steps == optimal_steps:
                print(i, goal_steps)
                break

    # print(alg.q_table)
    print(count)
    # maze_plot(map=env.unwrapped.desc, v_table=alg.v_table.reshape(*map_size), state_importance=alg.state_importance.reshape(*map_size), metric=metric)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg_type', type=str, default="Bottleneck")
    parser.add_argument('--seed', type=int, default=1)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    main(**args)


