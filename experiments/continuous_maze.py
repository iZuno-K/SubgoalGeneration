from environments.continuous_space_maze import ContinuousSpaceMaze
from sac.algos import SAC
from sac.policies import GMMPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.misc.sampler import SimpleSampler, NormalizeSampler
# from sac.misc.instrument import run_sac_experiment
import tensorflow as tf
import misc.mylogger as mylogger
# from sac.envs import (
#     GymEnv,
#     MultiDirectionSwimmerEnv,
#     MultiDirectionAntEnv,
#     MultiDirectionHumanoidEnv,
#     CrossMazeAntEnv,
# )
# from rllab.envs.normalized_env import normalize
from datetime import datetime
from pytz import timezone

def main():
    goal = (20, 45)
    env = ContinuousSpaceMaze(goal=goal)
    # env = normalize(ContinuousSpaceMaze(goal=(20, 45)), normalize_obs=True)
    # env = normalize(GymEnv('HalfCheetah-v2'))
    # max_replay_buffer_size = int(1e6)
    max_replay_buffer_size = int(1e6)
    sampler_params = {'max_path_length': 1000, 'min_pool_size': 1000, 'batch_size': 128}
    # sampler = SimpleSampler(**sampler_params)
    sampler = NormalizeSampler(**sampler_params)
    entropy_coeff = 0.
    dynamic_coeff = False
    # env_id = 'ContinuousSpaceMaze{}_{}_RB{}_entropy_{}__Normalize'.format(goal[0], goal[1], max_replay_buffer_size, entropy_coeff)
    # env_id = 'SinglePath_ContinuousSpaceMaze20_45_RB1e6_entropy_0__Normalize'
    env_id = 'Test'

    print('environment set done')

    # define value function
    layer_size = 100

    qf = NNQFunction(env_spec=env.spec,
                     hidden_layer_sizes=(layer_size, layer_size))
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(layer_size, layer_size))

    # use GMM policy
    policy = GMMPolicy(
        env_spec=env.spec,
        K=4,
        hidden_layer_sizes=[layer_size, layer_size],
        qf=qf,
        reg=1e-3,
        squash=True
    )

    # TODO
    base_kwargs = dict(
        epoch_length=1000,
        n_epochs=2000,
        # scale_reward=1,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=10,
        eval_deterministic=True,
    )

    pool = SimpleReplayBuffer(env_spec=env.spec, max_replay_buffer_size=max_replay_buffer_size)
    base_kwargs = dict(base_kwargs, sampler=sampler)

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=pool,
        qf=qf,
        vf=vf,
        lr=3e-4,
        scale_reward=1.,
        discount=0.99,
        tau=1e-2,
        target_update_interval=1,
        action_prior='uniform',
        save_full_state=False,
        dynamic_coeff=dynamic_coeff,
        entropy_coeff=entropy_coeff
    )
    print("1")
    name = env_id + datetime.now().strftime("-%m%d-%Hh-%Mm-%ss")
    mylogger.make_log_dir(name)

    algorithm._sess.run(tf.global_variables_initializer())

    print("2")
    algorithm.train()


if __name__ == '__main__':
    main()
