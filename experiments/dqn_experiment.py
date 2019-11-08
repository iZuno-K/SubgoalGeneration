from baselines import deepq
import misc.baselines_logger as logger
from baselines.common.atari_wrappers import make_atari, make_atari_nature
import argparse
import tensorflow as tf
import misc.log_scheduler as array_logger_getter
import yaml
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='/tmp/dqn')
    parser.add_argument('--env_id', type=str, default=None)
    parser.add_argument('--policy-mode', type=str, choices=["large_variance", "Default"], default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--exploitation_ratio_on_bottleneck', type=float, default=None)
    parser.add_argument('--bottleneck_threshold_ratio', type=float, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--total_timesteps', type=int, default=int(1e7))
    parser.add_argument('--exploration_final_eps', type=float, default=0.01)
    parser.add_argument('--exploration_fraction', type=float, default=0.1)
    parser.add_argument('--save_array_flag', action='store_true')
    parser.add_argument('--use_my_env_wrapper', action='store_true')
    parser.add_argument('--prioritized_replay', action='store_true')

    return vars(parser.parse_args())


def main():
    args = parse_args()
    logdir = args.pop('logdir')
    # logger.configure(dir=logdir, enable_std_out=True)
    logger.configure(dir=logdir, enable_std_out=False)
    with open(os.path.join(logger.get_dir(), "hyparam.yaml"), 'w') as f:
        yaml.dump(args, f, default_flow_style=False)

    policy_mode = args.pop('policy_mode')
    save_array_flag = args.pop('save_array_flag')
    use_my_env_wrapper = args.pop('use_my_env_wrapper')
    env_id = args.pop('env_id')

    if policy_mode == "large_variance":
        if args["exploitation_ratio_on_bottleneck"] is None or args["bottleneck_threshold_ratio"] is None:
            raise AssertionError
        if args["exploitation_ratio_on_bottleneck"] is not None:
            array_logger = array_logger_getter.get_logger()
            array_logger.set_log_dir(logdir, exist_ok=True)
            array_logger.set_save_array_flag(save_array_flag)


    if use_my_env_wrapper:
        env = make_atari_nature(env_id)
    else:
        env = make_atari(env_id)
    env = deepq.wrap_atari_dqn(env)
    num_cpu = 1
    config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu,
        gpu_options=tf.GPUOptions(visible_device_list=args.pop("gpu"), allow_growth=True),
    )
    config.gpu_options.allow_growth = True
    # nature_set = {'network': 'cnn', 'prioritized_replay': False, 'buffer_size': int(1e5), 'total_time_steps': int(2e6)}
    model = deepq.learn(
        env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=False,
        lr=1e-4,
        # total_timesteps=int(1e7),
        # total_timesteps=int(2e3)+1,
        buffer_size=10000,
        # exploration_fraction=0.1,
        # exploration_final_eps=0.01,
        train_freq=4,
        # learning_starts=1000,
        # target_network_update_freq=100,
        learning_starts=1000,
        # target_network_update_freq=1000,
        target_network_update_freq=500,
        gamma=0.99,
        # prioritized_replay=False,
        batch_size=64,
        # print_freq=1,
        # print_freq=200,
        print_freq=1000,
        config=config,
        bottleneck_threshold_update_freq=1000,
        **args,
    )

    model.save(os.path.join(logger.get_dir(), 'Breakout_final_model.pkl'))
    env.close()


if __name__ == '__main__':
    main()
