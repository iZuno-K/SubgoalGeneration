from baselines import deepq
import misc.baselines_logger as logger
from baselines.common.atari_wrappers import make_atari
import argparse
import tensorflow as tf
import misc.log_scheduler as array_logger_getter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='/tmp/dqn')
    parser.add_argument('--policy-mode', type=str, choices=["large_variance", "Default"], default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--exploitation_ratio_on_bottleneck', type=float, default=None)
    parser.add_argument('--bottleneck_threshold_ratio', type=float, default=None)
    # parser.add_argument('--exploitation_ratio_on_bottleneck', type=float, default=0.95)
    # parser.add_argument('--bottleneck_threshold_ratio', type=float, default=0.2)

    return vars(parser.parse_args())


def main():
    args = parse_args()
    logdir = args.pop('logdir')
    # logger.configure(dir=logdir, enable_std_out=True)
    logger.configure(dir=logdir, enable_std_out=False)
    policy_mode = args.pop('policy_mode')
    if policy_mode == "large_variance":
        if args["exploitation_ratio_on_bottleneck"] is None or args["bottleneck_threshold_ratio"] is None:
            raise AssertionError
    if args["exploitation_ratio_on_bottleneck"] is not None:
        array_logger = array_logger_getter.get_logger()
        array_logger.set_log_dir(logdir, exist_ok=True)

    env = make_atari('BreakoutNoFrameskip-v4')
    env = deepq.wrap_atari_dqn(env)
    num_cpu = 1
    config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu,
    )
    config.gpu_options.allow_growth = True

    model = deepq.learn(
        env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        lr=1e-4,
        total_timesteps=int(1e7),
        # total_timesteps=int(2e3)+1,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        # learning_starts=1000,
        # target_network_update_freq=100,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        batch_size=64,
        # print_freq=1,
        print_freq=200,
        config=config,
        bottleneck_threshold_update_freq=1000,
        **args,
    )

    model.save('Breakout_model.pkl')
    env.close()

if __name__ == '__main__':
    main()
