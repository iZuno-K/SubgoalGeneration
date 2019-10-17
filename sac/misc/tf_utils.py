import tensorflow as tf
from rllab import config


def get_default_session():
    return tf.get_default_session() or create_session()


def create_session(**kwargs):
    """ Create new tensorflow session with given configuration. """
    if "config" not in kwargs:
        kwargs["config"] = get_configuration()
    return tf.InteractiveSession(**kwargs)


def get_configuration():
    """ Returns personal tensorflow configuration. """
    # if config.USE_GPU:
    #     raise NotImplementedError
    #
    # config_args = dict()
    # return tf.ConfigProto(**config_args)q
    num_cpu = 4
    config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu,
        device_count={'GPU': 0},
        # gpu_options=tf.GPUOptions(allow_growth=True),
    )
    return config
