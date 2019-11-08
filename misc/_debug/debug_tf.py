import tensorflow as tf
import numpy as np
from tensorflow.contrib.distributions import Uniform


def debug_pow():
    dim = 4
    a = np.array([np.arange(dim), np.arange(dim)])  # (batch=2, dim)
    ph = tf.placeholder(tf.float32, (None, dim))
    pow =  tf.pow(ph, 4)
    with tf.Session() as sess:
        result = sess.run(pow, feed_dict={ph: a})
        print(result)


def debug_argmax():
    dim = 4
    a = np.array([np.arange(dim), np.arange(dim)])  # (batch=2, dim)
    ph = tf.placeholder(tf.float32, (None, dim))
    t = tf.reduce_sum(ph, axis=1)
    b = tf.argmax(t)
    c = ph[b]
    with tf.Session() as sess:
        result = sess.run([b, c], feed_dict={ph: a})
        print(result)


def debug_rand():

    class RandomTest():
        def __init__(self):
            self.q_var, self.q_var2 = self.build_network()

        @staticmethod
        def build_network():
            action_sample_t = Uniform(low=-1., high=1.)
            qf_t = action_sample_t.sample((4, 100))
            q_mean_t = tf.reduce_mean(qf_t, axis=1)  # (batch,)
            diff = qf_t - tf.expand_dims(q_mean_t, axis=1)  # (batch, self._n_approx)
            q_var_t = tf.reduce_mean(tf.square(diff), axis=1)  # (batch,)
            q_var2_t = tf.reduce_mean(tf.square(diff), axis=1)  # (batch,)

            # これそもそもエラーで動かないんだが、なぜ昔は動いていいた？
            # expand dim して dimension合わせてか引かないと本当はいけない
            # q_var3_t = tf.reduce_mean(tf.square(qf_t - q_mean_t), axis=1)  # (batch,)
            # q_kurt_t = tf.reduce_mean(tf.pow((qf_t - q_mean_t), 4), axis=1) / q_var3_t / q_var3_t  # (batch,)

            return q_var_t, q_var2_t

        def run(self):
            return tf.get_default_session().run([self.q_var, self.q_var2])

    r = RandomTest()
    with tf.Session() as sess:
        for i in range(10):
            # a, b, c, d = r.run()
            a, b = r.run()
            # print(a - b, a - c)
            print(a - b)

if __name__ == '__main__':
    # debug_pow()
    # debug_argmax()
    debug_rand()