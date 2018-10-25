import tensorflow as tf
import numpy as np


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

if __name__ == '__main__':
    # debug_pow()
    debug_argmax()