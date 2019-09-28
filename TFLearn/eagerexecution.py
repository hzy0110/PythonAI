from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.enable_eager_execution()

tf.executing_eagerly()
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

a = tf.constant([[1, 2],
                 [3, 4]])
print(a)

b = tf.add(a, 1)
print(b)

print(a * b)

import numpy as np

c = np.multiply(a, b)
print(c)

print(a.numpy())