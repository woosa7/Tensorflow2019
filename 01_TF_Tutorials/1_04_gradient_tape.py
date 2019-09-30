"""
Tensorflow tutorials - ML basics with Keras
Gradient Tape
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.enable_eager_execution()


# ---------------------------------------------------------------------
# Gradient tapes
# ---------------------------------------------------------------------

x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x)
for i in [0, 1]:
    for j in [0, 1]:
        print(i, j, dz_dx[i][j].numpy())
        assert dz_dx[i][j].numpy() == 8.0

print('')


# ---------------------------------------------------------------------
x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# Use the tape to compute the derivative of z with respect to the intermediate value y.
dz_dy = t.gradient(z, y)
print(dz_dy.numpy())
assert dz_dy.numpy() == 8.0


# ---------------------------------------------------------------------
# Recording control flow
# ---------------------------------------------------------------------

def f(x, y):
    output = 1.0
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(output, x)
    return output

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
    return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0


# ---------------------------------------------------------------------
# Higher-order gradients
# ---------------------------------------------------------------------
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x * x * x
    # Compute the gradient inside the 't' context manager
    # which means the gradient computation is differentiable as well.
    dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0
