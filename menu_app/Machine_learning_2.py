import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# # create new constant tensor
# c = tf.constant([[1, 2],
#                 [2, 3],
#                 [3, 4]], dtype=tf.float16)
#
# print(c)
# # change same sings
# c2 = tf.cast(c, dtype=tf.float32)
# print(c2)
#
# # transform tensor from array
# c3 = np.array(c2)
# print(c3)
# c4 = c.numpy()
# print(c4)
#
# # Create new variable tensor
# print('--------------')
# v1 = tf.Variable(-1.2)
# v2 = tf.Variable([1, 2, 3, 4], dtype=tf.float32)
# v3 = tf.Variable(c)
# v1.assign(0)
# v2.assign_add([4, 4, 4, 4])
# v2.assign_sub([8, 8, 8, 8])
# print(v1, v2, v3)
#
# # change tensor from matrix
# a = tf.constant(range(30))
# b = tf.reshape(a, [5, 6])
# print(b.numpy())


# create 1000 point and nose

TOTAL_POINTS = 1000

x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0, maxval=10)
noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)

k_true = 0.7
b_true = 2.0

y = x * k_true + b_true + noise


plt.scatter(x, y, s=2)
plt.show()

k = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 50
learning_rate = 0.02

BATCH_SIZE = 100
num_steps = TOTAL_POINTS // BATCH_SIZE
# nesterova disable, the best Adam
# opt = tf.optimizers.SGD(momentum=0.5, nesterov=True, learning_rate=0.02)
# opt = tf.optimizers.Adagrad(learning_rate=0.4)
# opt = tf.optimizers.Adadelta(learning_rate=4.0)
# opt = tf.optimizers.RMSprop(learning_rate=0.01)
opt = tf.optimizers.Adam(learning_rate=0.1)

for n in range(EPOCHS):
    for n_batch in range(num_steps):
        y_batch = y[n_batch * BATCH_SIZE : (n_batch + 1) * BATCH_SIZE]
        x_batch = x[n_batch * BATCH_SIZE : (n_batch + 1) * BATCH_SIZE]
        with tf.GradientTape() as t:
            f = k * x_batch + b
            loss = tf.reduce_mean(tf.square(y_batch - f))
        dk, db = t.gradient(loss, [k, b])
        opt.apply_gradients(zip([dk, db], [k, b]))
        # k.assign_sub(learning_rate*dk)
        # b.assign_sub(learning_rate*db)
print(k, b, sep='\n')
y_pk = k * x + b
plt.scatter(x, y, s=2)
plt.scatter(x, y_pk, c='r', s=2)
plt.show()

