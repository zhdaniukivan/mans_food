import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# create new constant tensor
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
#
# ###############################################################
# # create 1000 point and nose
#
# TOTAL_POINTS = 1000
#
# x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0, maxval=10)
# noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)
#
# k_true = 0.7
# b_true = 2.0
#
# y = x * k_true + b_true + noise
#
#
# plt.scatter(x, y, s=2)
# plt.show()
#
# k = tf.Variable(0.0)
# b = tf.Variable(0.0)
#
# EPOCHS = 50
# learning_rate = 0.02
#
# BATCH_SIZE = 100
# num_steps = TOTAL_POINTS // BATCH_SIZE
# # nesterova disable, the best Adam
# # opt = tf.optimizers.SGD(momentum=0.5, nesterov=True, learning_rate=0.02)
# # opt = tf.optimizers.Adagrad(learning_rate=0.4)
# # opt = tf.optimizers.Adadelta(learning_rate=4.0)
# # opt = tf.optimizers.RMSprop(learning_rate=0.01)
# opt = tf.optimizers.Adam(learning_rate=0.1)
#
# for n in range(EPOCHS):
#     for n_batch in range(num_steps):
#         y_batch = y[n_batch * BATCH_SIZE : (n_batch + 1) * BATCH_SIZE]
#         x_batch = x[n_batch * BATCH_SIZE : (n_batch + 1) * BATCH_SIZE]
#         with tf.GradientTape() as t:
#             f = k * x_batch + b
#             loss = tf.reduce_mean(tf.square(y_batch - f))
#         dk, db = t.gradient(loss, [k, b])
#         opt.apply_gradients(zip([dk, db], [k, b]))
#         # k.assign_sub(learning_rate*dk)
#         # b.assign_sub(learning_rate*db)
# print(k, b, sep='\n')
# y_pk = k * x + b
# plt.scatter(x, y, s=2)
# plt.scatter(x, y_pk, c='r', s=2)
# plt.show()
##################################################################
#
# class DenseNN(tf.Module):
#     def __init__(self, outputs):
#         super().__init__()
#         self.outputs = outputs
#         self.fl_init = False
#
#     def __call__(self, x):
#         if not self.fl_init:
#             self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name='w')
#             self.b = tf.zeros([self.outputs], dtype=tf.float32, name='b')
#
#             self.w = tf.Variable(self.w)
#             self.b = tf.Variable(self.b)
#
#             self.fl_init = True
#         y = x @ self.w + self.b
#         return y
#
# model = DenseNN(1)
# print(model(tf.constant([[1.0, 2.0]])))
#
# x_train = tf.random.uniform(minval=0, maxval=10, shape=(100, 2))
# y_train = [a + b for a, b in x_train]
#
# loss = lambda x, y: tf.reduce_mean(tf.square(x - y))
# opt = tf.optimizers.Adam(learning_rate=0.01)
#
# EPOCHS = 50
# for n in range(EPOCHS):
#     for x, y in zip(x_train, y_train):
#         x = tf.expand_dims(x, axis=0)
#         y = tf.constant(y, shape=(1, 1))
#
#         with tf.GradientTape() as tape:
#             f_loss = loss(y, model(x))
#         grads = tape.gradient(f_loss, model.trainable_variables)
#         opt.apply_gradients(zip(grads, model.trainable_variables))
#     print(f_loss.numpy())
# print(model.trainable_variables)
# print(model(tf.constant([[1.0, 2.0]])))

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])

y_train = to_categorical(y_train, 10)


class DenseNN(tf.Module):
    def __init__(self, outputs, activate='relu'):
        super().__init__()
        self.outputs = outputs
        self.activate = activate
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name='w')
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name='b')

            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)

            self.fl_init = True

        y = x @ self.w + self.b

        if self.activate == 'relu':
            return tf.nn.relu(y)
        elif self.activate == 'softmax':
            return tf.nn.softmax(y)
        return y



layer_1 = DenseNN(128)
layer_2 = DenseNN(10, activate='softmax')

def model_predict(x):
    y = layer_1(x)
    y = layer_2(y)
    return y

cross_entropy = lambda y_true, y_pred: tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))
# opt = tf.optimizers.Adam(learning_rate=0.001)
opt = tf.keras.optimizers.legacy.SGD(learning_rate=0.1)

BATCH_SIZE = 32
EPOCHS = 10
TOTAL = x_train.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

for n in range(EPOCHS):
    loss = 0
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            f_loss = cross_entropy(y_batch, model_predict(x_batch))

        loss += f_loss
        grads = tape.gradient(f_loss, [layer_1.trainable_variables, layer_2.trainable_variables])
        opt.apply_gradients(zip(grads[0], layer_1.trainable_variables))
        opt.apply_gradients(zip(grads[1], layer_2.trainable_variables))
    print(loss.numpy())

y = model_predict(x_test)
y2 = tf.argmax(y, axis=1).numpy()
acc = len(y_test[y_test == y2])/y_test.shape[0]*100
print(acc)

# acc = tf.metrics.Accuracy()
# acc.update_state(y_test, y2)
# print(acc.result().nunpy()*100)



