"""
Convolution

image --> convolutional filter (= kernel or window) --> feature map

pooling layer : feature map의 이동(shift)과 왜곡(distortion)을 방지.

"""

# ---------------------------------------------
# Simple CNN
# ---------------------------------------------

import numpy as np
import tensorflow as tf
from libs.connections import linear
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# ---------------------------------------------------------------------
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.reset_default_graph()
tf.set_random_seed(1)

# data
mnist=input_data.read_data_sets("../temp/MNIST_data/", one_hot=True)

# 훈련 데이터 이미지
# f,axes = plt.subplots(figsize=(7,7), nrows=3, ncols=4, sharey=True, sharex=True)
# for ii in range(12):
#     plt.subplot(3,4,ii+1)
#     plt.imshow(mnist.train.images[ii].reshape(28,28),cmap='Greys_r')
# plt.show()


# ---------------------------------------------------------------------
learning_rate = 0.001
epochs = 100
batch_size = 100

X = tf.placeholder(tf.float32, [None,784])
X_img = tf.reshape(X, [-1,28,28,1])

Y = tf.placeholder(tf.float32, [None,10])


# 첫 번째 합성곱층
K1 = tf.Variable(tf.random_normal([5,5,1,20], stddev=0.01))
a1 = tf.nn.conv2d(X_img, K1, strides=[1,1,1,1], padding='VALID')
a1 = tf.layers.batch_normalization(a1, training=True)
a1 = tf.nn.relu(a1)

# 첫 번째 풀링층
h1 = tf.nn.max_pool(a1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

# 두 번째 합성곱층
K2 = tf.Variable(tf.random_normal([5,5,20,50], stddev=0.01))
a2 = tf.nn.conv2d(h1, K2, strides=[1,1,1,1], padding='VALID')
a2 = tf.layers.batch_normalization(a2, training=True)
a2 = tf.nn.relu(a2)

# 두 번째 풀링층
h2 = tf.nn.max_pool(a1,ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

# 출력을 1D로 변환
Flat = tf.reshape(h2, [-1, np.prod(h2.get_shape().as_list()[1:4])])

# 완전 연결 신경망
W1 = tf.get_variable("W1", shape=[np.prod(h1.get_shape().as_list()[1:4]), 50],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([50]))
L1 = tf.matmul(Flat, W1)+b1

# 최종 출력
Y_pred =linear(L1, 10, activation=tf.nn.softmax)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred,labels=Y))

optim = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_predict = tf.equal(tf.argmax(Y_pred,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))


# ---------------------------------------------------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        avg_loss = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _, loss, acc = sess.run([optim, cost, accuracy], feed_dict={X:batch_xs, Y:batch_ys})

            avg_loss += loss/total_batch

        if (epoch+1)%10 == 0 or epoch == 0:
            print('Epoch: %d' %(epoch+1),'cost= %f, accuracy= %f' %(avg_loss, acc))

    # 검정 데이터의 정확도
    print('accuracy (test) :', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
