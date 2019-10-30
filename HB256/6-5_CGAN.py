"""
Conditional GAN
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import sys


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.reset_default_graph()
tf.set_random_seed(2)

#--------------------------------------------------
# data 불러오기
#--------------------------------------------------
mnist = input_data.read_data_sets('../temp/MNIST_data', one_hot=True)

sample_size = 10000
x_data, y_data = mnist.train.next_batch(sample_size)  # y : one_hot
print(y_data.shape)
print(y_data[0])

input_dim = x_data.shape[1]  # 784
y_dim = y_data.shape[1]      # 10


#-------------------------------------------
# hyperparameter & variable 설정
#-------------------------------------------
learning_rate = 0.001
batch_size = 256
z_size = 100
n_epochs = 5000
g_hidden_size = 128
d_hidden_size = 128
alpha = 0.1             # Leaky ReLu, alpha=0 --> ReLu

keep_prob = tf.placeholder(tf.float32,name='keep_prob') # dropout

X = tf.placeholder(tf.float32, shape=[None, input_dim], name='X')
Y = tf.placeholder(tf.float32, shape=[None, y_dim], name='Y')
Z = tf.placeholder(tf.float32, shape=[None, z_size], name='Z')


# 생성에 사용되는 가중치와 편의
G_W1 = tf.get_variable('G_W1', shape=[z_size + y_dim, g_hidden_size], initializer=tf.contrib.layers.xavier_initializer())
G_b1 = tf.Variable(tf.zeros(shape=[g_hidden_size]), name='G_b1')
G_W2 = tf.get_variable('G_W2', shape=[g_hidden_size, input_dim], initializer=tf.contrib.layers.xavier_initializer())
G_b2 = tf.Variable(tf.zeros(shape=[input_dim]), name='G_b2')
theta_G = [G_W1, G_W2, G_b1, G_b2]

#판별자에 사용되는 가중치와 편의
D_W1 = tf.get_variable('D_W1', shape=[input_dim + y_dim, d_hidden_size], initializer=tf.contrib.layers.xavier_initializer())
D_b1 = tf.Variable(tf.random_normal([d_hidden_size]), name='D_b1')
D_W2 = tf.get_variable('D_W2', shape=[d_hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())
D_b2 = tf.Variable(tf.random_normal([1]), name='D_b2')
theta_D = [D_W1, D_W2, D_b1, D_b2]


#------------------------------------------------
# 생성자 정의: 입력된 진짜 이미지와 유사한 가짜 이미지 생성
#------------------------------------------------
def generator(z,y):
    with tf.variable_scope('generator'):
        zy = tf.concat([z,y], axis=1)
        h1 = tf.matmul(zy, G_W1) + G_b1
        h1 = tf.maximum(alpha*h1, h1)
        h2 = tf.matmul(h1, G_W2) + G_b2
        out = tf.nn.tanh(h2)
        return out

#------------------------------------------------
# 판별자 정의: 입력된 진짜 데이터와 가짜 데이터를 정확하게 분류
#------------------------------------------------
def discriminator(x,y):
    with tf.variable_scope('discriminator'):
        xy = tf.concat([x,y], axis=1)
        h1 = tf.matmul(xy, D_W1) + D_b1
        h1 = tf.maximum(alpha*h1, h1)
        h1 = tf.nn.dropout(h1, keep_prob)
        h2 = tf.matmul(h1, D_W2) + D_b2
        h2 = tf.nn.dropout(h2, keep_prob)
        prob = tf.nn.sigmoid(h2)
        return prob, h2


# 가짜데이터
G = generator(Z,Y)
# D(.|Y). 진짜데이터와 가짜데이터의 조건부 판별망 결과
D_real, D_logit_real = discriminator(X,Y)
D_fake, D_logit_fake = discriminator(G,Y)

# 비용함수
# 판별자는 진짜데이터는 1, 가짜데이터는 0이 되게
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake

# 생성자는 가짜데이터의 판별망 결과가 1이 되게 만들어야 한다
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

# Optimizer
# 판별망을 훈련할 때는  theta_D 만 업데이트 된다.
D_solver = tf.train.AdamOptimizer(learning_rate,beta1=0.1).minimize(D_loss, var_list=theta_D)
# 생성망 훈련할 때는  theta_G 만 업데이트 된다.
G_solver = tf.train.AdamOptimizer(learning_rate,beta1=0.1).minimize(G_loss, var_list=theta_G)


#------------------------------------------------
# 텐서플로 그래프 생성 및 학습
#------------------------------------------------
sess = tf.Session();
sess.run(tf.global_variables_initializer())

losses = []

for epoch in range(n_epochs):
    n_batch = int(sample_size/batch_size)
    avg_loss = 0
    for ii in range(n_batch):
        ii = 0
        if ii != n_batch:
            batch_X = x_data[ii*batch_size:(ii+1)*batch_size]
            batch_Y = y_data[ii*batch_size:(ii+1)*batch_size]
        else:
            batch_X = x_data[(ii+1)*batch_size:]
            batch_Y = y_data[(ii+1)*batch_size:]

        batch_X = batch_X*2 - 1
        batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))

        D_loss_curr, _ = sess.run([D_loss, D_solver], feed_dict={X:batch_X, Z:batch_z, Y:batch_Y, keep_prob:0.9})
        G_loss_curr, _ = sess.run([G_loss, G_solver], feed_dict={X:batch_X, Z:batch_z, Y:batch_Y, keep_prob:0.9})

        losss = D_loss_curr + G_loss_curr
        avg_loss += losss/n_batch
        losses.append((D_loss_curr, G_loss_curr, avg_loss))

    if (epoch+1)%100 == 0:
        print('Epoch: {0}, Discriminator Loss: {1:.5f}, Generator Loss: {2:.5f}'.format(epoch+1, D_loss_curr, G_loss_curr))

    #------------------------------------------------
    if (epoch+1)%1000 == 0:
        tf.set_random_seed(0)
        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
        sample_y = np.zeros(shape=[16, y_dim])
        sample_y[:, 7] = 1      # 7만 그리기

        gen_samples = sess.run(G,feed_dict={Z:sample_z, Y:sample_y})
        f,axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
        f.suptitle(epoch+1)
        for ii in range(16):
            plt.subplot(4,4,ii+1)
            plt.imshow(gen_samples[ii].reshape((28,28)), cmap='Greys_r')
        plt.show()


# 판별자, 생성자의 비용함수 그림
fig, ax = plt.subplots(figsize=(7,7))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
#plt.plot(losses.T[2], label='Avg Loss')
plt.title("Training Losses")
plt.legend()
plt.show()


# 진짜 이미지 그림
# f,axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
# for ii in range(16):
#     plt.subplot(4,4,ii+1)
#     plt.imshow(x_data[ii].reshape(28,28),cmap='Greys_r')
# plt.show()
