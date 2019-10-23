"""
DBN (Deep Belief Network)

deep neural network의 가중치 초기값을 구하기 위해 사용하는 generative graphical model.
학습 데이터가 적을 때 유용하다.

은닉층은 RBM 또는 autoencoder를 사용해 쌓을 수 있는데, DBN은 RBM을 사용한 것이다.
마지막 은닉층은 feature를 생성하는 역할을 한다.

분류를 위한 DBN은 다음 두 가지 방법으로 만들 수 있다.
(1) 마지막 은닉층에 softmax 함수를 사용하는 방법
(2) 마지막 은닉층 위에 1개의 은닉층과 라벨을 추가하는 방법


"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("../temp/MNIST_data", one_hot=True)


# 학습관련 매개변수 설정
n_input     = 784
n_hidden1   = 500
n_hidden2   = 256
display_step= 10
n_epoch     = 200 
batch_size  = 256

lr_rbm      = tf.constant(0.001, tf.float32)
lr_class    = tf.constant(0.01, tf.float32)

n_class     = 10
n_iter      = 200


# ---------------------------------------------------------------------------
# 입력 및 출력을 정의함
x  = tf.placeholder(tf.float32, [None, n_input], name="x") 
y  = tf.placeholder(tf.float32, [None,10], name="y") 

# 첫 번째 은닉층 관련 가중치 및 편향을 정의함
W1  = tf.Variable(tf.random_normal([n_input, n_hidden1], 0.01), name="W1") 
b1_h = tf.Variable(tf.zeros([1, n_hidden1],  tf.float32, name="b1_h")) 
b1_i = tf.Variable(tf.zeros([1, n_input],  tf.float32, name="b1_i")) 

# 두 번째 은닉층 관련 가중치 및 편향을 정의함
W2  = tf.Variable(tf.random_normal([n_hidden1, n_hidden2], 0.01), name="W2") 
b2_h = tf.Variable(tf.zeros([1, n_hidden2],  tf.float32, name="b2_h")) 
b2_i = tf.Variable(tf.zeros([1, n_hidden1],  tf.float32, name="b2_i")) 

# 라벨층 관련 가중치 및 편향을 정의함
W_c = tf.Variable(tf.random_normal([n_hidden2,n_class], 0.01), name="W_c") 
b_c = tf.Variable(tf.zeros([1, n_class],  tf.float32, name="b_c")) 


# ---------------------------------------------------------------------------
# 확률을 이산 상태, 즉 0과 1로 변환함 
def binary(prob):
    return tf.floor(prob + tf.random_uniform(tf.shape(prob), 0, 1))

# Gibbs 표본추출 단계
def cd_step(x_k,W,b_h,b_i):
    h_k = binary(tf.sigmoid(tf.matmul(x_k, W) + b_h)) 
    x_k = binary(tf.sigmoid(tf.matmul(h_k, tf.transpose(W)) + b_i))
    return x_k

# 표본추출 단계 실행     
def cd_gibbs(k,x_k,W,b_h,b_i):
    for i in range(k):
        x_out = cd_step(x_k,W,b_h,b_i) 
    # k 반복 후에 깁스 표본을 반환함
    return x_out

# ---------------------------------------------------------------------------
# 2개의 은닉층을 갖는 DBN에 대한 CD-2 알고리즘
# 1. 현재 입력값을 기반으로 깁스 표본추출을 통해 새로운 입력값 x_s를 구함
# 2. 새로운 x_s를 기반으로 새로운 은닉노드 값 h_s를 구함    

x_s = cd_gibbs(2,x,W1,b1_h,b1_i) 
act_h1_s = binary(tf.sigmoid(tf.matmul(x_s, W1) + b1_h)) 
h1_s = cd_gibbs(2,act_h1_s,W2,b2_h,b2_i) 
act_h2_s = binary(tf.sigmoid(tf.matmul(h1_s, W2) + b2_h)) 

# 입력값이 주어질 때 은닉노드 값 h를 구함
act_h1 = tf.sigmoid(tf.matmul(x, W1) + b1_h) 
act_h2 = tf.sigmoid(tf.matmul(act_h1_s, W2) + b2_h) 

# 경사 하강법을 이용한 가중치 및 편향 업데이트 
size_batch = tf.cast(tf.shape(x)[0], tf.float32)

W1_add  = tf.multiply(lr_rbm/size_batch, tf.subtract(tf.matmul(tf.transpose(x), \
          act_h1), tf.matmul(tf.transpose(x_s), act_h1_s)))
b1_i_add = tf.multiply(lr_rbm/size_batch, tf.reduce_sum(tf.subtract(x, x_s), \
           0, True))
b1_h_add = tf.multiply(lr_rbm/size_batch, tf.reduce_sum(tf.subtract(act_h1, act_h1_s), \
           0, True))

W2_add  = tf.multiply(lr_rbm/size_batch, tf.subtract(tf.matmul(tf.transpose(act_h1_s), \
          act_h2), tf.matmul(tf.transpose(h1_s), act_h2_s)))
b2_i_add = tf.multiply(lr_rbm/size_batch, tf.reduce_sum(tf.subtract(act_h1_s, h1_s), \
           0, True))
b2_h_add = tf.multiply(lr_rbm/size_batch, tf.reduce_sum(tf.subtract(act_h2, act_h2_s), \
        0, True))

updt = [W1.assign_add(W1_add), b1_i.assign_add(b1_i_add), b1_h.assign_add(b1_h_add),\
        W2.assign_add(W2_add), b2_i.assign_add(b2_i_add), b2_h.assign_add(b2_h_add)]

#-------------------------------------------------------------
# 소프트맥스 층을 추가한 분류용-DBN 을 위한 연산과정
#------------------------------------------------------------- 

logits = tf.matmul(act_h2,W_c) + b_c
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr_class).minimize(cost)
correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#-------------------------------------------------------------                   
# RBM을 쌓아 올려가며 DBN을 학습하는 텐서플로 그래프 실행   
#-------------------------------------------------------------
with tf.Session() as sess:
    # Initialize the variables of the Model
    init = tf.global_variables_initializer()
    sess.run(init)
    
    n_batch = int(mnist.train.num_examples/batch_size)
    # Start the training 
    for epoch in range(n_epoch):
        # Loop over all batches
        for i in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run the weight update 
            batch_xs = (batch_xs > 0)*1
            _ = sess.run([updt], feed_dict={x:batch_xs})
            
        # Display the running step 
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1))

    print("RBM training Completed !")
 
#--------------------------------------------------------------
# 소프트맥스 층을 추가한 분류용-DBN 학습 및 예측
#--------------------------------------------------------------
    for i in range(n_iter):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # 최적화 과정 실행
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if i % 10 == 0:
            # MINIST 훈련용 이미지의 배치에 대한 손실과 정확도를 계산
            tr_loss, tr_acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(tr_loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(tr_acc))
        
    print("Optimization Finished!")

    # MINIST 검정용 이미지에 대한 정확도 계산 
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y: mnist.test.labels}))

    sess.close()
    # Testing Accuracy: 0.8769