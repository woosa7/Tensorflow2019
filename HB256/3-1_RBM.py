"""
예제 3-1: 이진 입력 RBM을 MNIST 데이터에 적용

RBM은 입력 데이터에서 feature를 추출하거나
신경망의 가중치 초기값을 구하기 위한 예비 학습 등 비지도 학습에서 주로 사용함.

"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


# MNIST 파일 읽어들임
mnist = input_data.read_data_sets("../temp/MNIST_data", one_hot=True)   # label = one hot.

# 학습관련 매개변수 설정
n_input     = 784
n_hidden    = 500
display_step = 10
num_epochs  = 100
batch_size  = 256
lr          = tf.constant(0.001, tf.float32)

# 입력, 가중치 및 편향 정의
x  = tf.placeholder(tf.float32, [None, n_input], name="x") 

W  = tf.Variable(tf.random_normal([n_input, n_hidden], 0.01), name="W")
b_h = tf.Variable(tf.zeros([1, n_hidden],  tf.float32, name="b_h")) 
b_i = tf.Variable(tf.zeros([1, n_input],  tf.float32, name="b_i")) 


# ---------------------------------------------------------------------------
# 확률을 이산 상태, 즉 0과 1로 변환
def binary(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


# 단계별 Gibbs 표본추출
def cd_step(x_k):
    h_k = binary(tf.sigmoid(tf.matmul(x_k, W) + b_h)) 
    x_k = binary(tf.sigmoid(tf.matmul(h_k, tf.transpose(W)) + b_i))
    return x_k


# 표본추출 실행
def cd_gibbs(k,x_k):
    for i in range(k):
        x_out = cd_step(x_k)

    # k번 반복 후 Gibbs 표본 반환
    return x_out


# ---------------------------------------------------------------------------
# CD-2 알고리즘
# 1. 현재 입력값을 기반으로 Gibbs 표본추출을 통해 새로운 입력값 x_s를 구함
# 2. 새로운 x_s를 기반으로 새로운 은닉노드 값 act_h_s를 구함

x_s = cd_gibbs(2,x) 
act_h_s = tf.sigmoid(tf.matmul(x_s, W) + b_h) 

# 입력값이 주어질 때 은닉노드 값 act_h를 구함

act_h = tf.sigmoid(tf.matmul(x, W) + b_h)

# 은닉노드 값이 주어질 때 입력값을 추출함

_x = binary(tf.sigmoid(tf.matmul(act_h, tf.transpose(W)) + b_i))

print(x_s)
print(act_h_s)
print(act_h)
print(_x)

# ---------------------------------------------------------------------------
# 경사 하강법을 이용한 가중치 및 편향 업데이트 
W_add  = tf.multiply(lr/batch_size, tf.subtract(tf.matmul(tf.transpose(x), act_h), tf.matmul(tf.transpose(x_s), act_h_s)))

bi_add = tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(x, x_s), 0, True))
bh_add = tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(act_h, act_h_s), 0, True))

train_op = [W.assign_add(W_add), b_i.assign_add(bi_add), b_h.assign_add(bh_add)]


with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)
    
    total_batch = int(mnist.train.num_examples/batch_size)

    for epoch in range(num_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # 가중치 업데이터 실행
            batch_xs = (batch_xs > 0)*1
            _ = sess.run([train_op], feed_dict={x:batch_xs})
            
        # 실행 단계 보여주기
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1))
                  
    print("RBM training Completed !")


    # 20개의 검정용 이미지에 대해 은닉노드의 값을 계산        
    out = sess.run(act_h,feed_dict={x:(mnist.test.images[:20]> 0)*1})
    label = mnist.test.labels[:20]
    
    # 20개의 실제 검정용 이미지
    plt.figure(1)
    for k in range(20):
        plt.subplot(4, 5, k+1)
        image = (mnist.test.images[k]> 0)*1
        image = np.reshape(image,(28,28))
        plt.imshow(image,cmap='gray')
    plt.show()

    # 20개의 생성된 검정용 이미지
    plt.figure(2)
    for k in range(20):
        plt.subplot(4, 5, k+1)
        image = sess.run(_x,feed_dict={act_h:np.reshape(out[k],(-1,n_hidden))})
        image = np.reshape(image,(28,28))
        plt.imshow(image,cmap='gray')
        print(np.argmax(label[k]))
    plt.show()

    W_out = sess.run(W)
    print('W_out', W_out.shape)

    sess.close()
