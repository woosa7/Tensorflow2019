"""
예제 3-6: 잡음 제거 오토인코더: Pro Deep Learning + Hands on ML

Denoising autoencoder

"""

# 필요한 라이브러리를 불러들임
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# MNIST 데이터 읽어들임
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../temp/MNIST_data", one_hot=True)

# 학습관련 매개변수 설정
learning_rate = 0.01
batch_size = 150
display_step = 10
examples_to_show = 10
noise_level = 1.0

n_input = 28*28 # MNIST 데이터 입력 (784)
n_hidden = 300 # 은닉노드 개수


# 잡음 포함 입력, 가중치 및 편향을 정의함
X = tf.placeholder("float", [None, n_input])
X_noisy = X + noise_level * tf.random_normal(tf.shape(X))

weights = {
    'encoder_h': tf.Variable(tf.random_normal([n_input,n_hidden])),
    'decoder_h': tf.Variable(tf.random_normal([n_hidden,n_input])),
}    
biases = {
    'encoder_b': tf.Variable(tf.random_normal([n_hidden])),
    'decoder_b': tf.Variable(tf.random_normal([n_input])),
}

# ---------------------------------------------------------------------------
# 인코더를 구축함
def encoder(x):
    e_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h']),
                                   biases['encoder_b']))
    return e_layer

# 디코더를 구축함
def decoder(x):
    d_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, tf.transpose(weights['encoder_h'])),
                                   biases['decoder_b']))
    return d_layer

# ---------------------------------------------------------------------------
# 오토인코더 모형 구축
op_encoder = encoder(X_noisy)
op_decoder = decoder(op_encoder)

# 예측값
y_true = X
y_pred = op_decoder

# 손실함수 및 최적화 정의 
loss = tf.reduce_mean(tf.square(y_true - y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 텐서플로 그래프 실행
start_time = time.time()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    num_batch = int(mnist.train.num_examples/batch_size)

    num_epoch = 100
    for epoch in range(num_epoch):
        # 모든 미니배치에 대해 반복함
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, los = sess.run([optimizer, loss], feed_dict={X: batch_xs})

        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "loss =", "{:.9f}".format(los))

    print("Optimization Finished!")
    end_time = time.time()
    print('elapsed time:', end_time - start_time)


    # 검정 데이터 집합에 인코더와 디코더를 적용함
    n_digit = 8
    reconstruct_digit = sess.run(y_pred, feed_dict={X: mnist.test.images[:n_digit]})
    # 검정 데이터에 속하는 원래 이미지와 복원된 이미지를 비교함 
    fig, a = plt.subplots(2, n_digit, figsize=(10, 2))
    for i in range(n_digit):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)),cmap='gray')
        a[1][i].imshow(np.reshape(reconstruct_digit[i], (28, 28)),cmap='gray')
    plt.show()

    # 인코더 및 디코더 관련 가중치를 저장함 
    w_encod = sess.run(weights['encoder_h'])
    w_decod = sess.run(weights['decoder_h'])
