# ---------------------------------------------
# Simple CNN
# ---------------------------------------------

import tensorflow as tf
from collections import namedtuple 
from libs.connections import  conv2d, linear 
from tensorflow.examples.tutorials.mnist import input_data


# ---------------------------------------------------------------------
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.reset_default_graph()
tf.set_random_seed(1)

mnist=input_data.read_data_sets("../temp/MNIST_data/",one_hot=True)

# ---------------------------------------------------------------------
learning_rate = 0.0001
epochs = 20
batch_size = 256

X = tf.placeholder(tf.float32, [None,784])
X_img = tf.reshape(X, [-1,28,28,1])

Y = tf.placeholder(tf.float32, [None,10])


# ResNet 블록 구조(bottleneck 구조) 
LayerBlock = namedtuple('LayerBlock', ['num_repeats', 'num_filters', 'bottleneck_size'])
blocks = [LayerBlock(3, 128, 32), LayerBlock(3, 256, 64), LayerBlock(3, 512, 128), LayerBlock(3, 1024, 256)]

# 채널수 64의 합성곱 출력을 만들고 다운샘플링
net = conv2d(X_img, 64, k_h=7, k_w=7, name='conv1', activation=tf.nn.relu)
net = tf.nn.max_pool(net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# ResNet 블록구조의 입력 생성 
net = conv2d(net, blocks[0].num_filters, k_h=1, k_w=1, stride_h=1, stride_w=1, padding='VALID', name='conv2')

# ResNet 블록 반복
for block_i, block in enumerate(blocks):
    for repeat_i in range(block.num_repeats):
        name = 'block_%d/repeat_%d' % (block_i, repeat_i)
        conv1 = conv2d(net, block.bottleneck_size, k_h=1, k_w=1,
                       padding='VALID', stride_h=1, stride_w=1,
                       activation=tf.nn.relu, name=name + '/conv_in')

        conv2 = conv2d(conv1, block.bottleneck_size, k_h=3, k_w=3,
                       padding='SAME', stride_h=1, stride_w=1,
                       activation=tf.nn.relu, name=name + '/conv_bottleneck')

        conv3 = conv2d(conv2, block.num_filters, k_h=1, k_w=1,
                       padding='VALID', stride_h=1, stride_w=1,
                       activation=tf.nn.relu, name=name + '/conv_out')

        net = conv3 + net

    try:
        # upscale to the next block size
        next_block = blocks[block_i + 1]
        net = conv2d(net, next_block.num_filters, k_h=1, k_w=1,
                     padding='SAME', stride_h=1, stride_w=1, bias=False,
                     name='block_%d/conv_upscale' % block_i)
    except IndexError:
        pass

# 평균 풀링을 이용하여 블록 구조의 최종 출력의 차원 변환
net = tf.nn.avg_pool(net, ksize=[1, net.get_shape().as_list()[1], net.get_shape().as_list()[2], 1],
                     strides=[1, 1, 1, 1], padding='VALID')

# ResNet 블록 구조의 최종 출력을 1D로 변환
Flat = tf.reshape(net, [-1, net.get_shape().as_list()[1] * net.get_shape().as_list()[2] * net.get_shape().as_list()[3]])

print(Flat)

# 최종 출력을 위해 소프트맥스함수 지정
Y_pred = linear(Flat, 10, activation=tf.nn.softmax)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred, labels=Y))
optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_predict = tf.equal(tf.argmax(Y_pred,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))


# ---------------------------------------------------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        avg_loss = 0
        total_batch=int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _, loss, acc = sess.run([optim, cost, accuracy], feed_dict={X: batch_xs, Y: batch_ys})

            avg_loss += loss / total_batch

        print('Epoch: %d' % (epoch + 1), 'cost= %f, accuracy= %f' % (avg_loss, acc))


    # 훈련 데이터, 검정 데이터의 분류 정확도
    # print('accuracy (test) :', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    acc_tr=0
    acc_ts=0
    for ii in range(100): #메모리 문제를 피하기 위해 자료를 100개로 분할
        xr,yr = mnist.train.next_batch(256)
        acc_tr = acc_tr + 0.01*sess.run(accuracy, feed_dict={X:xr, Y:yr})
        xt,yt = mnist.test.next_batch(100)
        acc_ts = acc_ts + 0.01*sess.run(accuracy, feed_dict={X:xt, Y:yt})

    print('accuracy (train):', acc_tr)
    print('accuracy (test) :', acc_ts)
