"""
예제 3-3: 분류용-RBM을 MNIST 데이터에 적용

기존의 RBM에 label 층을 추가해 분류를 위해 사용할 수 있다.

"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
import sys

# ---------------------------------------------------------------------------
## 데이터 불러오기
iris = datasets.load_iris()

irisX = iris.data
irisY = iris.target

# one hot vector 생성
irisY = pd.get_dummies(irisY)  
irisY = np.array(irisY)  

## 이진 입력 RBM을 위한 입력 데이터의 정규화 
minmax = np.amin(irisX, 0), np.amax(irisX, 0)
no_irisX = (irisX-minmax[0])/(minmax[1]-minmax[0])


# ---------------------------------------------------------------------------
# 훈련 데이터와 검정 데이터를 7:3 비율로 분리
# 3개의 label이 각 50개씩 존재.

np.random.seed(2019)
ind1 = np.random.permutation(50)
p_ind2 = np.arange(50,100)
ind2 = np.random.permutation(p_ind2) 
p_ind3 = np.arange(100,150)
ind3 = np.random.permutation(p_ind3)

tr_ind1 = ind1[:35]
tr_ind2 = ind2[:35]
tr_ind3 = ind3[:35]
tr_ind = np.concatenate((tr_ind1,tr_ind2,tr_ind3),axis=0)

te_ind1 = ind1[35:]
te_ind2 = ind2[35:]
te_ind3 = ind3[35:]
te_ind = np.concatenate((te_ind1,te_ind2,te_ind3),axis=0)

trX = no_irisX[tr_ind]
teX = no_irisX[te_ind]
trY = irisY[tr_ind]
teY = irisY[te_ind]


# ---------------------------------------------------------------------------
## 학습관련 매개변수 설정 
n_class     = 3
n_input     = 4
n_hidden    = 20
display_step = 10
num_epochs  = 300
batch_size  = 5
lr          = tf.constant(0.01, tf.float32)


## 입력, 가중치 및 편향을 정의함
x  = tf.placeholder(tf.float32, [None, n_input], name="x") 
y  = tf.placeholder(tf.float32, [None,n_class], name="y")

W_xh  = tf.Variable(tf.random_normal([n_input, n_hidden], 0.01), name="W_xh") 
W_hy = tf.Variable(tf.random_normal([n_hidden,n_class], 0.01), name="W_hy")

b_i = tf.Variable(tf.zeros([1, n_input],  tf.float32, name="b_i")) 
b_h = tf.Variable(tf.zeros([1, n_hidden],  tf.float32, name="b_h"))
b_y = tf.Variable(tf.zeros([1, n_class],  tf.float32, name="b_y")) 


## 확률을 이산 상태, 즉 0과 1로 변환함
def binary(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))
          
## Gibbs 표본추출 단계
def gibbs_step(x_k,y_k):
        h_k = binary(tf.sigmoid(tf.matmul(x_k, W_xh) + tf.matmul(y_k, tf.transpose(W_hy)) + b_h)) 
        x_k = binary(tf.sigmoid(tf.matmul(h_k, tf.transpose(W_xh)) + b_i))

        y_k = tf.nn.softmax(tf.matmul(h_k, W_hy) + b_y)
        return x_k,y_k

## 표본추출 단계 실행    
def gibbs_sample(k,x_k,y_k):
    for i in range(k):
        x_out,y_out = gibbs_step(x_k,y_k) 

    return x_out,y_out


# ---------------------------------------------------------------------------
## CD-2 알고리즘
# 현재 입력값 및 출력값을 기반으로 깁스 표본추출을 통해 새로운 입력값 x_s, y_s를 구함
x_s,y_s = gibbs_sample(2,x,y)

# 새로운 x_s, y_s를 기반으로 새로운 은닉노드 값 act_h_s를 구함       
act_h_s = tf.sigmoid(tf.matmul(x_s, W_xh) + tf.matmul(y_s, tf.transpose(W_hy)) + b_h)  

# 입력값 및 출력값이 주어질 때 은닉노드 값 act_h를 구함
act_h = tf.sigmoid(tf.matmul(x, W_xh) + tf.matmul(y, tf.transpose(W_hy)) + b_h) 

# 은닉노드 값이 주어질 때 입력값을 추출함
_x = binary(tf.sigmoid(tf.matmul(act_h, tf.transpose(W_xh)) + b_i))


## 경사 하강법을 이용한 분류용-RBM의 가중치 및 편향 업데이트 
W_xh_add = tf.multiply(lr/batch_size, tf.subtract(tf.matmul(tf.transpose(x), act_h), \
	       tf.matmul(tf.transpose(x_s), act_h_s)))
W_hy_add = tf.multiply(lr/batch_size, tf.subtract(tf.matmul(tf.transpose(act_h), y), \
	       tf.matmul(tf.transpose(act_h_s), y_s)))

bi_add = tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(x, x_s), 0, True))
bh_add = tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(act_h, act_h_s), 0, True))
by_add = tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(y, y_s), 0, True))

train_op = [W_xh.assign_add(W_xh_add), W_hy.assign_add(W_hy_add), b_i.assign_add(bi_add), b_h.assign_add(bh_add), b_y.assign_add(by_add)]


with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(num_epochs):
        ind2 = np.random.permutation(len(trX))
        num_batch = int(len(trX)/batch_size)

        for i in range(num_batch):
            batch_xs = trX[ind2[i*batch_size:(i+1)*batch_size]]
            batch_ys = trY[ind2[i*batch_size:(i+1)*batch_size]]
            # 가중치 업데이터 실행 
            # batch_x = (batch_x > 0)*1
            _ = sess.run([train_op], feed_dict={x:batch_xs, y:batch_ys})
            
        # 매 에포크마다 실행 단계 보여주기 
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1))

    print("Discriminative RBM training Completed !")


    ## 훈련 데이터에 대한 분류용-RBM의 정확도 계산
    tr_lab1 = np.zeros((len(trX),n_class)); tr_lab1[:,0] = 1
    tr_lab2 = np.zeros((len(trX),n_class)); tr_lab2[:,1] = 1
    tr_lab3 = np.zeros((len(trX),n_class)); tr_lab3[:,2] = 1
    
    tr_f1_xl = tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(trX,tf.float32), W_xh) \
    	       + tf.matmul(tf.cast(tr_lab1,tf.float32), tf.transpose(W_hy)) + b_h),1)
    tr_f2_xl = tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(trX,tf.float32), W_xh) \
    	       + tf.matmul(tf.cast(tr_lab2,tf.float32), tf.transpose(W_hy)) + b_h),1)
    tr_f3_xl = tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(trX,tf.float32), W_xh) \
    	       + tf.matmul(tf.cast(tr_lab3,tf.float32), tf.transpose(W_hy)) + b_h),1)

    tr_f_xl = b_y + tf.transpose([tr_f1_xl,tr_f2_xl,tr_f3_xl])
    tr_y_hat = tf.nn.softmax(tr_f_xl)
        
    tr_correct_pred = tf.equal(tf.argmax(tr_y_hat,1), tf.argmax(trY, 1))
    tr_accuracy = tf.reduce_mean(tf.cast(tr_correct_pred, tf.float32))

    print("Train Accuracy:", sess.run(tr_accuracy))


    ## 검정 데이터에 대한 분류용-RBM의 정확도 계산 
    te_lab1 = np.zeros((len(teX),n_class)); te_lab1[:,0] = 1
    te_lab2 = np.zeros((len(teX),n_class)); te_lab2[:,1] = 1
    te_lab3 = np.zeros((len(teX),n_class)); te_lab3[:,2] = 1
    
    te_f1_xl = tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(teX,tf.float32), W_xh) \
    	       + tf.matmul(tf.cast(te_lab1,tf.float32), tf.transpose(W_hy)) + b_h),1)
    te_f2_xl = tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(teX,tf.float32), W_xh) \
    	       + tf.matmul(tf.cast(te_lab2,tf.float32), tf.transpose(W_hy)) + b_h),1)
    te_f3_xl = tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(teX,tf.float32), W_xh) \
    	       + tf.matmul(tf.cast(te_lab3,tf.float32), tf.transpose(W_hy)) + b_h),1)

    te_f_xl = b_y + tf.transpose([te_f1_xl,te_f2_xl,te_f3_xl])
    te_y_hat = tf.nn.softmax(te_f_xl)
    
    te_correct_pred = tf.equal(tf.argmax(te_y_hat,1), tf.argmax(teY, 1))
    te_accuracy = tf.reduce_mean(tf.cast(te_correct_pred, tf.float32))

    print("Test Accuracy :", sess.run(te_accuracy))
    
    sess.close()


#Train Accuracy: 0.9238095
#Test Accuracy : 0.95555556