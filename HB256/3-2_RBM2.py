"""
실수 입력 RBM과 이진 입력 RBM을 iris 데이터에 적용하여 2차원 산점도를 그리는 예제
"""
# 필요한 라이브러리를 불러들임
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 실수 입력 RBM과 이진 입력 RBM을 위한 시그모이드 함수
def sigmoid(x):
    return 1/(1+np.exp(-x))


# CD-1을 이용한 실수 입력 RBM 알고리즘
def Gau_Ber_RBM(data, _w, _a, _b, learning_rate, phase = 'training'):
    
    if phase == 'training':
        h0_p = sigmoid(np.matmul(data, _w) + _b)
        h0 = np.random.binomial(1, h0_p)
        
        x_mu = np.matmul(h0, np.transpose(_w)) + _a
        x = np.random.normal(x_mu, 1)
        
        h1_p = sigmoid(np.matmul(x, _w) + _b)
        h1 = np.random.binomial(1, h1_p)
        
        w = _w + learning_rate*(np.matmul(np.transpose(data), h0) - np.matmul(np.transpose(x), h1))/len(data)
        a = _a + learning_rate*(np.mean(data - x, 0))
        b = _b + learning_rate*(np.mean(h0 - h1, 0))
        
        return w, a, b
    
    elif phase == 'loss':
        h0_p = sigmoid(np.matmul(data, _w) + _b)
        h0 = np.round(h0_p)
        x = np.matmul(h0, np.transpose(_w)) + _a
        reconstruction_error = np.mean((data-x)**2)
        return reconstruction_error
    
    else:
        print('phase must be training or loss')


# CD-1을 이용한 이진 입력 RBM 알고리즘
def Ber_Ber_RBM(data, _w, _a, _b, learning_rate, phase = 'training'):
    
    if phase=='training':
        h0_p = sigmoid(np.matmul(data, _w) + _b)
        h0 = np.random.binomial(1, h0_p)
        
        x_p = sigmoid(np.matmul(h0, np.transpose(_w)) + _a)
        x = np.random.binomial(1, x_p)
        
        h1_p = sigmoid(np.matmul(x, _w) + _b)
        h1 = np.random.binomial(1, h1_p)
        
        w = _w + learning_rate*(np.matmul(np.transpose(data), h0) - np.matmul(np.transpose(x), h1))/len(data)
        a = _a + learning_rate*(np.mean(data - x, 0))
        b = _b + learning_rate*(np.mean(h0 - h1, 0))
        
        return w, a, b
    
    elif phase=='loss':
        h0_p = sigmoid(np.matmul(data, _w) + _b)
        h0 = np.round(h0_p)
        x_p = sigmoid(np.matmul(h0, np.transpose(_w)) + _a)
        x = np.round(x_p)      
        reconstruction_error = np.mean((data-x)**2)
        return reconstruction_error
    
    else:
        print('phase must be training or loss')


# ---------------------------------------------------------------------------
# iris 데이터 불러오기
url='https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
iris = pd.read_csv(url)
iris.head()        
        
n = iris.shape[0]
irisX = np.array(iris.iloc[:,:4])
irisY = iris.iloc[:,4]


# 실수 입력 RBM을 위한 입력 데이터의 표준화(standardization) 
moments = np.mean(irisX, 0), np.var(irisX, 0)
standardized_irisX = (irisX-moments[0])/np.sqrt(moments[1])  


# 이진 입력 RBM을 위한 입력 데이터의 정규화(normalization) 
minmax = np.amin(irisX, 0), np.amax(irisX, 0)
normalized_irisX = (irisX-minmax[0])/(minmax[1]-minmax[0])  


# 입력노드 4개 은닉노드 2개의 RBM 구축
_w = np.random.normal(size = [4,2], scale = 0.1)
_a = np.zeros([4])
_b = np.zeros([2])

tr_h = sigmoid(np.matmul(normalized_irisX, _w) + _b)


# 학습률, 에포크 회수 및 미니배치 크기 설정
learning_rate = 5*1e-3
max_epoch = 1500
batch_size = 5


# ---------------------------------------------------------------------------
# 실수 입력 RBM 학습
for learning_epoch in range(max_epoch):
    rannum = np.random.permutation(len(standardized_irisX))
    num_batch = int(len(standardized_irisX)/batch_size)

    for it in range(num_batch):
        batch_X = standardized_irisX[rannum[it*batch_size:(it+1)*batch_size]]
        w, a, b = Gau_Ber_RBM(batch_X, _w, _a, _b, learning_rate, phase = 'training')

    if (learning_epoch+1)%100==0:
        print(Gau_Ber_RBM(standardized_irisX, w, a, b, learning_rate, phase = 'loss'))


real_h = sigmoid(np.matmul(standardized_irisX, w) + b)      

# 실수 입력 RBM 결과 산점도
plt.scatter(real_h[np.where(irisY=='setosa')[0], 0], real_h[np.where(irisY=='setosa')[0], 1], color = 'red')
plt.scatter(real_h[np.where(irisY=='virginica')[0], 0], real_h[np.where(irisY=='virginica')[0], 1], color = 'blue')
plt.scatter(real_h[np.where(irisY=='versicolor')[0], 0], real_h[np.where(irisY=='versicolor')[0], 1], color = 'black')
plt.show()


# ---------------------------------------------------------------------------
# 이진 입력 RBM 학습
for learning_epoch in range(max_epoch):
    rannum = np.random.permutation(len(normalized_irisX))
    num_batch = int(len(normalized_irisX)/batch_size)

    for it in range(num_batch):
        batch_X = normalized_irisX[rannum[it*batch_size:(it+1)*batch_size]]
        w, a, b = Ber_Ber_RBM(batch_X, _w, _a, _b, learning_rate, phase = 'training')

    if (learning_epoch+1)%100==0:
        print(Ber_Ber_RBM(normalized_irisX, w, a, b, learning_rate, phase = 'loss'))


binary_h = sigmoid(np.matmul(normalized_irisX, w) + b)      

# 이진 입력 RBM 결과 산점도
plt.scatter(binary_h[np.where(irisY=='setosa')[0], 0], binary_h[np.where(irisY=='setosa')[0], 1], color = 'red')
plt.scatter(binary_h[np.where(irisY=='virginica')[0], 0], binary_h[np.where(irisY=='virginica')[0], 1], color = 'blue')
plt.scatter(binary_h[np.where(irisY=='versicolor')[0], 0], binary_h[np.where(irisY=='versicolor')[0], 1], color = 'black')
plt.show()
