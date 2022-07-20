import numpy as np
import matplotlib.pylab as plt

"""#계단 함수 구현하기

def step_function(x):
    if x>0:
        return 1
    else:
        return 0


#소수점 판별 + astype 메서드 사용

def step_function(x):
    y = x>0
    return y.astype(np.int)"""

"""x = np.array([-1.0,1.0,2.0])
print(x)

y = x > 0
print(y)"""

"""#astype 메서드는 원하는 자료형으로 변환시켜준다.
print(y.astype(np.int))
"""

#계단함수 그래프 그리기

def step_function(x):
    return np.array(x>0, dtype = np.int)
"""x = np.arange(-5.0,5.0,0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()"""


#시그모이드 함수 구현하기

def sigmoid(x):
    return 1 / (1+np.exp(-x))

"""x = np.arange(-5.0,5.0,0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()"""

#Relu 함수
def relu(x):
    return np.maximum(0,x)
"""x = np.arange(-5.0,5.0,0.1)
y = relu(x)
plt.plot(x,y)
plt.ylim(-0.5,5.5)
plt.show()
"""

#합성곱 신경망(순방향 신경망)
"""
import numpy as np

#입력층에서 1층으로 신호 전달
X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])
print("입력층에서 1층으로 신호 전달")
print(W1.shape)
print(X.shape)
print(B1.shape)

#전달된 신호
A1 = np.dot(X,W1) + B1

#1층에서 활성화 함수 사용 후 Z1에 저장
Z1 = sigmoid(A1)
print("전달된 신호 A1과 활성화 함수 사용한 Z1")
print(A1)
print(Z1)

#1층에서 2층으로 신호 전달

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
print("1층에서 2층으로 신호 전달 x=>z1")
print(Z1.shape)
print(W2.shape)
print(B2.shape)

#전달된 신호
A2 = np.dot(Z1,W2) +B2

#2층에서 활성화 함수 사용
Z2 = sigmoid(A2)
print("전달된 신호 A2과 활성화 함수 사용한 Z2")
print(A2)
print(Z2)

#2층에서 출력층으로 신호 전달
def identity_fucntion(x): #항등함수(: 입력을 그대로 출력함. 흐름을 통일하기 위해 사용)
     return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_fucntion(A3)
print("2층에서 출력층으로 신호 전달")
print(A3)
print("항등함수 적용")
print(Y)

#3층 신경망 구현 정리
def init_network(): # 가중치와 편향 초기화
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['B1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['B2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['B3'] = np.array([0.1, 0.2])

    return network

def forward(network, x): # 신호가 순방향으로 전달
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['B1'], network['B2'], network['B3']

    a1 = np.dot(x, W1) + B1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + B2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + B3
    y = identity_fucntion(a3)

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print("hi")
print(y)

#소프트맥스 함수

def softmax(a):
    c = np.max(a)
    #해당 출력층 뉴런 입력 신호 지수 함수
    exp_a = np.exp(a-c)
    #모든 입력 신호 지수 함수의 합 
    sum_exp_a = np.sum(exp_a) 
    y = exp_a / sum_exp_a
    return y"""