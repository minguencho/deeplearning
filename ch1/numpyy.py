import numpy as np
x = np.array([1.0,2.0,3.0])
y = np.array([2.0,4.0,6.0])
print(x+y)
print(x-y)
print(x/y)

#broadcast
print(x/2.0)

#n차원 배열
A =np.array([[1,2],[3,4]])
print(A)
print(A.shape)
print(A.dtype)
B =np.array([[3,0],[0,6]])
print(A+B)
print(A*B)
print(A*10)

#행렬 확대 계산  = broadcast
C =np.array([[10,20]])
print(A*C)

#원소 접근
X = np.array([[1,2],[3,4],[5,6]])
print(X)
print(X[0,1])
for row in X:
    print(row)
X = X.flatten()
print(X)
print(X>15)
print(X[X>4])
