import numpy as np

#AND GATE
def AND(x1,x2):
    w1, w2, theta = 0.5,0.5,0.7
    tmp = x1*w1 + x2*w2 

    if tmp<= theta:
        return 0 
    elif tmp > theta:
        return 1


#AND GATE + bias
def ANDBI(x1,x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0 
    else:
        return 1



#NAND GATE => ANDGATE에서 W와 B 값 다름
def NAND(x1,x2):
   x = np.array([x1,x2])
   w = np.array([-0.5,-0.5])
   b = 0.7
   tmp = np.sum(x*w) + b
   if tmp <= 0 :
        return 0
   else:
        return 1    

#OR GATE => ANDGATE에서 W와 B 값 다름
def OR(x1,x2):
   x = np.array([x1,x2])
   w = np.array([0.5,0.5])
   b = -0.2
   tmp = np.sum(x*w) + b
   if tmp <= 0 :
        return 0
   else:
        return 1    

#XOR GATE (다층 퍼셉트론)
def XOR(x1,x2):
    #1층
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    #2층
    y = AND(s1,s2)
    return y

print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))