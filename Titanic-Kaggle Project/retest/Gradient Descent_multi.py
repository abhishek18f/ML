import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/abhishek/ex1data2.txt", delimiter=',')
m=len(data)
p=np.size(data,1)
alpha = 0.01
iterations = 1500
theta = np.zeros((p,1))

max= np.amax(data, axis=0)
min= np.amin(data, axis=0)

sum = np.sum(data,axis=0)

for i in range(0,p):
    for j in range(0, m):
        data[j,i] -= sum[i]
        data[j,i] /= (max[i] - min[i])

y = data[:,-1:]
X = np.ones((m,p))
X[:,1:] = data[:,0:p-1]

for i in range(0,iterations):
     cost = np.dot(X,theta) - y 
     G = np.dot(X.T,cost)/m
     theta -= alpha*G


print("theta=")
print(theta)


