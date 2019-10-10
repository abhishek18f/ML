import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/abhishek/Documents/xyz/ML/retest/ex1data1.txt", delimiter=',')

x= data[:,0]
y= data[:,1]

#print(len(y))
plt.plot(x,y,'rx')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population in 10,000s')
plt.show()
