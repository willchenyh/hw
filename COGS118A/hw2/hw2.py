import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

data = sio.loadmat('data.mat')
x = data['x'].reshape([-1, 1])
y = data['y'].reshape([-1, 1])

plt.plot(x, y)
plt.grid()


X = np.hstack((np.ones((len(x),1)),np.power(x,1)))

X_t = X.transpose((1,0))
sol = np.dot(np.linalg.inv(np.dot(X_t,X)),np.dot(X_t,y))

plt.hold(True)
plt.plot(x,sol[0]+sol[1]*x)

plt.title('Least square line fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
