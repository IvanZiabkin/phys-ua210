# import allowed modules
import numpy as np
import matplotlib.pyplot as plt

#x: -10 to 10 parameter
x = np.arange(-10, 10.01, 0.01)

#y: Gaussian w/ mean = 0 , std = 3
y = (1/(3*np.sqrt(2*np.pi)))*np.exp(-0.5*(x*x)/(3**2))
plt.plot(x, y)
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("gaussian.png")