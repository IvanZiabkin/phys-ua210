#Problem 2
import numpy as np
import matplotlib.pyplot as plt
from gaussxw import gaussxwab
N = 20
m = 1
def f(x,a):
    return np.sqrt(8*m)/(np.sqrt(a**4-x**4))

def T(a):
    x,w = gaussxwab(N, 0, a)
    s=0.0
    for k in range(N): 
        s+=w[k]*f(x[k], a)
    return s
vals = np.zeros(101)
a = np.linspace(0,2,101)
for i in range(101):
    vals[i] = T(a[i])
x=range(101)
plt.plot(x, vals)
plt.xlabel("Amplitude")
plt.ylabel("Period")
plt.show()