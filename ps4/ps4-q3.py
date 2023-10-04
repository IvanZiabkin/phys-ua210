#Problem 3
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.special as sc
from gaussxw import gaussxw
#Part A
def H(n,x):
    H_m = 1
    H_n = 2*x
    for i in range(n-1): 
        H_p = 2*x*H_n - 2*n*H_m
        H_m = H_n
        H_n = H_p
    return H_n

def psi(n,x):
    return (H(n,x)*math.e**(-x**2/2))/np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi))

x = np.linspace(-4, 4, 1001)
for n in range(4):
    p = psi(n,x)
    txt = "Psi: n = " + str(n)
    plt.plot(x, p, label=txt)
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Psi(n,x)")
plt.show()

#Part B
x = np.linspace(-10, 10, 1001)
p = psi(30,x)
plt.figure()
plt.plot(x, p)
plt.xlabel("X")
plt.ylabel("Psi(n=30, x)")
plt.show()

#Part C

def I(N):
    s=0.0
    q = 0.0048
    x,w=gaussxw(N)
    xp = q*((x+1)/(1-x))
    for k in range(N):
        s+=w[k]+(abs(psi(5, xp[k])))**2*(2*q/((1-xp[k])**2))*(xp[k])**2
    return np.sqrt(s)
print(I(100))


#Part D
(x, w) = sc.roots_hermite(100)
def f(x):
    q = 0.0805
    xp = q*((x+1)/(1-x))
    return ((2*q/((1-xp)**2))*x**2*abs(H(5,x)/np.sqrt(2**n * math.factorial(5) * np.sqrt(np.pi))))**2
print(np.sqrt((f(x)*w).sum()))