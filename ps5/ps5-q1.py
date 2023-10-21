#Problem 1
import numpy as np
import matplotlib.pyplot as plt

def preint(x,a):  #part A
    return x**(a-1) * np.e**(-x)

x  = np.linspace(0,5,1001)
for a in np.linspace(2,4, 3):
    g = np.array(np.zeros(1001))
    for i in np.linspace(0,1000,1001):
        gam = preint(x[int(i)],a)
        g[int(i)] = gam
    plt.plot(np.linspace(0,5,1001),g)
    plt.xlabel("X")
    plt.ylabel("Gamma")
    
#Part B -- in report doc

#Part C -- report doc

#Part D -- 
def preint_real(x,a):  
    return np.e**((-x+(a-1)*np.log(x)))

#Part E -- using Simpson's Rule

def gamma(a):
    dz = 0.5*0.001
    dG = np.zeros(1000)
    for i in np.linspace(1,1000,1000):
        z = int(i)/1000
        if z==1:
            continue
        if i==1:
            dG[int(i-1.0)] = (1/3) * preint_real(0.5*(z/(1-z)),a) / ((1-z)**2) #0.5s on z terms comes from parts b/c
        elif i%2==0.0:
            dG[int(i-1.0)] = (4/3) * preint_real(0.5*(z/(1-z)),a) / ((1-z)**2)
        else:
            dG[int(i-1.0)] = (2/3) * preint_real(0.5*(z/(1-z)),a) / ((1-z)**2)
    return (dz*sum(dG))

print(gamma(1.5))

print(gamma(3))
print(gamma(6))
print(gamma(10))