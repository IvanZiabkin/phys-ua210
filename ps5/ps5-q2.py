#Problem 2
import numpy as np
import matplotlib.pyplot as plt
import io


#Part A
s = open('/Users/ziabkinfamily/Downloads/signal.dat').read().replace('|',' ') 
data = np.loadtxt(io.StringIO(s),skiprows=1,dtype=str)
t = np.zeros(1000)
sig = np.zeros(1000)

for i in range(1000):
    t[i] = data[i,0]
    sig[i] = data[i,1]
plt.scatter(t,sig)
plt.xlabel("Time")
plt.ylabel("Signal")
plt.figure()

#Part B -- 

A = np.zeros((len(t), 4))
A[:, 0] = 1.
A[:, 1] = t
A[:, 2] = t**2
A[:, 3] = t**3
(u, w, vt) = np.linalg.svd(A, full_matrices=False)
w[3] = 10**20 #not super sure why but absent this w[3] = 0 which causes runtime error below
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(sig)
ym = A.dot(c) 
plt.scatter(t,sig, c='b')
plt.scatter(t,ym, c='k')
plt.xlabel("Time")
plt.ylabel("Signal")
plt.figure()

#Part C -- 
c = np.corrcoef(sig, ym)
print(c)

#Part D -- 
A = np.zeros((len(t), 8))
A[:, 0] = 1.
A[:, 1] = t
A[:, 2] = t**2
A[:, 3] = t**3
A[:, 4] = t**4
A[:, 5] = t**5
A[:, 6] = t**6
A[:, 7] = t**7
(u, w, vt) = np.linalg.svd(A, full_matrices=False)
w[7] = 10**60
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(sig)
ym = A.dot(c) 
plt.scatter(t,sig, c='b')
plt.scatter(t,ym, c='k')
plt.xlabel("Time")
plt.ylabel("Signal")

del(A, u, w, vt, ainv, c)

#Part E -- 
A = np.zeros((len(t), 10))
plt.figure()
A[:, 0] = 1 #having a 0 here makes everything turn out wrong, so I'm keeping a 1
A[:, 1] = np.cos(t/2)+np.sin(t/2)
A[:, 2] = np.cos(t)+np.sin(t)
A[:, 3] = np.cos(2*t)+np.sin(2*t)
A[:, 4] = np.cos(3*t)+np.sin(3*t)
A[:, 5] = np.cos(4*t)+np.sin(4*t)
A[:, 6] = np.cos(5*t)+np.sin(5*t)
A[:, 7] = np.cos(6*t)+np.sin(6*t)
A[:, 8] = np.cos(7*t)+np.sin(7*t)
A[:, 9] = np.cos(8*t)+np.sin(8*t)
(u, w, vt) = np.linalg.svd(A, full_matrices=False)
w[3]=0.6
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(sig)
ym = A.dot(c) 
plt.scatter(t,sig, c='b')
plt.scatter(t,ym, c='k')
plt.xlabel("Time")
plt.ylabel("Signal")

