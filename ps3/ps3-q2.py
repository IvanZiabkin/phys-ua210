#Problem 2
import numpy as np
import matplotlib.pyplot as plt

values_manual = np.zeros(7)
arr = np.array([1,2,3,4,5,6,7])
for i in arr: #iterate for 7 points on graph
    op_count = 0
    A = np.ones((i*10,i*10), float) #actual matrix values are irrelevant
    B = np.ones((i*10,i*10), float)
    C_manual = np.zeros((i*10,i*10), float)
    C_dot = np.zeros(i*10, float)
    for j in range(i*10):
        for k in range(i*10):
            for l in range(i*10): 
                op_count += 1
                C_manual[j,k] = A[j,l] * B[l,k]
    values_manual[i-1] = op_count
    C_dot = np.dot(A, B)
plt.plot(10*arr, values_manual)
plt.xlabel("N")
plt.ylabel("# of Operations")
plt.show()
print("Last result, first for dot() and then for manual: ")
print(C_dot)
print(C_manual)