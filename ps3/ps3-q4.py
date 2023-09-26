#Problem 4
import numpy as np
import matplotlib.pyplot as plt
N=1000
r = np.random.uniform(0, 1, N)
tau = 3.053*60
decay_time = -tau/np.log(2) * np.log(1-r)
arr = np.zeros(1000)
secs = range(1000)
for t in secs:
    remaining = t<decay_time
    arr[t] = sum(remaining)
plt.plot(secs, arr)
plt.xlabel("Time (s)")
plt.ylabel("Number of Remaining Atoms")