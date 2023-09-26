#Problem 3
import numpy as np
import matplotlib.pyplot as plt
N = 10000
bi213 = np.ones(N)
tl209 = np.zeros(N)
pb209 = np.zeros(N)
bi209 = np.zeros(N)
def prob(tau): 
    return (1-2**(-1/tau))
prob_bi213 = prob(46*60)
prob_tl209 = prob(2.2*60)
prob_pb209 = prob(3.3*60)
bi1 = np.zeros(20000)
bi2 = np.zeros(20000)
tl= np.zeros(20000)
pb= np.zeros(20000)
for i in range(20000):
    #establish probabilities
    bi213_rng_1 = np.random.uniform(0,1,N)
    bi213_rng_2 = np.random.uniform(0,1,N)
    tl209_rng = np.random.uniform(0,1,N)
    pb209_rng = np.random.uniform(0,1,N)
    #last set of decay, a
    set_bi = pb209_rng<prob_pb209
    bi209 += set_bi*pb209
    f=bi209>1
    bi209=bi209-f
    pb209 -= set_bi*pb209
    #middle set of decay, b
    set_pb_tl = tl209_rng<prob_tl209
    pb209+=set_pb_tl*tl209
    tl209 -= set_pb_tl*tl209
    #first set of decay,c
    set_tl = np.ones(N)
    bi213_decay = bi213_rng_1<prob_bi213
    set_pb_bi = bi213_rng_2<(0.9791)
    set_tl = set_tl - set_pb_bi
    pb209 += set_pb_bi*bi213_decay
    tl209 += set_tl*bi213_decay
    bi213 -= bi213_decay*bi213
    #total number
    bi1[i] = np.sum(bi213)
    bi2[i] = np.sum(bi209)
    tl[i] = np.sum(tl209)
    pb[i] = np.sum(pb209)
x=range(20000)
plt.plot(x, bi1, label="213Bi")
plt.plot(x, bi2, label="209Bi")
plt.plot(x, tl, label="209Tl")
plt.plot(x, pb, label="209Pb")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Number of Atoms")
plt.show()