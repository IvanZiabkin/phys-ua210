#Problem 2
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import scipy.optimize as optimize

x, data = np.loadtxt("/survey.csv", skiprows=1, unpack=True, delimiter=",")

plt.scatter(x, data)

xpath = []

def squirrel(xk):
    global xpath
    xpath.append(np.array(xk))
    
def p(x, params):
    b0=params[0]
    b1 = params[1]
    return 1/(1+jnp.exp(-(b0+b1*x)))

LL_presum = np.zeros(100)
def negloglike(params, *args):
    epsilon = 10**(-6)
    x=args[0]
    data = args[1]
    prob = p(x, params)
    LL_presum = data*jnp.log(prob/(1-prob+epsilon)) + jnp.log(1-prob+epsilon)
    out = -np.sum(LL_presum)
    return out

negloglike_grad = jax.grad(negloglike)
pst = np.array([2., 2.])
xpath = [pst]
r = optimize.minimize(negloglike, pst, args=(x, data), method='Nelder-Mead', tol=1e-6, callback=squirrel)
xpath = np.array(xpath)
print("Beta Values: ")
print(r.x)
x=np.arange(0.1, 100.1, 0.1)
pset = np.zeros(1000)
for i in range(1000):
    pset[i] = p(x[i], np.array(r.x))
plt.plot(x, pset)
plt.xlabel("Age")
plt.ylabel("Yes Answer Probability")
x, data = np.loadtxt("/survey.csv", skiprows=1, unpack=True, delimiter=",")
def hessian(f):
  return jax.jacfwd(jax.grad(f))
h = hessian(negloglike)
hmat = np.array(h((r.x), x, data))
covar = np.linalg.inv(hmat)
print("Beta Covariance: ")
print(covar)
print("Standard Deviations: ")
print(np.sqrt(np.diag(covar)))
sigma2 = np.diag(covar)
corr = covar
for i in np.arange(2):
    for j in np.arange(2):
        corr[i, j] = covar[i, j] / np.sqrt(sigma2[i] * sigma2[j])
print("Correlation: ")
print(corr)