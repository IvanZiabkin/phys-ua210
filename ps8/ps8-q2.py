#Problem 2
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def eqs(trange, initials):
    sigma= 10
    r=28
    b=8/3
    x = initials[0]
    y = initials[1]
    z = initials[2]
    diffx = (y-x)*sigma
    diffy = r*x-y-x*z
    diffz = x*y-b*z
    output = [diffx, diffy, diffz]
    return output

trange = [0, 50]
initials = [0,1,0]

solution = solve_ivp(eqs, trange, initials)

plt.plot(solution.t, solution.y[1,:])
plt.title("Lorenz Eq., Y Solutions")
plt.xlabel("t")
plt.ylabel("y")
plt.figure()
plt.plot(solution.t, solution.y[2,:])
plt.title("Lorenz Eq., Z Solutions")
plt.xlabel("t")
plt.ylabel("z")