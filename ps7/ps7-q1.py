#Problem 1
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def f(x):
    return (x-0.3)**2 * np.exp(x)
#Main strategy and conditions -- parabolic method is pulled directly from lecture notes, rest is original

def brent(func=None, astart=None, bstart=None, cstart=None,
                       tol=1.e-5, maxiter=10000):
    runs = np.zeros(10000)
    a = astart
    b = bstart
    c = cstart
    bold = b + 2. * tol
    niter = 0
    while((np.abs(bold - b) > tol) & (niter < maxiter)):
        bold = b
        b = parabolic_step(f, a, b, c)
        if b<a or b>c: #condition 1
            b = golden(func, a, b, c)
            return (b)
        if(b < bold):
            c = bold
        else:
            a = bold
        runs[niter] = b
        if niter>1: #condition 2
            if b>runs[niter-2]:
                b = golden(func, a, b, c)
                return (b)
        niter = niter + 1
    return(b)

def parabolic_step(func=None, a=None, b=None, c=None):
    fa = f(a)
    fb = f(b)
    fc = f(c)
    denom = (b - a) * (fb - fc) - (b -c) * (fb - fa)
    numer = (b - a)**2 * (fb - fc) - (b -c)**2 * (fb - fa)
    # If singular, just return b 
    if(np.abs(denom) < 1.e-15):
        x = b
    else:
        x = b - 0.5 * numer / denom
    return(x)

#Golden Section -- pulled directly from lecture notes

def golden(func=None, astart=None, bstart=None, cstart=None, tol=1.e-5):
    gsection = (3. - np.sqrt(5)) / 2
    a = astart
    b = bstart
    c = cstart
    while(np.abs(c - a) > tol):
        if((b - a) > (c - b)):
            x = b
            b = b - gsection * (b - a)
        else:
            x = b + gsection * (c - b)
        step = np.array([b, x])
        fb = func(b)
        fx = func(x)
        if(fb < fx):
            c = x
        else:
            a = b
            b = x 
    return(b)

print(brent(f, 0., 1., 2.))
print(opt.brent(f))