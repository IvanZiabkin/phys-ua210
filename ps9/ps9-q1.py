#HW 9 Problem 1
import numpy as np
import matplotlib.pyplot as plt
import banded as band

global L, M, hbar, h

L = 10**(-8)
M = 9.109*10**(-31)
hbar = 1.055*10**(-34)
h = 10**(-18)
x_0 = L/2
sigma = 10**(-10)
kappa = 5*10**(10)


#Part A
def psi_0(x, x_0, sigma, kappa):
    return (np.exp(-(x-x_0)**2 / (2 * sigma**2)) * np.exp(1j*kappa*x))

def initletters(letters, a):
    letters[0] = 1+h*hbar*1j/(2*M*a**2)
    letters[1] = -h*hbar*1j/(4*M*a**2)
    letters[2] = 1-h*hbar*1j/(2*M*a**2)
    letters[3] = h*hbar*1j/(4*M*a**2)
    return letters

# def initA(N,a):
#     N = N+1
#     A = np.zeros((N,N), dtype=np.complex_)
#     letters = initletters(np.zeros(4, dtype=np.complex_), a)
#     for i in range(N):
#         A[i,i] = letters[0]
#         if i+2<=N:
#             A[i, i+1]= letters[1]
#             A[i+1, i] = letters[1]
#     return A

N = 1000
a = L/N
x = np.linspace(0, L, N+1)
x[1000] = 0
psi = psi_0(x, x_0, sigma, kappa)
letters=initletters(np.zeros(4, dtype=np.complex_), a)
v = np.zeros(N+1, dtype=np.complex_)
A = np.zeros((3, N+1), dtype=np.complex_)
A[0] = np.ones((1, N+1)) * letters[1]
A[1] = np.ones((1, N+1)) * letters[0]
A[2] = np.ones((1, N+1)) * letters[1]


for j in range(1,1000):
    for i in range(1, N):
        a = L/N
        letters=initletters(np.zeros(4, dtype=np.complex_), a)
        v[i] = letters[2]*psi[i]+letters[3]*(psi[i+1]+psi[i-1])
    solution = band.banded(A, v, 1,1)
    psi = solution
    psi[0] = 0
    psi[1000] = 0
    if j%200==0:
        f=str(j)
        f = "Time (h): " +f
        plt.plot(x, np.real(psi), label=f)
        plt.title("Schrodinger Wavefunction")
        plt.xlabel("Position (m)")
        plt.ylabel("Probability Density")
        plt.legend(loc=2)


# def initB(N,a):
#     N = N-1
#     B = np.zeros((N,N), dtype=np.complex_)
#     letters = initletters(np.zeros(4, dtype=np.complex_), a)
#     for i in range(N):
#         B[i,i] = letters[2]
#         if i+2<=N:
#             B[i, i+1]= letters[3]
#             B[i+1, i] = letters[3]
#     return B

#note to self: start w/ psi_0. Plug into eq for v_i. Then solve Ax = v for x using banded
# (v = Bpsi so we're basically doing A psi(t+1) = B psi(t))
# note to self: do i from 1:N. 0 and N+1 are BC... will be factored in during v_i
