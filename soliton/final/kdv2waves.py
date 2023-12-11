#KdV -- 2 waves
import numpy as np
import matplotlib.pyplot as plt

def compute_kdv2():
    # Define the initial soliton wave
    def initial_waveform(x):
        c1 = 1
        xi1 = 5.5
        c2 = 1.5
        xi2 = 4.5
        u1 = c1/2 * 1/((np.cosh(0.5*np.sqrt(c1)*(x-xi1)))**2)
        u2 = c2/2 * 1/((np.cosh(0.5*np.sqrt(c2)*(x-xi2)))**2)
        u1[0] = 0
        u1[-1] = 0
        u2[0] = 0
        u2[-1] = 0
        u = u1+u2
        return u

    def single_wave(x, c, xi):
        u = c/2 * 1/((np.cosh(0.5*np.sqrt(c)*(x-xi)))**2)
        u[0] = 0
        u[-1] = 0
        return u

    # Define parameters for the simulation
    L = 10      # Length of the spatial domain
    T =  1.5  # Total time to run the simulation
    dx = 0.1   # Space step
    dt = 0.01   # Time step
    Nx = int(L/dx) + 1  # Number of spatial points
    Nt = int(T/dt) + 1  # Number of time points
    epsilon = 2  # Wave speed for the advection equation
    mu = 0.001

    u = np.zeros((Nx, Nt))
    u[:, 0] = initial_waveform(np.linspace(0, L, Nx))
    for j in range(0, Nt - 2):
        for i in range(1, Nx - 2):
            nonlinear_term = (u[i-1,j]+u[i, j] + u[i+1,j])/3 * epsilon*(u[i+1, j] - u[i-1, j]) / (2 * dx) - mu/(2*dx**3) * (u[i+2,j]-2*u[i+1,j]  + 2*u[i-1,j] - u[i-2,j])
            u[i, j+1] = u[i, j] - dt * nonlinear_term
    return u

def plot_kdv2(u):
    L = 10      # Length of the spatial domain
    T =  1.5  # Total time to run the simulation
    dx = 0.1   # Space step
    dt = 0.01   # Time step
    Nx = int(L/dx) + 1  # Number of spatial points
    Nt = int(T/dt) + 1  # Number of time points
    plt.figure(figsize=(12, 6))
    plt.plot(np.linspace(0, L, Nx), u[:, 0], label='Initial Wave')
    for time_step in [int(Nt/4), int(Nt/2), int(3*Nt/4)]:
        plt.plot(np.linspace(0, L, Nx), u[:, time_step])
    plt.title('Evolution of a Soliton Wave using KdV Equation')
    plt.xlabel('Space')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.show()