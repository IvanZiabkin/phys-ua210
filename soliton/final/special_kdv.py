"""Compute alternative IC case for KdV"""

import numpy as np
import matplotlib.pyplot as plt

def compute_special():
    # Define the initial soliton wave function
    def initial_waveform(x, x0, sigma):
        u = 1/2 * (1-np.tanh((x-x0)/sigma))
        u[0] = 1
        u[1] = 1
        u[-1] = 0
        u[-2] = 0
        return u
    
    # Define parameters for the simulation
    L = 10        # Length of the spatial domain
    T = 2         # Total time to run the simulation
    dx = 0.1      # Space step
    dt = 0.01     # Time step
    Nx = int(L/dx) + 1  # Number of spatial points
    Nt = int(T/dt) + 1  # Number of time points
    epsilon = 1  # Wave speed for the advection equation
    mu = 0.001
    x0 = 5
    sigma = 1
    
    u = np.zeros((Nx, Nt))
    u[:, 0] = initial_waveform(np.linspace(0, L, Nx), x0, sigma)
    for j in range(0, Nt - 2):
        for i in range(1, Nx - 2):
            nonlinear_term = (u[i-1,j]+u[i, j] + u[i+1,j])/3 * epsilon*(u[i+1, j] - u[i-1, j]) / (2 * dx) - mu/(2*dx**3) * (u[i+2,j]-2*u[i+1,j]  + 2*u[i-1,j] - u[i-2,j])
            u[i, j+1] = u[i, j] - dt * nonlinear_term
            u[0] = 1
            u[1] = 1
            u[-1] = 0
            u[-2] = 0
    return u
def plot_special(u):
    L = 10        # Length of the spatial domain
    T = 2         # Total time to run the simulation
    dx = 0.1      # Space step
    dt = 0.01     # Time step
    Nx = int(L/dx) + 1  # Number of spatial points
    Nt = int(T/dt) + 1  # Number of time points
    plt.figure(figsize=(12, 6))
    plt.plot(np.linspace(0, L, Nx), u[:, 0], label='Initial Wave')
    for time_step in [int(Nt/4), int(Nt/2), int(3*Nt/4)]:
        plt.plot(np.linspace(0, L, Nx), u[:, time_step], label=f'State (t={time_step*dt})')
    plt.title('Evolution of a Hyperbolic Tangent Wave using KdV Equation')
    plt.xlabel('Space')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.show()