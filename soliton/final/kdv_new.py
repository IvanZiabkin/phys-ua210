"""Computes basic KDV case"""
import numpy as np
import matplotlib.pyplot as plt

def compute_kdv(mu):
    # Define the initial soliton wave
    def initial_waveform(x, c, xi):
        u = c/2 * 1/((np.cosh(0.5*np.sqrt(c)*(x-xi)))**2)
        u[0] = 0
        u[-1] = 0
        return u
    
    # Define parameters for the simulation
    L = 10        # Length of the spatial domain
    T = 2         # Total time to run the simulation
    dx = 0.1      # Space step
    dt = 0.01     # Time step
    Nx = int(L/dx) + 1  # Number of spatial points
    Nt = int(T/dt) + 1  # Number of time points
    epsilon = 1  # Wave speed for the advection equation
    c = 1
    xi = 5
    
    u = np.zeros((Nx, Nt))
    u[:, 0] = initial_waveform(np.linspace(0, L, Nx), c, xi)
    for j in range(0, Nt - 2):
        for i in range(1, Nx - 2):
            nonlinear_term = (u[i-1,j]+u[i, j] + u[i+1,j])/3 * epsilon*(u[i+1, j] - u[i-1, j]) / (2 * dx) - mu/(2*dx**3) * (u[i+2,j]-2*u[i+1,j]  + 2*u[i-1,j] - u[i-2,j])
            u[i, j+1] = u[i, j] - dt * nonlinear_term
            u[0,j] = 0 # reset boundary conditions
            u[1,j] = 0
            u[-1,j] = 0
            u[-2,j] = 0
    return u
def plot_kdv(u):
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
    plt.title('Evolution of a Soliton Wave using KdV Equation')
    plt.xlabel('Space')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.show()