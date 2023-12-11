"""Computes Burger Data"""
#Burgers
import numpy as np
import matplotlib.pyplot as plt
def compute_burgers():
# Define the initial Gaussian wave packet function
    def gaussian_wave(x, center, width):
        return np.exp(-((x - center)**2) / (2 * width**2))

# Define parameters for the simulation
    L = 10        # Length of the spatial domain
    T = 2         # Total time to run the simulation
    dx = 0.1      # Space step
    dt = 0.01     # Time step
    Nx = int(L/dx) + 1  # Number of spatial points
    Nt = int(T/dt) + 1  # Number of time points
    c = 1       # Wave speed for the advection equation
    center = L / 2  # Center of the initial Gaussian
    width = 1.0     

    u = np.zeros((Nx, Nt))
    u[:, 0] = gaussian_wave(np.linspace(0, L, Nx), center, width)  
# Apply zero boundary conditions
    u[0, :] = 0
    u[-1, :] = 0

    for j in range(0, Nt - 1):
        for i in range(1, Nx - 1):
            nonlinear_term = (u[i-1,j]+u[i, j] + u[i+1,j])/3 * (u[i+1, j] - u[i-1, j]) / (2 * dx)
            u[i, j+1] = u[i, j] - dt * nonlinear_term
    return u
def plot_burgers(u):    
    L = 10        # Length of the spatial domain
    T = 2         # Total time to run the simulation
    dx = 0.1      # Space step
    dt = 0.01     # Time step
    Nx = int(L/dx) + 1  # Number of spatial points
    Nt = int(T/dt) + 1  # Number of time points
    plt.figure(figsize=(12, 6))
    plt.plot(np.linspace(0, L, Nx), u[:, 0], label='Initial Gaussian')
    for time_step in [int(Nt/4), int(Nt/2), int(3*Nt/4), Nt-1]:
        plt.plot(np.linspace(0, L, Nx), u[:, time_step], label=f'State (t={time_step*dt})')
        plt.title('Evolution of a Gaussian Wave using Burgers\' Equation')
        plt.xlabel('Space')
        plt.ylabel('u(x,t)')
    plt.legend()
    plt.show()