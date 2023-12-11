"""Compute Advection Eq case"""
import numpy as np
import matplotlib.pyplot as plt

def compute_advect():
    # Define the initial Gaussian wave packet function
    def gaussian_wave(x, center, width):
        u = np.exp(-((x - center)**2) / (2 * width**2))
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
    c = 1       # Wave speed for the advection equation
    
    # Initialize the solution grid with zeros
    u = np.zeros((Nx, Nt))
    
    # Set the initial condition as a Gaussian wave packet
    center = L / 2  # Center of the initial Gaussian
    width = 1.0     # Width of the initial Gaussian
    u[:, 0] = gaussian_wave(np.linspace(0, L, Nx), center, width)
    
    # Apply zero boundary conditions
    u[0, :] = 0
    u[-1, :] = 0
    
    # Run the simulation using the FTCS scheme for the advection equation
    for j in range(0, Nt - 1):
        for i in range(1, Nx - 1):
            # Forward time, centered space scheme for linear advection
            u[i, j+1] = u[i, j] - c * dt / (2 * dx) * (u[i+1, j] - u[i-1, j])
    return u
  
def plot_advect(u):  
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
    plt.title('Advection of a Gaussian Wave using FTCS Scheme')
    plt.xlabel('Space')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.show()