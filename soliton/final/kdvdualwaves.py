import numpy as np
import matplotlib.pyplot as plt
def k(c, xi):
    #KdV

    # Define the initial soliton wave function
    def initial_waveform(x, c, xi):
        u = c/2 * 1/((np.cosh(0.5*np.sqrt(c)*(x-xi)))**2)
        u[0] = 0
        u[-1] = 0
        return u

    # Define parameters for the simulation
    L = 10        # Length of the spatial domain
    T = 1.5       # Total time to run the simulation
    dx = 0.1      # Space step
    dt = 0.01     # Time step
    Nx = int(L/dx) + 1  # Number of spatial points
    Nt = int(T/dt) + 1  # Number of time points
    epsilon = 2  # Wave speed for the advection equation
    mu = 0.001

    u = np.zeros((Nx, Nt))
    u[:, 0] = initial_waveform(np.linspace(0, L, Nx), c, xi)
    for j in range(0, Nt - 2):
        for i in range(1, Nx - 2):
            nonlinear_term = (u[i-1,j]+u[i, j] + u[i+1,j])/3 * epsilon*(u[i+1, j] - u[i-1, j]) / (2 * dx) - mu/(2*dx**3) * (u[i+2,j]-2*u[i+1,j]  + 2*u[i-1,j] - u[i-2,j])
            u[i, j+1] = u[i, j] - dt * nonlinear_term
    
    plt.plot(np.linspace(0, L, Nx), u[:, 0], label=f'Initial Wave {xi}')
    for time_step in [int(Nt/4), int(Nt/2), int(3*Nt/4)]:
        plt.plot(np.linspace(0, L, Nx), u[:, time_step])
    plt.title('Evolution of a Soliton Wave using KdV Equation')
    plt.xlabel('Space')
    plt.ylabel('u(x,t)')
    plt.legend()

def dual():
    k(1,5.5)
    k(1.5,4.5)