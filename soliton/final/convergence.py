import numpy as np
import matplotlib.pyplot as plt

def conv():
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
    center = L / 2  # Center of the initial Gaussian
    width = 1.0     # Width of the initial Gaussian
    
    Nx2 = 2 * Nx  # Double the number of spatial points
    Nx4 = 4 * Nx  # Quadruple the number of spatial points
    
    # Initialize the solution grids for the new resolutions
    u_Nx2 = np.zeros((Nx2, Nt))
    u_Nx4 = np.zeros((Nx4, Nt))
    
    # Set the initial conditions for the new grids
    x_values_Nx2 = np.linspace(0, L, Nx2)
    x_values_Nx4 = np.linspace(0, L, Nx4)
    u_Nx2[:, 0] = gaussian_wave(x_values_Nx2, center, width)
    u_Nx4[:, 0] = gaussian_wave(x_values_Nx4, center, width)
    
    # Apply zero boundary conditions for the new grids
    u_Nx2[0, :] = 0; u_Nx2[-1, :] = 0
    u_Nx4[0, :] = 0; u_Nx4[-1, :] = 0
    
    # Run the simulation for each new grid resolution using the FTCS scheme
    # We will assume the wave speed c and other parameters remain the same
    for u_current, dx_current in [(u_Nx2, L/(Nx2-1)), (u_Nx4, L/(Nx4-1))]:
        for j in range(0, Nt - 1):
            for i in range(1, len(u_current) - 1):
                u_current[i, j+1] = u_current[i, j] - c * dt / (2 * dx_current) * (u_current[i+1, j] - u_current[i-1, j])
                
                # Plot the final states from each grid resolution to compare
    plt.figure(figsize=(14, 7))
                
                # Plot for original grid resolution
                
    u = np.zeros((Nx, Nt))
    u[:, 0] = gaussian_wave(np.linspace(0, L, Nx), center, width)
    u[0, :] = 0
    u[-1, :] = 0
        
        # Run the simulation using the FTCS scheme for the advection equation
    for j in range(0, Nt - 1):
        for i in range(1, Nx - 1):
            # Forward time, centered space scheme for linear advection
            u[i, j+1] = u[i, j] - c * dt / (2 * dx) * (u[i+1, j] - u[i-1, j])
                
    plt.plot(np.linspace(0, L, Nx), u[:, -1], label='Final State (Original Nx)', linewidth=2)
                
    # Plot for doubled grid resolution
    plt.plot(x_values_Nx2, u_Nx2[:, -1], label='Final State (Nx2)', linestyle='--', linewidth=2)
    
    # Plot for quadrupled grid resolution
    plt.plot(x_values_Nx4, u_Nx4[:, -1], label='Final State (Nx4)', linestyle=':', linewidth=2)
    
    plt.title('Convergence Test with Increased Grid Points')
    plt.xlabel('Space')
    plt.ylabel('u(x, t)')
    plt.legend()
    plt.grid(True)
    plt.show()

