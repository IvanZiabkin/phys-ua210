import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for the finite difference scheme
dx = 0.1      # Space step
dt = 0.01     # Time step
L = 10        # Length of the spatial domain
T = 1         # Total time
c = 1         # Wave speed (used in the CFL condition)
C = 0.5       # CFL constant

# Compute the number of points in the spatial and temporal domains
Nx = int(L/dx) + 1  # Number of spatial points
Nt = int(T/dt) + 1  # Number of time points

# Initialize the solution grid with zeros
u = np.zeros((Nx, Nt))

# Set initial conditions (assuming some initial function I(x))
def I(x):
    # Initial condition function, can be any function of x
    return np.sin(np.pi * x / L)

u[:, 0] = I(np.linspace(0, L, Nx))  # Apply the initial condition to the first time step

# Apply the boundary conditions
u[0, :] = u[0, 0]    # u(x=0, t) = u(x=0, t=0) for all t
u[-1, :] = u[-1, 0]  # u(x=L, t) = u(x=L, t=0) for all t

# Finite difference scheme
for j in range(0, Nt - 1):
    for i in range(1, Nx - 1):
        # Implementing the evolution scheme
        u[i, j+1] = (u[i-1, j] + u[i, j] + u[i+1, j]) / 3


# We consider the case where the wave speed c is constant
if dt > C * (dx**2 / c):
    print(f"Stability condition not met! Reduce dt below {C * (dx**2 / c)}.")


# Plot the initial condition
plt.plot(np.linspace(0, L, Nx), u[:, 0], label='Initial Condition')

# Plot the final state
plt.plot(np.linspace(0, L, Nx), u[:, -1], label='Final State (t=1)')

plt.title('Evolution of the wave')
plt.xlabel('Space')
plt.ylabel('u(x,t)')
plt.legend()
plt.show()