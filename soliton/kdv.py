#Leapfrog Method -- KdV

import numpy as np
import matplotlib.pyplot as plt
    
    # Parameters
L = 20.0   # Spatial domain length
T = 4.0    # Temporal domain length
Nx = 400   # Number of spatial grid points
Nt = 4000  # Number of time steps
dx = L / Nx # x step
dt = T / Nt # t step

mu = 1
c = 0.000000001

    # Initial condition
def initial_waveform(x):
    c = 2
    xi = 10
    u = c/2 * 1/((np.cosh(0.5*np.sqrt(c)*(x-xi)))**2)
    return u

    # Spatial grid
x = np.linspace(0, L, Nx)
u = initial_waveform(x)  # Set the initial waveform here
    # Time-stepping loop for the linear advection equation (µ = 0)
u_old = np.copy(u)
for n in range(Nt):
    u_new = np.copy(u)
    for j in range(1, Nx - 2):
        u_new[j] = u_old[j] - c * dt /(3* dx) * (u[j+1] - u[j-1])*(u[j-1]+u[j]+u[j+1]) - 1/(2*dx**3) * (u[j+2]-2*u[j+1]  + 2*u[j-1] - u[j-2])
        u_old = u
    u = u_new
    
    # Plotting
plt.figure(figsize=(10, 5))
plt.plot(x, initial_waveform(x), label='Initial Waveform', linestyle='--')
plt.plot(x, u, label='Advection Solution')
plt.title('Burgers Equation (µ = 0)')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.show()
