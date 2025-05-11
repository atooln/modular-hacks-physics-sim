import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.signal import convolve2d

# Grid parameters
h = 0.02
x = np.arange(-1, 1 + h, h)
y = np.arange(-1, 1 + h, h)
X, Y = np.meshgrid(x, y, indexing='ij')
nx, ny = len(x), len(y)

# Time parameters
r_x = 0.2  # directional diffusion in X (chip lanes)
r_y = 0.05  # slower diffusion in Y
tau = h**2 * max(r_x, r_y)
T = 3.0
t = np.arange(0, T + tau, tau)
nt = len(t)

# Initial temperature field (hot chip core)
u = np.zeros((nx, ny, nt))
u[:, :, 0] = np.exp(-((X / 0.1)**2 + (Y / 0.05)**2))  # elliptical hot zone

# Spatially varying conductivity (high in chip center)
D = np.ones((nx, ny))
cx, cy = nx // 2, ny // 2
D[cx-10:cx+10, cy-10:cy+10] = 3.0  # high conductivity in core

# Directional Laplacian convolution kernels
Kx = np.array([[0, 0, 0],
               [1, -2, 1],
               [0, 0, 0]])

Ky = np.array([[0, 1, 0],
               [0, -2, 0],
               [0, 1, 0]])

# Simulation loop
for k in range(nt - 1):
    lap_x = convolve2d(u[:, :, k], Kx, mode='same', boundary='symm')
    lap_y = convolve2d(u[:, :, k], Ky, mode='same', boundary='symm')
    lap = r_x * lap_x + r_y * lap_y
    u[:, :, k+1] = u[:, :, k] + tau * D * lap

# Visualization
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(0, 1)
surf = [ax.plot_surface(X, Y, u[:, :, 0], cmap='inferno')]

def update(frame):
    ax.clear()
    ax.set_zlim(0, 1)
    ax.set_title(f'Time step: {frame}')
    return [ax.plot_surface(X, Y, u[:, :, frame], cmap='inferno')]

ani = animation.FuncAnimation(fig, update, frames=range(0, nt, 5), blit=False)
ani.save("gpu_chip_heat_sim.gif", writer='pillow', fps=5)
