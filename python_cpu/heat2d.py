import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Parameters
r = 1 / 4
h = 0.05
tau = r * h**2

x = np.arange(-1, 1 + h, h)
y = np.arange(-1, 1 + h, h)
t = np.arange(0, 1 + tau, tau)

# Grid size
nx, ny, nt = len(x), len(y), len(t)

# Initialize u
u = np.zeros((nx, ny, nt))

# Initial condition: 2D bell curve
X, Y = np.meshgrid(x, y, indexing='ij')
u[:, :, 0] = np.exp(-((3 * X)**2 + (3 * Y)**2))

# Dirichlet boundary at t=0
u[0, :, 0] = 0
u[-1, :, 0] = 0
u[:, 0, 0] = 0
u[:, -1, 0] = 0

# Time evolution
for k in range(nt - 1):
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u[i, j, k+1] = (
                (1 - 4*r) * u[i, j, k]
                + r * u[i-1, j, k]
                + r * u[i+1, j, k]
                + r * u[i, j-1, k]
                + r * u[i, j+1, k]
            )

    # Neumann boundary conditions
    u[:, 0, k+1] = u[:, 1, k+1]
    u[:, -1, k+1] = u[:, -2, k+1]
    u[0, :, k+1] = u[1, :, k+1]
    u[-1, :, k+1] = u[-2, :, k+1]

    # Corners
    u[0, 0, k+1] = (u[0, 1, k+1] + u[1, 0, k+1]) / 2
    u[0, -1, k+1] = (u[0, -2, k+1] + u[1, -1, k+1]) / 2
    u[-1, 0, k+1] = (u[-2, 0, k+1] + u[-1, 1, k+1]) / 2
    u[-1, -1, k+1] = (u[-2, -1, k+1] + u[-1, -2, k+1]) / 2

# Create figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(0, 1)

# Initial surface plot
surf = [ax.plot_surface(X, Y, u[:, :, 0], cmap='viridis')]

def update(frame):
    ax.clear()
    ax.set_zlim(0, 1)
    ax.set_title(f'Time step: {frame}')
    return [ax.plot_surface(X, Y, u[:, :, frame], cmap='viridis')]

# Frames every 5 time steps
frame_indices = list(range(0, nt, 5))
ani = animation.FuncAnimation(
    fig, update, frames=frame_indices, blit=False
)

# Save to GIF (requires Pillow)
ani.save("heat_diffusion.gif", writer='pillow', fps=5)
