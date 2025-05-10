import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Parameters
r = 1
h = 0.01
k = r * h

x = np.arange(0, 1 + h, h)
t = np.arange(0, 2 + k, k)

u = np.zeros((len(t), len(x)))

# Initial condition: sin curve between points 40 to 61
xSin = np.linspace(0, np.pi, 22)
u[0, 40:62] = np.sin(xSin)

# Initial velocity is zero
phi = np.zeros(len(x))

# First time step
for i in range(1, len(x) - 1):
    u[1, i] = 0.5 * (r**2 * u[0, i+1] + 2 * (1 - r**2) * u[0, i] + r**2 * u[0, i-1]) + k * phi[i]

# Time evolution
for j in range(1, len(t) - 1):
    for i in range(1, len(x) - 1):
        u[j+1, i] = (r**2 * u[j, i+1] +
                     2 * (1 - r**2) * u[j, i] +
                     r**2 * u[j, i-1] -
                     u[j-1, i])
    # Absorbing boundaries
    u[j+1, 0] = u[j, 1] + ((r - 1) / (r + 1)) * (u[j+1, 1] - u[j, 0])
    u[j+1, -1] = u[j, -2] + ((r - 1) / (r + 1)) * (u[j+1, -2] - u[j, -1])

# Animation: 3D Surface plot that evolves over time
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, T = np.meshgrid(x, t)
surf = [ax.plot_surface(X, T, u, cmap='viridis')]

def update_surface(frame):
    ax.clear()
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(x,t)")
    ax.set_title(f"Wave Evolution at time step {frame}")
    ax.plot_surface(X[:frame+1, :], T[:frame+1, :], u[:frame+1, :], cmap='viridis')
    return fig,

ani = animation.FuncAnimation(fig, update_surface, frames=len(t), interval=50)
ani.save("wave_surface.gif", writer='pillow', fps=10)
print("3D surface animation saved as wave_surface.gif")
