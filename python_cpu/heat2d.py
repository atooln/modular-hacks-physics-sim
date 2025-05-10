import numpy as np
import matplotlib.pyplot as plt
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

# Initial velocity (phi) is zero
phi = np.zeros(len(x))

# First time step using Taylor expansion
for i in range(1, len(x) - 1):
    u[1, i] = 0.5 * (r**2 * u[0, i+1] + 2 * (1 - r**2) * u[0, i] + r**2 * u[0, i-1]) + k * phi[i]

# Time evolution
for j in range(1, len(t) - 1):
    for i in range(1, len(x) - 1):
        u[j+1, i] = (r**2 * u[j, i+1] +
                     2 * (1 - r**2) * u[j, i] +
                     r**2 * u[j, i-1] -
                     u[j-1, i])
    
    # Absorbing Boundary Conditions
    u[j+1, 0] = u[j, 1] + ((r - 1) / (r + 1)) * (u[j+1, 1] - u[j, 0])
    u[j+1, -1] = u[j, -2] + ((r - 1) / (r + 1)) * (u[j+1, -2] - u[j, -1])

# Animate 2D wave evolution
fig2, ax2 = plt.subplots()
line, = ax2.plot(x, u[0])
ax2.set_xlim([0, 1])
ax2.set_ylim([-1.2, 1.2])
ax2.set_title("1D Wave Propagation")
ax2.set_xlabel("x")
ax2.set_ylabel("u(x, t)")
plt.show()
