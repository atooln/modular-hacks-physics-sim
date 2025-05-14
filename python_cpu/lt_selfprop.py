import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Improved Graph: denser + better structure (Barabási–Albert)
G = nx.barabasi_albert_graph(n=50, m=2, seed=42)
pos = nx.spring_layout(G, seed=42)

# Lower thresholds to encourage activation
thresholds = {node: random.uniform(0.1, 0.25) for node in G.nodes()}

# Higher edge influence
influence_weights = {
    (u, v): random.uniform(0.3, 0.7)
    for u, v in G.edges()
}
nx.set_edge_attributes(G, influence_weights, "weight")

# More seed users
active_nodes = set(random.sample(list(G.nodes), 5))
activation_history = [set(active_nodes)]

# Linear Threshold diffusion model
def simulate_step(active_set):
    new_active = set(active_set)
    for node in G.nodes():
        if node in active_set:
            continue
        neighbors = list(G.neighbors(node))
        total_influence = sum(
            influence_weights.get((nbr, node), influence_weights.get((node, nbr), 0))
            for nbr in neighbors if nbr in active_set
        )
        if total_influence >= thresholds[node]:
            new_active.add(node)
    return new_active

# Run simulation
steps = 20
for _ in range(steps):
    next_active = simulate_step(activation_history[-1])
    if next_active == activation_history[-1]:
        break
    activation_history.append(next_active)

# Create animation
fig, ax = plt.subplots(figsize=(8, 6))
def update(num):
    ax.clear()
    current_active = activation_history[num]
    node_colors = ['orange' if node in current_active else 'lightgray' for node in G.nodes()]
    nx.draw(G, pos, node_color=node_colors, with_labels=True, ax=ax)
    ax.set_title(f"Step {num}: {len(current_active)} active nodes")

ani = animation.FuncAnimation(fig, update, frames=len(activation_history), interval=1000, repeat=False)

# Save as GIF
ani_path = "lt_diffusion_improved.gif"
ani.save(ani_path, writer="pillow", fps=1)
 