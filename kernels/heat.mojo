from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx
from sys import has_accelerator
from layout import Layout, LayoutTensor
from gpu.id import block_dim, block_idx, thread_idx
from math import ceildiv

alias float_dtype = DType.float32
alias sim_grid_width = 1000
alias sim_grid_height = 1000
alias vector_size = sim_grid_width * sim_grid_height
alias layout = Layout.row_major(vector_size)

fn heat2d(
    current: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
    previous: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
    out: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
):
    """Iterative calculation of heat values based on stencil method."""

    # Calculate 2D thread indices
    var x = block_idx.x * block_dim.x + thread_idx.x
    var y = block_idx.y * block_dim.y + thread_idx.y

    # Check if the thread is within grid bounds
    if x < sim_grid_width and y < sim_grid_height:
        # Boundary conditions: edges and corners
        if x == 0 or x == sim_grid_width - 1 or y == 0 or y == sim_grid_height - 1:
            # Apply Dirichlet boundary condition (fixed value, e.g., 0.0)
            out[y * sim_grid_width + x] = 0.0
        else:
            # Interior points: apply 5-point stencil for heat equation
            # Example: u_new = u_current + alpha * (u_left + u_right + u_top + u_bottom - 4 * u_center)
            alias alpha = 0.25  # Diffusion coefficient (adjust as needed)
            var value = current[y * sim_grid_width + x] + alpha * (
                current[y * sim_grid_width + (x - 1)] +      # Left
                current[y * sim_grid_width + (x + 1)] +      # Right
                current[(y - 1) * sim_grid_width + x] +      # Top
                current[(y + 1) * sim_grid_width + x] -      # Bottom
                4.0 * current[y * sim_grid_width + x]        # Center
            )
            out[y * sim_grid_width + x] = value




