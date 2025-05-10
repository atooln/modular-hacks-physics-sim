from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from sys import has_accelerator
from layout import Layout, LayoutTensor
from math import exp, floor
from math import ceildiv

# Simulation parameters
alias h = 0.05
alias r = 0.25
alias tau = r * h * h
alias T = 1.0
alias steps = floor(T / tau)
alias N = Int(ceildiv(2.0, h) + 1) # x ∈ [-1, 1]
alias width = N
alias height = N
alias dtype = DType.float32
alias layout = Layout.row_major(N, N)

# GPU kernel for FTCS update
fn heat_step(input: LayoutTensor[dtype], output: LayoutTensor[dtype]):
    var x = block_idx.x * block_dim.x + thread_idx.x
    var y = block_idx.y * block_dim.y + thread_idx.y

    if x > 0 and y > 0 and x < width - 1 and y < height - 1:
        # Apply Dirichlet boundary condition (fixed value, e.g., 0.0)
        output[y * width + x] = 0.0
    else:
        var center = input[y, x]
        var up = input[y - 1, x]
        var down = input[y + 1, x]
        var left = input[y, x - 1]
        var right = input[y, x + 1]
        output[y, x] = (1.0 - 4.0 * r) * center + r * (up + down + left + right)

# Kernel to apply Neumann boundary conditions
fn apply_neumann_bc(grid: LayoutTensor[dtype], width: Int, height: Int):
    var x = block_idx.x * block_dim.x + thread_idx.x
    var y = block_idx.y * block_dim.y + thread_idx.y

    if y == 0 and x < width:
        grid[0, x] = grid[1, x]
        grid[height - 1, x] = grid[height - 2, x]

    if x == 0 and y < height:
        grid[y, 0] = grid[y, 1]
        grid[y, width - 1] = grid[y, width - 2]

    if x == 0 and y == 0:
        grid[0, 0] = (grid[1, 0] + grid[0, 1]) / 2.0
        grid[0, width - 1] = (grid[0, width - 2] + grid[1, width - 1]) / 2.0
        grid[height - 1, 0] = (grid[height - 2, 0] + grid[height - 1, 1]) / 2.0
        grid[height - 1, width - 1] = (grid[height - 2, width - 1] + grid[height - 1, width - 2]) / 2.0

# Initialize context and buffers
def main():
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        # create host buffers
        in_host_buffer = ctx.enqueue_create_host_buffer[dtype](N * N)
        ctx.synchronize()

        for i in range(N):
            for j in range(N):
                var x = Float32(-1.0 + h * i)
                var y = Float32(-1.0 + h * j)
                in_host_buffer[i * N + j] = exp(-9.0 * (x * x + y * y))

        in_device_buffer = ctx.enqueue_create_buffer[dtype](N * N)

        ctx.enqueue_copy[dtype](
            dst_buf=in_device_buffer,
            src_buf=in_host_buffer,
        )

        out_device_buffer = ctx.enqueue_create_buffer[dtype](N * N)

        ctx.synchronize()
        
        in_tensor = LayoutTensor[dtype, layout](in_host_buffer)
        out_tensor = LayoutTensor[dtype, layout](out_device_buffer)
            
        # Run simulation
        alias block = (16, 16)
        alias grid = ((N + 15) // 16, (N + 15) // 16)

        for _ in range(steps):
            ctx.enqueue_function[heat_step](
                in_tensor, 
                out_tensor,
                block_dim=block,
                grid_dim=grid
            )
            # ctx.enqueue_function[apply_neumann_bc](out_tensor, N, N)
            ctx.synchronize()

        # Read final result
        # with out_tensor.map_to_host() as final_result:
            # let center_val = final_result[(N // 2) * N + (N // 2)]
            # print("Center value after diffusion:", center_val)





