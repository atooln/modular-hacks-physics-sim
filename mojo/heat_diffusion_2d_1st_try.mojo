from gpu import thread_idx, block_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from math import exp, floor

# Simulation parameters
const h: f32 = 0.05
const r: f32 = 0.25
const tau: f32 = r * h * h
const T: f32 = 1.0
const steps: Int = floor(T / tau)
const N: Int = floor(2.0 / h) + 1  # x ∈ [-1, 1]
const dtype = DType.float32

# Initialize context and buffers
var ctx = DeviceContext()
var grid1 = ctx.enqueue_create_buffer[dtype](N * N)
var grid2 = ctx.enqueue_create_buffer[dtype](N * N)

# Define layout
let layout = Layout.strided((N, N), (N, 1))
let tensor1 = LayoutTensor[dtype](grid1, layout)
let tensor2 = LayoutTensor[dtype](grid2, layout)

# Initial condition: 3D Bell curve
with tensor1.buffer.map_to_host() as host_buf:
    for i in range(N):
        for j in range(N):
            let x = -1.0 + h * i
            let y = -1.0 + h * j
            host_buf[i * N + j] = exp(-9.0 * (x * x + y * y))

# GPU kernel for FTCS update
fn heat_step(input: LayoutTensor[dtype], output: LayoutTensor[dtype], width: Int, height: Int, r: f32):
    let x = block_idx().x * block_dim().x + thread_idx().x
    let y = block_idx().y * block_dim().y + thread_idx().y

    if x > 0 and y > 0 and x < width - 1 and y < height - 1:
        let center = input[y, x]
        let up = input[y - 1, x]
        let down = input[y + 1, x]
        let left = input[y, x - 1]
        let right = input[y, x + 1]
        output[y, x] = (1.0 - 4.0 * r) * center + r * (up + down + left + right)

# Kernel to apply Neumann boundary conditions
fn apply_neumann_bc(grid: LayoutTensor[dtype], width: Int, height: Int):
    let x = thread_idx().x
    let y = thread_idx().y

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

# Run simulation
let block = (16, 16)
let grid = ((N + 15) // 16, (N + 15) // 16)

for step in range(steps):
    ctx.launch_kernel(heat_step, grid, block, args=(tensor1, tensor2, N, N, r))
    ctx.launch_kernel(apply_neumann_bc, grid, block, args=(tensor2, N, N))
    let tmp = tensor1
    tensor1 = tensor2
    tensor2 = tmp

# Read final result
with tensor1.buffer.map_to_host() as final_result:
    let center_val = final_result[(N // 2) * N + (N // 2)]
    print("Center value after diffusion:", center_val)