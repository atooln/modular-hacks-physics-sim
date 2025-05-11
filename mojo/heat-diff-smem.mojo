# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from gpu import thread_idx, global_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from sys import has_accelerator
from layout import Layout, LayoutTensor
from math import exp, ceildiv
from time import perf_counter_ns
from memory import stack_allocation
from gpu.memory import AddressSpace

# Simulation parameters
alias h = 0.05
alias r = 0.25
alias tau = r * h * h
alias T = 1.0
alias steps = Int(T / tau) + 1
alias x_min = -10.0
alias x_max = 10.0
alias N = Int((x_max - x_min) / h) + 1  # Grid size
alias width = N
alias height = N
alias dtype = DType.float32
alias layout = Layout.row_major(N, N)
alias tile_size = 16  # Tile size for shared memory (16x16 threads per block)
alias halo = 1  # Halo region for stencil
alias shared_tile_size = tile_size + 2 * halo  # Total tile size including halo

# Kernel for horizontal 1D convolution
fn heat_step_horizontal[
    layout: Layout
](
    input: LayoutTensor[dtype, layout, MutableAnyOrigin], 
    output: LayoutTensor[dtype, layout, MutableAnyOrigin]
):
    # Shared memory for tile
    var shared_mem = stack_allocation[
        shared_tile_size * shared_tile_size, 
        dtype, 
        address_space=AddressSpace.SHARED
    ]()

    var bx = block_idx.x
    var by = block_idx.y
    var tx = thread_idx.x
    var ty = thread_idx.y

    # Global indices
    # var gx = bx * tile_size + tx
    var gx = global_idx.x
    # var gy = by * tile_size + ty
    var gy = global_idx.y

    # Shared memory indices (offset by halo)
    var sx = tx + halo
    var sy = ty + halo

    # Load tile into shared memory, including halo regions
    if gx < width and gy < height:
        shared_mem[sy * shared_tile_size + sx] = SIMD[dtype,1](input[gy, gx][0])

    # Load halo regions
    if tx < halo:
        # Left halo
        var left_gx = gx - halo
        if left_gx >= 0 and gy < height:
            shared_mem[sy * shared_tile_size + (sx - halo)] = SIMD[dtype,1](input[gy, left_gx][0])
        # Right halo
        var right_gx = gx + tile_size
        if right_gx < width and gy < height:
            shared_mem[sy * shared_tile_size + (sx + tile_size)] = SIMD[dtype,1](input[gy, right_gx][0])
    if ty < halo:
        # Top halo
        var top_gy = gy - halo
        if top_gy >= 0 and gx < width:
            shared_mem[(sy - halo) * shared_tile_size + sx] = SIMD[dtype, 1](input[top_gy, gx][0])
        # Bottom halo
        var bottom_gy = gy + tile_size
        if bottom_gy < height and gx < width:
            shared_mem[(sy + tile_size) * shared_tile_size + sx] = SIMD[dtype, 1](input[bottom_gy, gx][0])

    barrier()

    # Apply horizontal 1D convolution
    if gx < width and gy < height:
        if gx == 0 or gx == width - 1 or gy == 0 or gy == height - 1:
            output[gy, gx] = 0.0
        else:
            var left = shared_mem[sy * shared_tile_size + (sx - 1)]
            var center = shared_mem[sy * shared_tile_size + sx]
            var right = shared_mem[sy * shared_tile_size + (sx + 1)]
            output[gy, gx] = r * left + (1.0 - 2.0 * r) * center + r * right

    barrier()

# Kernel for vertical 1D convolution
fn heat_step_vertical[
    layout: Layout
](
    input: LayoutTensor[dtype, layout, MutableAnyOrigin], 
    output: LayoutTensor[dtype, layout, MutableAnyOrigin]
):
    # Shared memory for tile
    var shared_mem = stack_allocation[
        shared_tile_size * shared_tile_size, 
        dtype, 
        address_space=AddressSpace.SHARED
    ]()

    var bx = block_idx.x
    var by = block_idx.y
    var tx = thread_idx.x
    var ty = thread_idx.y

    # Global indices
    var gx = bx * tile_size + tx
    var gy = by * tile_size + ty

    # Shared memory indices (offset by halo)
    var sx = tx + halo
    var sy = ty + halo

    # Load tile into shared memory, including halo regions
    if gx < width and gy < height:
        shared_mem[sy * shared_tile_size + sx] = SIMD[dtype, 1](input[gy, gx][0])

    # Load halo regions
    if tx < halo:
        # Left halo
        var left_gx = gx - halo
        if left_gx >= 0 and gy < height:
            shared_mem[sy * shared_tile_size + (sx - halo)] = SIMD[dtype,1](input[gy, left_gx][0])
        # Right halo
        var right_gx = gx + tile_size
        if right_gx < width and gy < height:
            shared_mem[sy * shared_tile_size + (sx + tile_size)] = SIMD[dtype, 1](input[gy, right_gx][0])
    if ty < halo:
        # Top性地
        var top_gy = gy - halo
        if top_gy >= 0 and gx < width:
            shared_mem[(sy - halo) * shared_tile_size + sx] = SIMD[dtype, 1](input[top_gy, gx][0])
        # Bottom halo
        var bottom_gy = gy + tile_size
        if bottom_gy < height and gx < width:
            shared_mem[(sy + tile_size) * shared_tile_size + sx] = SIMD[dtype, 1](input[bottom_gy, gx][0])

    barrier()

    # Apply vertical 1D convolution
    if gx < width and gy < height:
        if gx == 0 or gx == width - 1 or gy == 0 or gy == height - 1:
            output[gy, gx] = 0.0
        else:
            var up = shared_mem[(sy - 1) * shared_tile_size + sx]
            var center = shared_mem[sy * shared_tile_size + sx]
            var down = shared_mem[(sy + 1) * shared_tile_size + sx]
            output[gy, gx] = r * up + (1.0 - 2.0 * r) * center + r * down

    barrier()

# Main function to run the simulation
def main():
    var very_beginning_time = perf_counter_ns()

    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
        return

    var ctx = DeviceContext()
    print("Found GPU:", ctx.name())

    # Create host buffer and initialize with Gaussian
    var in_host_buffer = ctx.enqueue_create_host_buffer[dtype](N * N)
    ctx.synchronize()

    for i in range(N):
        for j in range(N):
            var x = Float32(x_min + h * i)
            var y = Float32(x_min + h * j)
            in_host_buffer[i * N + j] = exp(-9.0 * (x * x + y * y))

    # Create device buffers
    var in_device_buffer = ctx.enqueue_create_buffer[dtype](N * N)
    var out_device_buffer = ctx.enqueue_create_buffer[dtype](N * N)
    var temp_device_buffer = ctx.enqueue_create_buffer[dtype](N * N)  # For separable convolution

    # Copy initial data to device
    ctx.enqueue_copy[dtype](
        dst_buf=in_device_buffer,
        src_buf=in_host_buffer,
    )
    ctx.synchronize()

    # Create tensors
    var in_tensor = LayoutTensor[dtype, layout](in_device_buffer)
    var out_tensor = LayoutTensor[dtype, layout](out_device_buffer)
    var temp_tensor = LayoutTensor[dtype, layout](temp_device_buffer)

    # Grid and block dimensions
    alias block = (tile_size, tile_size)
    alias grid = (ceildiv(N, tile_size), ceildiv(N, tile_size))

    print("Grid size N:", N)
    print("Simulation steps:", steps)

    # Create host buffer for final result
    var result_host_buffer = ctx.enqueue_create_host_buffer[dtype](N * N)

    var start_time = perf_counter_ns()
    print("Memory allocation time:", (start_time - very_beginning_time) / 1e9, "s")

    # Run simulation with separable convolutions
    for _ in range(steps):
        # Horizontal pass
        ctx.enqueue_function[heat_step_horizontal[layout]](
            in_tensor, 
            temp_tensor,
            block_dim=block,
            grid_dim=grid
        )
        ctx.synchronize()

        # Vertical pass
        ctx.enqueue_function[heat_step_vertical[layout]](
            temp_tensor, 
            out_tensor,
            block_dim=block,
            grid_dim=grid
        )
        ctx.synchronize()

        # Swap buffers
        ctx.enqueue_copy[dtype](
            dst_buf=in_device_buffer,
            src_buf=out_device_buffer,
        )
        ctx.synchronize()

    print("Simulation completed; Steps:", steps)

    # Copy result back to host
    ctx.enqueue_copy[dtype](
        dst_buf=result_host_buffer,
        src_buf=out_device_buffer,
    )
    ctx.synchronize()

    var end_time = perf_counter_ns()
    print("Simulation time:", (end_time - start_time) / 1e9, "s")

    var very_end_time = perf_counter_ns()
    print("Result copy time:", (very_end_time - end_time) / 1e9, "s")
