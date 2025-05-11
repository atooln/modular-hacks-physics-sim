from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from sys import has_accelerator
from layout import Layout, LayoutTensor
from math import exp, floor
from math import ceildiv

from time import perf_counter_ns

# Simulation parameters
alias h = 0.05
alias r = 0.25
alias tau = r * h * h
alias T = 1.0
alias steps = Int(T / tau) + 1
# alias N = Int(ceildiv(2.0, h) + 1) # x ∈ [-1, 1]
alias N = Int((10.0 - (-10.0)) / h) + 1  # 41 points, x ∈ [-1, 1]
# alias N = 1024
alias width = N
alias height = N
alias dtype = DType.float32
alias layout = Layout.row_major(N, N)

from python import PythonObject, Python
from memory import UnsafePointer

# GPU kernel for FTCS update
fn heat_step(
    input: LayoutTensor[dtype, layout, MutableAnyOrigin], 
    output: LayoutTensor[dtype, layout, MutableAnyOrigin]
):
    var x = block_idx.x * block_dim.x + thread_idx.x
    var y = block_idx.y * block_dim.y + thread_idx.y
    
    if x < width and y < height:
        if x == 0 or y == 0 or x == width - 1 or y == height - 1:
            output[y, x] = 0.0
            pass
        else:        
            var center = input[y, x]
            var up = input[y - 1, x]
            var down = input[y + 1, x]
            var left = input[y, x - 1]
            var right = input[y, x + 1]
            var res = (1.0 - 4.0 * r) * center + r * (up + down + left + right)
            output[y, x] = res

# Kernel to apply Neumann boundary conditions
# fn apply_neumann_bc(grid: LayoutTensor[dtype], width: Int, height: Int):
#     var x = block_idx.x * block_dim.x + thread_idx.x
#     var y = block_idx.y * block_dim.y + thread_idx.y

#     if y == 0 and x < width:
#         grid[0, x] = grid[1, x]
#         grid[height - 1, x] = grid[height - 2, x]

#     if x == 0 and y < height:
#         grid[y, 0] = grid[y, 1]
#         grid[y, width - 1] = grid[y, width - 2]

#     if x == 0 and y == 0:
#         grid[0, 0] = (grid[1, 0] + grid[0, 1]) / 2.0
#         grid[0, width - 1] = (grid[0, width - 2] + grid[1, width - 1]) / 2.0
#         grid[height - 1, 0] = (grid[height - 2, 0] + grid[height - 1, 1]) / 2.0
#         grid[height - 1, width - 1] = (grid[height - 2, width - 1] + grid[height - 1, width - 2]) / 2.0

def save_to_file(filename: String, data: LayoutTensor[dtype, layout]):
    with open(filename, "w") as file:
        for i in range(N):
            for j in range(N):
                file.write(data[i, j])
                file.write(" ")
            file.write("\n")

def main():
    very_beginning_time = perf_counter_ns()
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        print("Found GPU:", ctx.name())
        # create host buffers
        in_host_buffer = ctx.enqueue_create_host_buffer[dtype](N * N)
        ctx.synchronize()

        for i in range(N):
            for j in range(N):
                var x = Float32(-1.0 + h * i)
                var y = Float32(-1.0 + h * j)
                in_host_buffer[i * N + j] = exp(-9.0 * (x * x + y * y))
        # in_host_buffer[(N // 2) * N + N // 2 ] = 1.0

        in_device_buffer = ctx.enqueue_create_buffer[dtype](N * N)

        ctx.enqueue_copy[dtype](
            dst_buf=in_device_buffer,
            src_buf=in_host_buffer,
        )

        out_device_buffer = ctx.enqueue_create_buffer[dtype](N * N)

        ctx.synchronize()
        
        in_tensor = LayoutTensor[dtype, layout](in_device_buffer)
        out_tensor = LayoutTensor[dtype, layout](out_device_buffer)
            
        # Run simulation
        alias block = (16, 16)
        alias grid = (ceildiv(N, 16), ceildiv(N, 16))

        print("N", N)
        print("steps", steps)

        # Create a HostBuffer for the result vector
        result_host_buffer = ctx.enqueue_create_host_buffer[dtype](N * N)

        start_time = perf_counter_ns()
        print("time for memory allocation", (start_time - very_beginning_time) / 1e9, "s")

        for _ in range(steps):
            ctx.enqueue_function[heat_step](
                in_tensor, 
                out_tensor,
                block_dim=block,
                grid_dim=grid
            )
            ctx.synchronize()
            ctx.enqueue_copy[dtype](
                dst_buf=in_device_buffer,
                src_buf=out_device_buffer,
            )
            ctx.synchronize()        


        print("Simulation completed; Steps:", steps)


        # Copy the result vector from the DeviceBuffer to the HostBuffer
        ctx.enqueue_copy(dst_buf=result_host_buffer, src_buf=out_device_buffer)

        # Finally, synchronize the DeviceContext to run all enqueued operations
        ctx.synchronize()

        end_time = perf_counter_ns()
        print("Simulation time:", (end_time - start_time) / 1e9, "s")

        # Save the result to a file

        # grid_numpy = result_host_buffer
        # print("grid_numpy", grid_numpy)
        # center value at the end

        very_end_time = perf_counter_ns()
        print("time writing", (very_end_time - end_time) / 1e9, "s")

        
        from gpu.host import DeviceContext
        from python import Python
        from memory import UnsafePointer
        from sys._assembly import inlined_assembly
        from sys.intrinsics import _unsafe_aliasing_address_to_pointer
        from gpu.host import HostBuffer, DeviceContext

        var ctx = DeviceContext()
        var buffer = ctx.enqueue_create_host_buffer[dtype](25)

        var np = Python.import_module("numpy")
        var np_arr = np.zeros(N*N, dtype="float32")

        var address = Scalar[DType.index](Int(np_arr.__array_interface__['data'][0]))
        var ptr = _unsafe_aliasing_address_to_pointer[dtype](address)

        for i in range(N):
            for j in range(N):
                var x = Float32(-1.0 + h * i)
                var y = Float32(-1.0 + h * j)
                np_arr[i * N + j] = result_host_buffer[i * N + j]

        print(np_arr)


        fig_file = Python.import_module("fig")
        fig_file.plot_surf(np_arr, N, N)
