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

from python import PythonObject, Python
from memory import UnsafePointer

# attempt to convert to numpy array
# fn to_numpy() raises -> PythonObject:
#         var np = Python.import_module("numpy")
#         var np_arr = np.zeros(41, 41)
#     #     var npArrayPtr = UnsafePointer[dtype](
#     #     mlir_op.pop.index_to_pointer[
#     #         _type = mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
#     #     ](
#     #         SIMD[DType.index,1](np_arr.__array_interface__['data'][0].__index__()).value
#     #     )
#     # )
#         var npArrayPtr = UnsafePointer[Scalar[dtype]](np_arr.__array_interface__['data'][0].__index__())

#         # print("np_arr", np_arr)
#         return np_arr

# GPU kernel for FTCS update
fn heat_step(input: LayoutTensor[dtype, layout, MutableAnyOrigin], output: LayoutTensor[dtype, layout, MutableAnyOrigin]):
    var x = block_idx.x * block_dim.x + thread_idx.x
    var y = block_idx.y * block_dim.y + thread_idx.y
    
    if x < width and y < height:
        if x == 0 or y == 0 or x == width - 1 or y == height - 1:
            # Apply Dirichlet boundary condition (fixed value, e.g., 0.0)
            output[y * width + x] = 0.0
        else:            
            var center = input[y, x]
            var up = input[y - 1, x]
            var down = input[y + 1, x]
            var left = input[y, x - 1]
            var right = input[y, x + 1]
            var res = (1.0 - 4.0 * r) * center + r * (up + down + left + right)
            # if x == y:
            #     print("x", x, "y", y, "res", res)
            output[y, x] = res

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
        print("Found GPU:", ctx.name())
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
        alias block = (4, 4)
        alias grid = (ceildiv(N, 4), ceildiv(N, 4))

        print("N", N)
        print("steps", steps)

        for i in range(steps):
            ctx.enqueue_function[heat_step](
                in_tensor, 
                out_tensor,
                block_dim=block,
                grid_dim=grid
            )
            # ctx.enqueue_function[apply_neumann_bc](out_tensor, N, N)
            ctx.synchronize()

            ctx.enqueue_copy[dtype](
                dst_buf=in_device_buffer,
                src_buf=out_device_buffer,
            )

        print("Simulation completed; Steps:", steps)

        # Read final result
        # with out_tensor.map_to_host() as final_result:
            # let center_val = final_result[(N // 2) * N + (N // 2)]
            # print("Center value after diffusion:", center_val)

        # Create a HostBuffer for the result vector
        result_host_buffer = ctx.enqueue_create_host_buffer[dtype](N * N)

        # Copy the result vector from the DeviceBuffer to the HostBuffer
        ctx.enqueue_copy(dst_buf=result_host_buffer, src_buf=out_device_buffer)

        # Finally, synchronize the DeviceContext to run all enqueued operations
        ctx.synchronize()

        with open("file.txt", 'w') as f:
            for i in range(N):
                for j in range(N):
                    f.write(result_host_buffer[i * N + j])
                    f.write(" ")
                f.write("\n")
            f.write("\n")

        print("Result vector:", result_host_buffer)
        print("center value:", result_host_buffer[(N // 2) * N + (N // 2)])

        grid_numpy = result_host_buffer
        print("grid_numpy", grid_numpy)
        # center value at the end





