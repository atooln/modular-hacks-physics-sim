from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor
from gpu.host import DeviceContext
from math import exp, pi, ceildiv

alias h = Float32(0.05)
alias c = Float32(3e8)
alias dt = Float32(0.99) * h / (c * 1.414)
alias xmin = Float32(-10.0)
alias xmax = Float32(10.0)
alias ymin = Float32(-10.0)
alias ymax = Float32(10.0)
alias nx = Int(((xmax - xmin) / h) + 1)
alias ny = Int(((ymax - ymin) / h) + 1)
alias num_steps = 1000
alias snapshot_interval = 20
alias num_frames = Int(num_steps / snapshot_interval)

alias eps0 = Float32(8.854e-12)
alias mu0 = Float32(4.0 * pi * 1e-7)
alias eps_inv = 1.0 / eps0
alias mu_inv = 1.0 / mu0
alias dx_inv = 1.0 / h
alias dy_inv = 1.0 / h

alias dtype = DType.float32
alias layout2d = Layout.row_major(nx, ny)

fn update_Ez(
    Ez: LayoutTensor[dtype, layout2d, MutableAnyOrigin],
    Hx: LayoutTensor[dtype, layout2d, MutableAnyOrigin],
    Hy: LayoutTensor[dtype, layout2d, MutableAnyOrigin]
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    var j = block_idx.y * block_dim.y + thread_idx.y
    if i > 0 and j > 0 and i < nx - 1 and j < ny - 1:
        var curlH = (Hy[j, i] - Hy[j, i - 1]) * dx_inv - (Hx[j, i] - Hx[j - 1, i]) * dy_inv
        Ez[j, i] = Ez[j, i] + dt * eps_inv * curlH

fn update_H(
    Ez: LayoutTensor[dtype, layout2d, MutableAnyOrigin],
    Hx: LayoutTensor[dtype, layout2d, MutableAnyOrigin],
    Hy: LayoutTensor[dtype, layout2d, MutableAnyOrigin]
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    var j = block_idx.y * block_dim.y + thread_idx.y
    if i < nx - 1 and j < ny - 1:
        var dEz_dy = (Ez[j + 1, i] - Ez[j, i]) * dy_inv
        var dEz_dx = (Ez[j, i + 1] - Ez[j, i]) * dx_inv
        Hx[j, i] = Hx[j, i] - dt * mu_inv * dEz_dy
        Hy[j, i] = Hy[j, i] + dt * mu_inv * dEz_dx

fn inject_point_source(
    Ez: LayoutTensor[dtype, layout2d, MutableAnyOrigin],
    src_i: Int, src_j: Int, value: Float32
):
    Ez[src_j, src_i] = Ez[src_j, src_i] + value

fn ricker(n: Int) -> Float32:
    var t0 = Float32(30) * dt
    var sigma = Float32(10) * dt
    var t = Float32(n) * dt
    var arg = (t - t0) / sigma
    return (1.0 - 2.0 * arg * arg) * exp(-arg * arg)

def main():
    var ctx = DeviceContext()

    var Ez_host = ctx.enqueue_create_host_buffer[dtype](nx * ny)
    var Hx_host = ctx.enqueue_create_host_buffer[dtype](nx * ny)
    var Hy_host = ctx.enqueue_create_host_buffer[dtype](nx * ny)
    var Ez_snapshots = ctx.enqueue_create_host_buffer[dtype](nx * ny * num_frames)

    var Ez_dev = ctx.enqueue_create_buffer[dtype](nx * ny)
    var Hx_dev = ctx.enqueue_create_buffer[dtype](nx * ny)
    var Hy_dev = ctx.enqueue_create_buffer[dtype](nx * ny)

    ctx.enqueue_copy[dtype](dst_buf=Ez_dev, src_buf=Ez_host)
    ctx.enqueue_copy[dtype](dst_buf=Hx_dev, src_buf=Hx_host)
    ctx.enqueue_copy[dtype](dst_buf=Hy_dev, src_buf=Hy_host)
    ctx.synchronize()

    var Ez_t = LayoutTensor[dtype, layout2d](Ez_dev)
    var Hx_t = LayoutTensor[dtype, layout2d](Hx_dev)
    var Hy_t = LayoutTensor[dtype, layout2d](Hy_dev)

    alias block = (16, 16)
    alias grid = (ceildiv(nx, 16), ceildiv(ny, 16))

    var src_i = nx // 2
    var src_j = ny // 2

    var step: Int = 0
    var frame: Int = 0

    while step < num_steps:
        ctx.enqueue_function[update_Ez](Ez_t, Hx_t, Hy_t, block_dim=block, grid_dim=grid)
        ctx.synchronize()

        var src_val = ricker(step)
        ctx.enqueue_function[inject_point_source](Ez_t, src_i, src_j, src_val, block_dim=(1, 1), grid_dim=(1, 1))
        ctx.synchronize()

        ctx.enqueue_function[update_H](Ez_t, Hx_t, Hy_t, block_dim=block, grid_dim=grid)
        ctx.synchronize()

        if step % snapshot_interval == 0:
            ctx.enqueue_copy[dtype](dst_buf=Ez_host, src_buf=Ez_dev)
            ctx.synchronize()
            for i in range(nx * ny):
                Ez_snapshots[frame * nx * ny + i] = Ez_host[i]
            frame = frame + 1

        step = step + 1

    ctx.enqueue_copy[dtype](dst_buf=Ez_host, src_buf=Ez_dev)
    ctx.synchronize()

    var center = Ez_host[src_j * nx + src_i]
    print("Ez at dipole after", num_steps, "steps:", center)

    from python import Python
    from sys.intrinsics import _unsafe_aliasing_address_to_pointer

    var np = Python.import_module("numpy")
    var np_arr = np.zeros(num_frames * nx * ny, dtype="float32")

    var address = Scalar[DType.index](Int(np_arr.__array_interface__["data"][0]))
    _ = _unsafe_aliasing_address_to_pointer[dtype](address)

    for i in range(num_frames * nx * ny):
        np_arr[i] = Ez_snapshots[i]

    var fig = Python.import_module("fig")
    fig.animate_field(np_arr, num_frames, nx, ny)
