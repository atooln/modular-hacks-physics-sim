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
    """Calculate the element-wise sum of two vectors on the GPU."""

    # Calculate the index of the vector element for the thread to process
    var tid = block_idx.x * block_dim.x + thread_idx.x

    # Don't process out of bounds elements
    if tid < vector_size:
        pass


