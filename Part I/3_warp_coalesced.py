import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile, record_function, ProfilerActivity

cuda_source = """
__device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void silu_coalesced_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        output[idx] = silu(input[idx]);
    }
}

__global__ void silu_uncoalesced_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        output[idx] = silu(input[idx]);
    }
}

torch::Tensor silu_coalesced(torch::Tensor input) {
    const int rows = input.size(0);
    const int cols = input.size(1);
    auto output = torch::empty_like(input);

    dim3 threads(32, 32); 
    dim3 blocks((cols + threads.x - 1) / threads.x,
                (rows + threads.y - 1) / threads.y);
    silu_coalesced_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output;
}

torch::Tensor silu_uncoalesced(torch::Tensor input) {
    const int rows = input.size(0);
    const int cols = input.size(1);
    auto output = torch::empty_like(input);

    dim3 threads(32, 32);  
    dim3 blocks((rows + threads.x - 1) / threads.x,
                (cols + threads.y - 1) / threads.y);
    silu_uncoalesced_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output;
}
"""

cpp_source = """
torch::Tensor silu_coalesced(torch::Tensor input);
torch::Tensor silu_uncoalesced(torch::Tensor input);
"""

module = load_inline(
    name="moss_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["silu_coalesced", "silu_uncoalesced"],
    verbose=True,
)

rows, cols = 8192, 8192
x = torch.randn((rows, cols), device="cuda")

with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/warp_coalesced"),
    record_shapes=False,
    with_stack=False,
) as prof:
    for _ in range(4):
        prof.step()
        with record_function("silu_torch"), torch.no_grad():
            out0 = torch.nn.functional.silu(x)
        with record_function("silu_coalesced"), torch.no_grad():
            out1 = module.silu_coalesced(x)
        with record_function("silu_uncoalesced"), torch.no_grad():
            out2 = module.silu_uncoalesced(x)

assert torch.allclose(out1, out0) and torch.allclose(out2, out0)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
