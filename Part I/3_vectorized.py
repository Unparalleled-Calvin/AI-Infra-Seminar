import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile, record_function, ProfilerActivity

cuda_source = """
__device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void silu_coalesced_float4_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col4 = blockIdx.x * blockDim.x + threadIdx.x; 
    int col = col4 << 2;

    if (row < rows && col + 3 < cols) {
        const float4* input_v4 = reinterpret_cast<const float4*>(input);
        float4 val = input_v4[row * (cols >> 2) + col4];

        val.x = silu(val.x);
        val.y = silu(val.y);
        val.z = silu(val.z);
        val.w = silu(val.w);

        float4* output_v4 = reinterpret_cast<float4*>(output);
        output_v4[row * (cols >> 2) + col4] = val;
    }
}

torch::Tensor silu_coalesced_float4(torch::Tensor input) {
    TORCH_CHECK(input.size(1) % 4 == 0, "Column size must be divisible by 4 for float4");

    int rows = input.size(0);
    int cols = input.size(1);

    auto output = torch::empty_like(input);

    dim3 threads(32, 32); 
    dim3 blocks((cols / 4 + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
    silu_coalesced_float4_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output;
}
"""

cpp_source = """
torch::Tensor silu_coalesced_float4(torch::Tensor input);
"""

module = load_inline(
    name="moss_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["silu_coalesced_float4"],
    verbose=True,
)

rows, cols = 8192, 8192
x = torch.randn((rows, cols), device="cuda").contiguous()

with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=0, warmup=1, active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/silu_float4"),
    record_shapes=False,
    with_stack=False,
) as prof:
    for _ in range(3):
        prof.step()
        with record_function("silu_float4"), torch.no_grad():
            out1 = module.silu_coalesced_float4(x)
        with record_function("silu_torch"), torch.no_grad():
            out0 = torch.nn.functional.silu(x)

assert torch.allclose(out1, out0)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
