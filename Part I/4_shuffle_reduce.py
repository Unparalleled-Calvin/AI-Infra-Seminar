import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile, record_function, ProfilerActivity

cuda_code = """
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int8_t offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduce_sum_large_tensor_kernel(const float* __restrict__ input, float* output, int N) {
    __shared__ float shared[32];  

    float val = 0.0f;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (idx < N) val += input[idx];
    if (idx + blockDim.x < N) val += input[idx + blockDim.x];

    val = warp_reduce_sum(val);

    if ((threadIdx.x & 31) == 0)
        shared[threadIdx.x >> 5] = val;

    __syncthreads();

    float block_sum = 0.0f;
    if (threadIdx.x < 32) {
        block_sum = shared[threadIdx.x];
        block_sum = warp_reduce_sum(block_sum);
    }

    if (threadIdx.x == 0)
        atomicAdd(output, block_sum);
}

torch::Tensor reduce_sum_large(torch::Tensor input) {
    auto output = torch::zeros({}, input.options());
    const int N = input.numel();
    const int threads = 1024;
    const int blocks = (N + threads * 2 - 1) / (threads * 2);
    reduce_sum_large_tensor_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}
"""

cpp_sig = "torch::Tensor reduce_sum_large(torch::Tensor input);"

module = load_inline(
    name="cuda_reduce_large",
    cpp_sources=cpp_sig,
    cuda_sources=cuda_code,
    functions=["reduce_sum_large"],
    verbose=False,
)

N = 1 << 20
x = torch.randn(N, device="cuda")

with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=0, warmup=1, active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/reduce_warp"),
    record_shapes=False,
    with_stack=False,
) as prof:
    for _ in range(3):
        prof.step()
        with record_function("torch_sum"), torch.no_grad():
            out0 = torch.sum(x)
        with record_function("cuda_reduce_sum_warp"), torch.no_grad():
            out1 = module.reduce_sum_large(x)


assert torch.allclose(out0, out1)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
