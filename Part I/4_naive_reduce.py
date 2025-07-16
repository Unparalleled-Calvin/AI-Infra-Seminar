import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile, record_function, ProfilerActivity

cuda_source = """
__global__ void reduce_neighbor_atomic_kernel(const float* input, float* output, int N) {
    __shared__ float sdata[128]; 

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    float val = 0.0f;
    if (i < N) val += input[i];
    if (i + blockDim.x < N) val += input[i + blockDim.x];
    sdata[tid] = val;

    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        if (tid % (stride<<1) == 0 && tid + stride < blockDim.x)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);  
    }
}

torch::Tensor reduce_neighbor_atomic(torch::Tensor input) {
    int N = input.size(0);
    int threads = 128;
    int blocks = (N + threads * 2 - 1) / (threads * 2);

    auto output = torch::zeros({}, input.options());
    
    reduce_neighbor_atomic_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N
    );

    return output;
}
"""

cpp_source = "torch::Tensor reduce_neighbor_atomic(torch::Tensor input);"

module = load_inline(
    name="moss_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["reduce_neighbor_atomic"],
    verbose=False,
)

N = 1 << 20
x = torch.randn(N, device="cuda")

with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=0, warmup=1, active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/reduce_atomic_prof"),
    record_shapes=False,
    with_stack=False,
) as prof:
    for i in range(3):
        prof.step()
        with record_function("cuda_reduce_atomic"), torch.no_grad():
            out1 = module.reduce_neighbor_atomic(x)
        with record_function("torch_sum"), torch.no_grad():
            out0 = torch.sum(x)

assert torch.allclose(out0, out1)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
