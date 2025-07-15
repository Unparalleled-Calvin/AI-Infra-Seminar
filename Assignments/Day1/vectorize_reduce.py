import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile, record_function, ProfilerActivity

cuda_source = """
__global__ void reduce_reorder_atomic_kernel(const float* input, float* output, int N) {
    __shared__ float sdata[1024 * 4]; //shared memory扩容到4倍

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    float4* sdata_v4 = reinterpret_cast<float4*>(sdata);
    const float4* input_v4 = reinterpret_cast<const float4*>(input);
    
    float4 val = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 tmp_val;
    if ((i << 2) < N) {
        tmp_val = input_v4[i];
        val.x += tmp_val.x;
        val.y += tmp_val.y;
        val.z += tmp_val.z;
        val.w += tmp_val.w;
    }
    if (((i + blockDim.x) << 2) < N) {
        tmp_val = input_v4[i + blockDim.x];
        val.x += tmp_val.x;
        val.y += tmp_val.y;
        val.z += tmp_val.z;
        val.w += tmp_val.w;
    }
    sdata_v4[tid] = val;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            val = sdata_v4[tid];
            tmp_val = sdata_v4[tid + s];
            val.x += tmp_val.x;
            val.y += tmp_val.y;
            val.z += tmp_val.z;
            val.w += tmp_val.w;
            sdata_v4[tid] = val;
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0] + sdata[1] + sdata[2] + sdata[3]);
    }
}

torch::Tensor reduce_reorder_atomic(torch::Tensor input) {
    int N = input.size(0);
    int threads = 1024;

    int blocks = (N + threads * 2 * 4 - 1) / (threads * 2 * 4);

    auto output = torch::zeros({}, input.options());
    
    reduce_reorder_atomic_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N
    );

    return output;
}
"""

cpp_source = "torch::Tensor reduce_reorder_atomic(torch::Tensor input);"

module = load_inline(
    name="moss_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["reduce_reorder_atomic"],
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
            out1 = module.reduce_reorder_atomic(x)
        with record_function("torch_sum"), torch.no_grad():
            out0 = torch.sum(x)

assert torch.allclose(out0, out1)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
