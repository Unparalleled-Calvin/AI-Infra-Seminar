import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile, record_function, ProfilerActivity


cuda_source = """
__global__ void square_vector_kernel(const float* vector, float* result, int col) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < col) 
        result[c] = vector[c] * vector[c];
}

torch::Tensor square_vector(torch::Tensor vector) {
    const auto col = vector.size(0);
    auto result = torch::empty_like(vector);

    dim3 threads_per_block(128);
    dim3 blocks_per_grid((col + threads_per_block.x - 1) / threads_per_block.x);
    square_vector_kernel<<<blocks_per_grid, threads_per_block>>>(
        vector.data_ptr<float>(), result.data_ptr<float>(), col
    );

    return result;
}
"""

cpp_source = "torch::Tensor square_vector(torch::Tensor vector);"

module = load_inline(
    name="moss_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["square_vector"],
)

a = torch.randn((16384000), device="cuda")
print(module.square_vector(a))

with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=0, warmup=1, active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/moss_op"),
    record_shapes=False,
    with_stack=True,
) as prof:
    for _ in range(3):
        prof.step()
        with record_function("square_vector_custom"):
            module.square_vector(a)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
