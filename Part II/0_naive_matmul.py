import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile, record_function, ProfilerActivity


cuda_source = """
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float val = 0.0f;
        for (int k = 0; k < K; ++k) {
            val += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = val;
    }
}

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix sizes");
    int M = A.size(0), K = A.size(1), N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(32, 32);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    matmul_naive_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    return C;
}
"""

cpp_source = """
torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B);
"""

module = load_inline(
    name="matmul_naive_mod",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_naive"],
    verbose=True,
)

M, K, N = 1024, 1024, 1024
A = torch.randn(M, K, device="cuda").contiguous()
B = torch.randn(K, N, device="cuda").contiguous()

with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=0, active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/matmul_naive"),
    record_shapes=False,
    with_stack=False,
) as prof:
    for _ in range(3):
        with record_function("matmul_naive"), torch.no_grad():
            C1 = module.matmul_naive(A, B)
        with record_function("torch_matmul"), torch.no_grad():
            C2 = torch.matmul(A, B)
        prof.step()

assert torch.allclose(C1, C2, atol=1e-3, rtol=1e-3)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
