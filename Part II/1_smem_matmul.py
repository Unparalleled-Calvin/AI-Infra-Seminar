import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile, record_function, ProfilerActivity


cuda_source = """
__global__ void matmul_shared_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + blockDim.x - 1) / blockDim.x; ++t) {
        if (row < M && t * blockDim.x + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * blockDim.x + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * blockDim.x + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * blockDim.x + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

torch::Tensor matmul_shared(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix sizes");
    int M = A.size(0), K = A.size(1), N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(32, 32);
    dim3 blocks((N + 32 - 1) / 32, (M + 32 - 1) / 32);

    matmul_shared_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    return C;
}
"""

cpp_source = """
torch::Tensor matmul_shared(torch::Tensor A, torch::Tensor B);
"""

module = load_inline(
    name="matmul_naive_mod",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_shared"],
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
        with record_function("matmul_coalesced"), torch.no_grad():
            C2 = module.matmul_shared(A, B)
        with record_function("torch_matmul"), torch.no_grad():
            C0 = torch.matmul(A, B)

        prof.step()

assert torch.allclose(C0, C2, atol=1e-3, rtol=1e-3)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
