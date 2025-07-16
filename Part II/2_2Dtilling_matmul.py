import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile, ProfilerActivity

cuda_source = """
#define BS 32
#define TS 4

__global__ void matmul_2d_tiling_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BS][BS];
    __shared__ float Bs[BS][BS];

    float accs[TS][TS] = {0.0f};

    const int C_start_row = blockIdx.y * BS + threadIdx.y * TS;
    const int C_start_col = blockIdx.x * BS + threadIdx.x * TS;

    for (int t = 0; t < K; t += BS) {   
        
        // load SMEM A: 32 * K = (K // 32) * (8 * 8 * 4 * 4)
        for (int i = 0; i < TS; ++i) { 
            for (int j = 0; j < TS; ++j) {
                int smem_row = threadIdx.y * TS + i;
                int smem_col = threadIdx.x * TS + j;

                int gmem_row = C_start_row + i; 
                int gmem_col = t + smem_col;    

                if (gmem_row < M && gmem_col < K) {
                    As[smem_row][smem_col] = A[gmem_row * K + gmem_col];
                } else {
                    As[smem_row][smem_col] = 0.0f;
                }
            }
        }

        // load SMEM B: 32 * K = (K // 32) * (8 * 8 * 4 * 4)
        for (int i = 0; i < TS; ++i) {
            for (int j = 0; j < TS; ++j) {
                int smem_row = threadIdx.y * TS + i;
                int smem_col = threadIdx.x * TS + j;

                int gmem_row = t + smem_row;    
                int gmem_col = C_start_col + j; 
                
                if (gmem_row < K && gmem_col < N) {
                    Bs[smem_row][smem_col] = B[gmem_row * N + gmem_col];
                } else {
                    Bs[smem_row][smem_col] = 0.0f;
                }
            }
        }

        __syncthreads();

        for (int k = 0; k < BS; ++k) { 
            for (int r = 0; r < TS; ++r) {
                for (int c = 0; c < TS; ++c) {
                    float A_val = As[threadIdx.y * TS + r][k];
                    float B_val = Bs[k][threadIdx.x * TS + c];
                    accs[r][c] += A_val * B_val;
                }
            }
        }

        __syncthreads();
    }

    for (int r = 0; r < TS; ++r) {
        for (int c = 0; c < TS; ++c) {
            int write_row = C_start_row + r;
            int write_col = C_start_col + c;
            if (write_row < M && write_col < N) {
                C[write_row * N + write_col] = accs[r][c];
            }
        }
    }
}

torch::Tensor matmul_2d_tiling(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix sizes for matmul");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(BS / TS, BS / TS);
    dim3 blocks((N + BS - 1) / BS, (M + BS - 1) / BS);

    matmul_2d_tiling_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}
"""

cpp_source = """
torch::Tensor matmul_2d_tiling(torch::Tensor A, torch::Tensor B);
"""

module = load_inline(
    name="matmul_2d_tiling_mod_fixed",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_2d_tiling"],
    verbose=True,
    extra_cuda_cflags=[
        "-Xptxas",
        "-v",
    ],
)

M, K, N = 1024, 1024, 1024
A = torch.randn(M, K, device="cuda").contiguous()
B = torch.randn(K, N, device="cuda").contiguous()

with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=0, active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        "./log/matmul_2d_tiling_fixed"
    ),
    record_shapes=False,
    with_stack=False,
) as prof:
    for i in range(3):
        with torch.no_grad():
            C_custom = module.matmul_2d_tiling(A, B)
        with torch.no_grad():
            C_torch = torch.matmul(A, B)
        prof.step()

assert torch.allclose(C_torch, C_custom, atol=1e-3)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
