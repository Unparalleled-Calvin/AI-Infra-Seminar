import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile, ProfilerActivity
import ctypes

cuda_source = """
#define BS 32 
#define TS 4  

__global__ void matmul_double_buffered_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[2][BS][BS];
    __shared__ float Bs[2][BS][BS];

    float accs[TS][TS] = {0.0f};

    const int C_start_row = blockIdx.y * BS + threadIdx.y * TS;
    const int C_start_col = blockIdx.x * BS + threadIdx.x * TS;

    int buf_idx = 0; 
    float A_val, B_val;


    // As[0]
    for (int i = 0; i < TS; ++i) {
        for (int j = 0; j < TS; ++j) {
            int smem_row = threadIdx.y * TS + i;
            int smem_col = threadIdx.x * TS + j;
            int gmem_row = C_start_row + i;
            int gmem_col = smem_col; // t=0
            if (gmem_row < M && gmem_col < K) 
                As[buf_idx][smem_row][smem_col] = A[gmem_row * K + gmem_col];
            else 
                As[buf_idx][smem_row][smem_col] = 0.0f;
        }
    }
    // Bs[0]
    for (int i = 0; i < TS; ++i) {
        for (int j = 0; j < TS; ++j) {
            int smem_row = threadIdx.y * TS + i;
            int smem_col = threadIdx.x * TS + j;
            int gmem_row = smem_row; // t=0
            int gmem_col = C_start_col + j;
            if (gmem_row < K && gmem_col < N) 
                Bs[buf_idx][smem_row][smem_col] = B[gmem_row * N + gmem_col];
            else 
                Bs[buf_idx][smem_row][smem_col] = 0.0f; 
        }
    }

    __syncthreads();

    for (int t = 0; t < K - BS; t += BS) {
        buf_idx = 1 - buf_idx;
        
        for (int i = 0; i < TS; ++i) {
            for (int j = 0; j < TS; ++j) {
                int smem_row = threadIdx.y * TS + i;
                int smem_col = threadIdx.x * TS + j;
                int gmem_row = C_start_row + i;
                int gmem_col = t + BS + smem_col;
                if (gmem_row < M && gmem_col < K) 
                    As[buf_idx][smem_row][smem_col] = A[gmem_row * K + gmem_col];
                else 
                    As[buf_idx][smem_row][smem_col] = 0.0f; 
            }
        }
        for (int i = 0; i < TS; ++i) {
            for (int j = 0; j < TS; ++j) {
                int smem_row = threadIdx.y * TS + i;
                int smem_col = threadIdx.x * TS + j;
                int gmem_row = t + BS + smem_row;
                int gmem_col = C_start_col + j;
                if (gmem_row < K && gmem_col < N)
                    Bs[buf_idx][smem_row][smem_col] = B[gmem_row * N + gmem_col];
                else 
                    Bs[buf_idx][smem_row][smem_col] = 0.0f;
            }
        }

        int compute_buf_idx = 1 - buf_idx;
        for (int k = 0; k < BS; ++k) {
            for (int r = 0; r < TS; ++r) {
                for (int c = 0; c < TS; ++c) {
                    A_val = As[compute_buf_idx][threadIdx.y * TS + r][k];
                    B_val = Bs[compute_buf_idx][k][threadIdx.x * TS + c];
                    accs[r][c] += A_val * B_val;
                }
            }
        }
        
        __syncthreads();
    }

    for (int k = 0; k < BS; ++k) {
        for (int r = 0; r < TS; ++r) {
            for (int c = 0; c < TS; ++c) {
                A_val = As[buf_idx][threadIdx.y * TS + r][k];
                B_val = Bs[buf_idx][k][threadIdx.x * TS + c];
                accs[r][c] += A_val * B_val;
            }
        }
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
    
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    dim3 threads(BS / TS, BS / TS);
    dim3 blocks((N + BS - 1) / BS, (M + BS - 1) / BS);

    matmul_double_buffered_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    return C;
}
"""

cpp_source = """
torch::Tensor matmul_2d_tiling(torch::Tensor A, torch::Tensor B);
"""

module = load_inline(
    name="matmul_double_buffered_mod",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_2d_tiling"],
    extra_cuda_cflags=[
        "-Xptxas",
        "-v",
    ],
    verbose=True,
)

M, K, N = 1024, 1024, 1024
A = torch.randn(M, K, device="cuda").contiguous()
B = torch.randn(K, N, device="cuda").contiguous()

with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=0, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        "./log/matmul_double_buffered"
    ),
    record_shapes=False,
    with_stack=False,
) as prof:
    for i in range(4):
        with torch.no_grad():
            C_custom = module.matmul_2d_tiling(A, B)
        with torch.no_grad():
            C_torch = torch.matmul(A, B)
        prof.step()

assert torch.allclose(C_torch, C_custom, atol=1e-3)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
