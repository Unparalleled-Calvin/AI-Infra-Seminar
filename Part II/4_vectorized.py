import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile, ProfilerActivity

cuda_source = """
#define BS 32 
#define TS 4  
#define VEC 4 

__global__ void matmul_vectorized_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const float4* A_vec = reinterpret_cast<const float4*>(A);
    const float4* B_vec = reinterpret_cast<const float4*>(B);
    float4* C_vec = reinterpret_cast<float4*>(C);

    __shared__ float As[BS][BS];
    __shared__ float Bs[BS][BS];

    float accs[TS][TS] = {0.0f};

    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    for (int t = 0; t < K; t += BS) {   
        
        for (int i = 0; i < TS; ++i) {
            int gmem_A_row = block_row * BS + thread_row * TS + i;
            int gmem_A_col_f4 = (t / VEC) + thread_col;
            
            int smem_A_row = thread_row * TS + i;
            int smem_A_col_start = thread_col * TS;

            if (gmem_A_row < M) {
                float4 temp_A = A_vec[gmem_A_row * (K / VEC) + gmem_A_col_f4];
                As[smem_A_row][smem_A_col_start + 0] = temp_A.x;
                As[smem_A_row][smem_A_col_start + 1] = temp_A.y;
                As[smem_A_row][smem_A_col_start + 2] = temp_A.z;
                As[smem_A_row][smem_A_col_start + 3] = temp_A.w;
            } else {
                As[smem_A_row][smem_A_col_start + 0] = 0.0f;
                As[smem_A_row][smem_A_col_start + 1] = 0.0f;
                As[smem_A_row][smem_A_col_start + 2] = 0.0f;
                As[smem_A_row][smem_A_col_start + 3] = 0.0f;
            }
        }

        for (int i = 0; i < TS; ++i) {
            int gmem_B_row = t + (thread_row * TS + i);
            int gmem_B_col_f4 = block_col * (BS / VEC) + thread_col;

            int smem_B_row = thread_row * TS + i;
            int smem_B_col_start = thread_col * TS;

            if (gmem_B_row < K) {
                float4 temp_B = B_vec[gmem_B_row * (N / VEC) + gmem_B_col_f4];
                Bs[smem_B_row][smem_B_col_start + 0] = temp_B.x;
                Bs[smem_B_row][smem_B_col_start + 1] = temp_B.y;
                Bs[smem_B_row][smem_B_col_start + 2] = temp_B.z;
                Bs[smem_B_row][smem_B_col_start + 3] = temp_B.w;
            } else {
                Bs[smem_B_row][smem_B_col_start + 0] = 0.0f;
                Bs[smem_B_row][smem_B_col_start + 1] = 0.0f;
                Bs[smem_B_row][smem_B_col_start + 2] = 0.0f;
                Bs[smem_B_row][smem_B_col_start + 3] = 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < BS; ++k) { 
            for (int r = 0; r < TS; ++r) {
                float A_val = As[thread_row * TS + r][k];
                for (int c = 0; c < TS; ++c) {
                    float B_val = Bs[k][thread_col * TS + c];
                    accs[r][c] += A_val * B_val;
                }
            }
        }
        __syncthreads();
    }

    const int C_start_row = block_row * BS + thread_row * TS;
    const int C_start_col = block_col * BS + thread_col * TS;
    for (int r = 0; r < TS; ++r) {
        int write_row = C_start_row + r;
        if (write_row < M) {
            int C_vec_col = C_start_col / VEC;
            if (C_start_col % VEC == 0) { 
                 C_vec[write_row * (N / VEC) + C_vec_col] = 
                    make_float4(accs[r][0], accs[r][1], accs[r][2], accs[r][3]);
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

    matmul_vectorized_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    return C;
}
"""

cpp_source = """
torch::Tensor matmul_2d_tiling(torch::Tensor A, torch::Tensor B);
"""

module = load_inline(
    name="matmul_vectorized_final_fix_mod",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_2d_tiling"],
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

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
