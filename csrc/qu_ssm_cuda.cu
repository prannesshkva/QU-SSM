/*
 * QU-SSM Native CUDA C++ Acceleration Kernel
 * 
 * Implements the fused Quasi-Unitary State Space Model recurrence in hardware registers
 * and Shared Memory, eliminating redundant DRAM reads and large intermediate allocations.
 *
 * Mathematical formulation:
 *   h_t = \gamma_t \cdot e^{i \theta_t} h_{t-1} + u_t
 *   where \gamma_t = \exp(\Delta S_t),  \Delta S_t = S_t - S_{t-1}
 *   S_t = raw_S_t / (1.0 - raw_S_t / -12.0)  (Algebraic Soft-Saturating Gradient Highway)
 *   \theta_t = (\theta_{proj, t} + \theta_{base}) \pmod{2\pi}
 *
 * Hardware Design:
 *   - Shared Memory staging: Broadcasts u, gate, and theta across the 8 quantum heads,
 *     eliminating 8x redundant global memory loads.
 *   - Fast trigonometric calculation via sincosf (hardware SFU).
 *   - Modulo 2*pi phase wrapping to prevent FP32 precision drift on S^1.
 *   - Warp-level reduction across the 8 quantum heads via __shfl_down_sync.
 *
 * Author: Prannesshkva (QU-SSM Core Team)
 *
 * SCOPE / LIMITATIONS (read before wiring this into training):
 *   - This file implements ONLY the forward recurrence. There is no backward
 *     kernel and no custom torch::autograd::Function wrapping it, so calling
 *     this op inside a graph that requires gradients will not produce correct
 *     (or any) gradients for u / raw_gate / theta_proj / theta_base.
 *   - It must only be invoked under torch.no_grad() / model.eval() (see the
 *     qu_ssm_cuda_ext.py wrapper, which enforces this). Training must
 *     continue to use the pure-PyTorch scan in modeling_qu_ssm.py, which is
 *     the mathematically-verified reference implementation.
 *   - N_HEADS is compiled in as a constant (8) and must equal config.d_state.
 *     A mismatched d_state will silently compute the wrong answer, so a
 *     runtime check is added in qu_ssm_fused_fwd_cuda() below.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

constexpr int N_HEADS = 8;
constexpr int CHANNELS_PER_BLOCK = 16;
constexpr int THREADS_PER_BLOCK = CHANNELS_PER_BLOCK * N_HEADS; // 128 threads = 4 warps
constexpr float TWO_PI = 6.283185307179586f;

__device__ __forceinline__ float smooth_decay(float raw_S) {
    // S = raw_S / (1.0 + raw_S / S_min) with S_min = -12.0f
    // Guarantees S in (-12, 0] with zero overflow risk: exp(-S) < exp(12) ≈ 162,754
    // Derivative dS/d(raw_S) = 1 / (1.0 + raw_S / S_min)^2 > 0 everywhere!
    return raw_S / (1.0f + raw_S / -12.0f);
}

template <typename scalar_t>
__global__ void qu_ssm_fused_fwd_kernel(
    const scalar_t* __restrict__ u,          // [B, L, D]
    const scalar_t* __restrict__ raw_gate,   // [B, L, D]
    const scalar_t* __restrict__ theta_proj, // [B, L, D]
    const float*    __restrict__ theta_base, // [N_HEADS]
    scalar_t*       __restrict__ out,        // [B, L, D]
    int B, int L, int D
) {
    int b = blockIdx.y;
    int ch_in_block = threadIdx.x / N_HEADS;
    int head_id = threadIdx.x % N_HEADS;
    int d = blockIdx.x * CHANNELS_PER_BLOCK + ch_in_block;

    if (b >= B || d >= D) return;

    // Shared Memory staging: 1 load per channel broadcast to all 8 heads
    __shared__ float s_u[CHANNELS_PER_BLOCK];
    __shared__ float s_g[CHANNELS_PER_BLOCK];
    __shared__ float s_th[CHANNELS_PER_BLOCK];

    float h_re = 0.0f;
    float h_im = 0.0f;
    float raw_S = 0.0f;
    float S = 0.0f;

    float my_theta_base = theta_base[head_id];
    int b_stride = L * D;

    // Process sequence tokens along time dimension
    for (int t = 0; t < L; ++t) {
        int idx = b * b_stride + t * D + d;

        // Cooperative load: Head 0 loads from DRAM into Shared Memory
        if (head_id == 0) {
            s_u[ch_in_block]  = static_cast<float>(u[idx]);
            s_g[ch_in_block]  = static_cast<float>(raw_gate[idx]);
            s_th[ch_in_block] = static_cast<float>(theta_proj[idx]);
        }
        __syncthreads();

        float u_val  = s_u[ch_in_block];
        float g_val  = s_g[ch_in_block];
        float th_val = s_th[ch_in_block];
        __syncthreads();

        // Numerically stable softplus: log_g = -softplus(g_val)
        float sp = (g_val > 20.0f) ? g_val : log1pf(expf(g_val));
        float log_g = -sp;

        // Differentiable algebraic soft-decay highway (Theorem 2)
        raw_S += log_g;
        float next_S = smooth_decay(raw_S);
        float delta_S = next_S - S;
        float gamma = expf(delta_S);
        S = next_S;

        // Modulo 2*pi phase wrapping on S^1 prevents argument-reduction drift
        float theta = fmodf(th_val + my_theta_base, TWO_PI);
        float sin_th, cos_th;
        sincosf(theta, &sin_th, &cos_th);

        // Complex state rotation and input injection (Theorem 1)
        float next_re = gamma * (cos_th * h_re - sin_th * h_im) + u_val;
        float next_im = gamma * (sin_th * h_re + cos_th * h_im);
        h_re = next_re;
        h_im = next_im;

        // Warp reduction across the 8 quantum heads
        unsigned int lane_id = threadIdx.x % 32;
        unsigned int group_id = lane_id / 8;
        unsigned int mask = 0xFF << (group_id * 8);

        float sum_re = h_re;
        sum_re += __shfl_down_sync(mask, sum_re, 4);
        sum_re += __shfl_down_sync(mask, sum_re, 2);
        sum_re += __shfl_down_sync(mask, sum_re, 1);

        // Lane 0 of each 8-head group writes output to DRAM
        if (head_id == 0) {
            out[idx] = static_cast<scalar_t>(sum_re);
        }
    }
}

torch::Tensor qu_ssm_fused_fwd_cuda(
    torch::Tensor u,
    torch::Tensor raw_gate,
    torch::Tensor theta_proj,
    torch::Tensor theta_base
) {
    TORCH_CHECK(u.is_cuda() && raw_gate.is_cuda() && theta_proj.is_cuda() && theta_base.is_cuda(),
                "qu_ssm_fused_fwd_cuda: all inputs must be CUDA tensors");
    TORCH_CHECK(u.dim() == 3 && raw_gate.dim() == 3 && theta_proj.dim() == 3,
                "qu_ssm_fused_fwd_cuda: u/raw_gate/theta_proj must be [B, L, D]");
    TORCH_CHECK(theta_base.dim() == 1 && theta_base.size(0) == N_HEADS,
                "qu_ssm_fused_fwd_cuda: theta_base must be a 1D tensor of length N_HEADS (",
                N_HEADS, "). This kernel is compiled for d_state == N_HEADS; if your "
                "config.d_state != ", N_HEADS, ", this kernel is not valid for it.");
    TORCH_CHECK(!u.requires_grad() && !raw_gate.requires_grad() && !theta_proj.requires_grad(),
                "qu_ssm_fused_fwd_cuda: this kernel has no backward pass. Only call it "
                "under torch.no_grad() (inference). Use the pure-PyTorch scan for training.");
    TORCH_CHECK(theta_base.scalar_type() == torch::kFloat32,
                "qu_ssm_fused_fwd_cuda: theta_base must be float32");

    auto u_c = u.contiguous();
    auto raw_gate_c = raw_gate.contiguous();
    auto theta_proj_c = theta_proj.contiguous();
    auto theta_base_c = theta_base.contiguous();

    const int B = u_c.size(0);
    const int L = u_c.size(1);
    const int D = u_c.size(2);

    auto out = torch::empty_like(u_c);

    dim3 blocks((D + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK, B);
    dim3 threads(THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(u_c.scalar_type(), "qu_ssm_fwd_cuda", ([&] {
        qu_ssm_fused_fwd_kernel<scalar_t><<<blocks, threads>>>(
            u_c.data_ptr<scalar_t>(),
            raw_gate_c.data_ptr<scalar_t>(),
            theta_proj_c.data_ptr<scalar_t>(),
            theta_base_c.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            B, L, D
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "qu_ssm_fused_fwd_cuda: kernel launch failed: ",
                cudaGetErrorString(err));

    return out;
}

// This was missing entirely in the previous version of this file: without a
// pybind11 module, torch's cpp_extension has nothing to bind to Python, so
// `qu_ssm_fused_fwd_cuda` was unreachable from Python regardless of whether
// the kernel compiled. See qu_ssm_cuda_ext.py for the loader that uses this.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_fwd", &qu_ssm_fused_fwd_cuda,
          "QU-SSM fused forward recurrence (CUDA, inference-only, no backward)");
}
