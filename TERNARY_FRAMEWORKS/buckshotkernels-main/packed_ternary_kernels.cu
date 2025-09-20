#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

/*
🚀 PACKED TERNARY CUDA KERNELS 🚀
2-bit packed ternary operations for 16x memory reduction
Direct bit manipulation with lookup tables for maximum speed
AUTHOR: SAAAM LLC | 2025 | Michael Wofford
*/

// Ternary encoding in 2 bits:
// -1 -> 00
//  0 -> 01
//  1 -> 10
// 11 -> reserved

#define TERNARY_NEG_ONE 0x0
#define TERNARY_ZERO    0x1
#define TERNARY_POS_ONE 0x2
#define TERNARY_RESERVED 0x3

// Lookup table for packed ternary multiplication
// Each entry handles 4 pairs of ternary values (1 byte each)
__constant__ int8_t packed_mul_lut[256];
__constant__ int16_t packed_dot_lut[256];

// Initialize lookup tables (called from host)
extern "C" void init_packed_ternary_lut() {
    int8_t mul_table[256];
    int16_t dot_table[256];

    for (int i = 0; i < 256; i++) {
        // Extract 4 ternary values from byte (2 bits each)
        int vals[4] = {
            (i >> 0) & 0x3,
            (i >> 2) & 0x3,
            (i >> 4) & 0x3,
            (i >> 6) & 0x3
        };

        // Decode to actual values
        float decoded[4];
        for (int j = 0; j < 4; j++) {
            switch (vals[j]) {
                case TERNARY_NEG_ONE: decoded[j] = -1.0f; break;
                case TERNARY_ZERO:    decoded[j] =  0.0f; break;
                case TERNARY_POS_ONE: decoded[j] =  1.0f; break;
                default:              decoded[j] =  0.0f; break;
            }
        }

        // Build multiplication lookup for first 2 pairs
        int mul_results[2];
        for (int j = 0; j < 2; j++) {
            float result = decoded[j] * decoded[j + 2];
            if (result < -0.5f)      mul_results[j] = TERNARY_NEG_ONE;
            else if (result > 0.5f)  mul_results[j] = TERNARY_POS_ONE;
            else                     mul_results[j] = TERNARY_ZERO;
        }

        mul_table[i] = (mul_results[0] | (mul_results[1] << 2));

        // Build dot product lookup (sum of all 4 multiplications)
        int16_t dot_sum = 0;
        for (int j = 0; j < 2; j++) {
            dot_sum += (int16_t)(decoded[j] * decoded[j + 2]);
        }
        dot_table[i] = dot_sum;
    }

    // Copy to GPU constant memory
    cudaMemcpyToSymbol(packed_mul_lut, mul_table, 256 * sizeof(int8_t));
    cudaMemcpyToSymbol(packed_dot_lut, dot_table, 256 * sizeof(int16_t));
}

// Pack float32 array into 2-bit ternary representation
extern "C" __global__ void pack_ternary_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float threshold,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_idx = idx / 4;  // 4 values per byte
    int bit_pos = (idx % 4) * 2;

    if (idx < size) {
        float val = input[idx];
        uint8_t encoded;

        if (fabsf(val) <= threshold) {
            encoded = TERNARY_ZERO;
        } else if (val > 0) {
            encoded = TERNARY_POS_ONE;
        } else {
            encoded = TERNARY_NEG_ONE;
        }

        // Atomic operation to set bits in packed array
        atomicOr(&output[output_idx], encoded << bit_pos);
    }
}

// Unpack 2-bit ternary back to float32
extern "C" __global__ void unpack_ternary_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int packed_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < packed_size) {
        uint8_t packed_byte = input[idx];

        // Unpack 4 values from this byte
        for (int i = 0; i < 4; i++) {
            int output_idx = idx * 4 + i;
            uint8_t encoded = (packed_byte >> (i * 2)) & 0x3;

            switch (encoded) {
                case TERNARY_NEG_ONE: output[output_idx] = -1.0f; break;
                case TERNARY_ZERO:    output[output_idx] =  0.0f; break;
                case TERNARY_POS_ONE: output[output_idx] =  1.0f; break;
                default:              output[output_idx] =  0.0f; break;
            }
        }
    }
}

// Ultra-fast packed ternary matrix multiplication
extern "C" __global__ void packed_ternary_matmul_kernel(
    const uint8_t* __restrict__ A_packed,    // Packed matrix A
    const uint8_t* __restrict__ B_packed,    // Packed matrix B
    int32_t* __restrict__ C,                 // Output matrix C
    int M, int N, int K
) {
    // Shared memory for tiled computation
    __shared__ uint8_t As[16][16];  // 16x16 tiles of packed data
    __shared__ uint8_t Bs[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t accumulator = 0;

    // Calculate packed dimensions (4 values per byte)
    int K_packed = (K + 3) / 4;
    int N_packed = (N + 3) / 4;

    // Loop over tiles
    for (int tile = 0; tile < (K_packed + 15) / 16; ++tile) {
        // Load A tile into shared memory
        int a_row = row;
        int a_col = tile * 16 + threadIdx.x;
        if (a_row < M && a_col < K_packed) {
            As[threadIdx.y][threadIdx.x] = A_packed[a_row * K_packed + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // Load B tile into shared memory
        int b_row = tile * 16 + threadIdx.y;
        int b_col = col;
        if (b_row < K_packed && b_col < N_packed) {
            Bs[threadIdx.y][threadIdx.x] = B_packed[b_row * N_packed + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute using lookup tables
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            uint8_t a_byte = As[threadIdx.y][k];
            uint8_t b_byte = Bs[k][threadIdx.x];

            // Combine bytes for lookup table access
            // Process 2 pairs at a time (4 values total)
            uint8_t lookup_idx = 0;

            // Interleave bits: a0a1b0b1 a2a3b2b3
            for (int bit_pair = 0; bit_pair < 2; bit_pair++) {
                uint8_t a_pair = (a_byte >> (bit_pair * 4)) & 0xF;
                uint8_t b_pair = (b_byte >> (bit_pair * 4)) & 0xF;
                lookup_idx = (a_pair << 4) | b_pair;

                // Use lookup table for dot product
                accumulator += packed_dot_lut[lookup_idx];
            }
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = accumulator;
    }
}

// Packed ternary convolution kernel
extern "C" __global__ void packed_ternary_conv2d_kernel(
    const uint8_t* __restrict__ input_packed,
    const uint8_t* __restrict__ weight_packed,
    int32_t* __restrict__ output,
    int N, int C, int H, int W,
    int Out_C, int KH, int KW,
    int Out_H, int Out_W,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    int n = blockIdx.z;
    int out_c = blockIdx.y;
    int out_h = blockIdx.x * blockDim.x + threadIdx.x;
    int out_w = threadIdx.y;

    if (n >= N || out_c >= Out_C || out_h >= Out_H || out_w >= Out_W) return;

    int32_t accumulator = 0;

    // Calculate packed dimensions
    int C_packed = (C + 3) / 4;
    int H_packed = H;  // Height doesn't change with packing
    int W_packed = (W + 3) / 4;

    // Convolution computation with packed data
    for (int in_c_pack = 0; in_c_pack < C_packed; ++in_c_pack) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw_pack = 0; kw_pack < (KW + 3) / 4; ++kw_pack) {
                int in_h = out_h * stride_h - pad_h + kh;
                int in_w_pack = (out_w * stride_w - pad_w) / 4 + kw_pack;

                // Bounds check
                if (in_h >= 0 && in_h < H_packed &&
                    in_w_pack >= 0 && in_w_pack < W_packed) {

                    // Get packed bytes
                    int input_idx = ((n * C_packed + in_c_pack) * H_packed + in_h) * W_packed + in_w_pack;
                    int weight_idx = ((out_c * C_packed + in_c_pack) * KH + kh) * ((KW + 3) / 4) + kw_pack;

                    uint8_t input_byte = input_packed[input_idx];
                    uint8_t weight_byte = weight_packed[weight_idx];

                    // Use lookup table for computation
                    uint8_t lookup_idx = (input_byte << 4) | weight_byte;
                    accumulator += packed_dot_lut[lookup_idx];
                }
            }
        }
    }

    // Write result
    int output_idx = ((n * Out_C + out_c) * Out_H + out_h) * Out_W + out_w;
    output[output_idx] = accumulator;
}

// Vectorized ternary operations using bit manipulation
extern "C" __global__ void packed_ternary_add_kernel(
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ B_packed,
    uint8_t* __restrict__ C_packed,
    int packed_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < packed_size) {
        uint8_t a_byte = A_packed[idx];
        uint8_t b_byte = B_packed[idx];
        uint8_t result_byte = 0;

        // Process 4 ternary additions per byte
        for (int i = 0; i < 4; i++) {
            uint8_t a_val = (a_byte >> (i * 2)) & 0x3;
            uint8_t b_val = (b_byte >> (i * 2)) & 0x3;

            // Decode, add, and re-encode
            int a_decoded = (a_val == TERNARY_NEG_ONE) ? -1 :
                           (a_val == TERNARY_POS_ONE) ? 1 : 0;
            int b_decoded = (b_val == TERNARY_NEG_ONE) ? -1 :
                           (b_val == TERNARY_POS_ONE) ? 1 : 0;

            int sum = a_decoded + b_decoded;
            uint8_t encoded_sum;

            if (sum <= -1)      encoded_sum = TERNARY_NEG_ONE;
            else if (sum >= 1)  encoded_sum = TERNARY_POS_ONE;
            else                encoded_sum = TERNARY_ZERO;

            result_byte |= (encoded_sum << (i * 2));
        }

        C_packed[idx] = result_byte;
    }
}

// Memory bandwidth test for packed operations
extern "C" __global__ void memory_bandwidth_test_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Simple copy to measure memory bandwidth
        output[idx] = input[idx];
    }
}

// Population count for sparsity analysis
extern "C" __global__ void popcount_sparsity_kernel(
    const uint8_t* __restrict__ packed_data,
    int* __restrict__ zero_counts,
    int packed_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < packed_size) {
        uint8_t byte_val = packed_data[idx];
        int local_zeros = 0;

        // Count zeros in this byte (4 ternary values)
        for (int i = 0; i < 4; i++) {
            uint8_t val = (byte_val >> (i * 2)) & 0x3;
            if (val == TERNARY_ZERO) {
                local_zeros++;
            }
        }

        // Atomic add to global counter
        atomicAdd(zero_counts, local_zeros);
    }
}