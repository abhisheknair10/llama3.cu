#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference.cuh"
#include "llama3/llama3.cuh"

#define CHECK_CUDA_ERROR()                                       \
    {                                                            \
        cudaError_t err = cudaGetLastError();                    \
        if (err != cudaSuccess) {                                \
            printf("CUDA error: %s in file '%s' in line %i\n",   \
                   cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    }

#define TILE_SIZE 16
#define VOCAB_SIZE 128256

// Sampling config
#define TEMPERATURE 0.7f
#define TOP_K 50
#define TOP_P 0.9f

const int MAX_THREADS_PER_BLOCK = 1024;

int h_NUM_TOKENS;

/* ************************************ HELPERS ************************************ */
void free_tensor_cuda(Tensor *t) {
    cudaFree(t->d_ndim);
    cudaFree(t->d_mem_len);
    cudaFree(t->d_shape);
    cudaFree(t->d_fp16_tensor);

    return;
}

// Print CUDA memory info
void printCudaMemoryInfo() {
    size_t free_memory = 0;
    size_t total_memory = 0;

    // Get the amount of free and total memory on the GPU
    cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);

    if (err == cudaSuccess) {
        // Convert memory sizes from bytes to megabytes (MB)
        printf("Free GPU Memory: %.2f MB\n", (float)free_memory / (1024 * 1024));
        printf("Total GPU Memory: %.2f MB\n", (float)total_memory / (1024 * 1024));
    } else {
        printf("Failed to get CUDA memory info: %s\n", cudaGetErrorString(err));
    }

    return;
}

// Kernel to check and print the embeddings
/*
__global__ void check_embedding(__half *fp16_tensor, int dim, int num_tokens) {
    for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
        printf("Token %d embeddings:\n", token_idx);
        for (int i = 0; i < dim; ++i) {
            printf("%f, ", __half2float(fp16_tensor[token_idx * dim + i]));
        }
        printf("\n");
        printf("\n\n");
    }

    return;
}

__global__ void check_embedding(__half *fp16_tensor, int dim, int num_tokens) {
    for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
        printf("Token %d embeddings:\n", token_idx);
        int max = 0;
        float curr_max = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float embedding = __half2float(fp16_tensor[token_idx * dim + i]);

            if (embedding > curr_max) {
                curr_max = embedding;
                max = i;
            }
        }
        printf("%d\n", max);
        printf("\n\n");
    }

    return;
}
*/
/* ************************************* Cache ************************************* */
// Allocate global mem cache on device
void *create_gmemcache(size_t mem_len, size_t type_size) {
    void *d_gcache;

    cudaMalloc((void **)&d_gcache, mem_len * type_size);
    CHECK_CUDA_ERROR();

    return d_gcache;
}

CudaCache *init_cache(Llama3 *llama3_model) {
    // Ahead Of Time memory allocations
    // Allocate once, use everywhere
    CudaCache *Cache = (CudaCache *)malloc(sizeof(CudaCache));

    // Allocate Memory --------------------------------------------------------
    Tensor *PN_X = _create_intermediary_prenorm_tensor_copy();

    Tensor *Q = _create_intermediary_attention_tensor(llama3_model->layers[0]->self_attn_q_proj);
    Tensor *K = _create_intermediary_attention_tensor(llama3_model->layers[0]->self_attn_k_proj);
    Tensor *V = _create_intermediary_attention_tensor(llama3_model->layers[0]->self_attn_v_proj);

    float *d_attention_score_cache = (float *)create_gmemcache(32 * 2048 * 2048, sizeof(float));

    __half *d_feedforward_cache_upgate = (__half *)create_gmemcache(2048 * 14336, sizeof(__half));

    __half *d_token_probdist = (__half *)create_gmemcache(VOCAB_SIZE, sizeof(__half));
    __half *h_token_probdist = (__half *)malloc(VOCAB_SIZE * sizeof(__half));
    float *f_token_probdist = (float *)malloc(VOCAB_SIZE * sizeof(float));

    // Save pointers to Struct --------------------------------------------------------
    Cache->PN_X = PN_X;

    Cache->Q = Q;
    Cache->K = K;
    Cache->V = V;

    Cache->d_attention_score_cache = d_attention_score_cache;
    Cache->d_feedforward_cache_upgate = d_feedforward_cache_upgate;

    Cache->h_token_probdist = h_token_probdist;
    Cache->d_token_probdist = d_token_probdist;
    Cache->f_token_probdist = f_token_probdist;

    return Cache;
}

/* ********************************* Inference Code ********************************* */

int inference(Llama3 *llama3_model, Tensor *X, int *d_tokens, int *h_tokens, CudaCache *Cache) {
    cudaMemcpy(d_tokens, h_tokens, sizeof(int) * h_tokens[0], cudaMemcpyHostToDevice);

    // Set NUM_TOKENS value in device memory
    h_NUM_TOKENS = h_tokens[0] - 1;

    tokens_to_embeddings(X, llama3_model, d_tokens);

    for (int i = 0; i < llama3_model->n_layers; ++i) {
        // Pre-attention normalization
        _deviceMemcpy_fp16_tensor(Cache->PN_X, X);
        compute_layer_norm(llama3_model->layers[i]->input_layernorm, X);

        // Attention tensor computation
        compute_qkv_tensors(Cache->Q, Cache->K, Cache->V, llama3_model->layers[i], X);

        // RoPE scaling
        rope_scaling(Cache->Q, Cache->K);

        // Attention computation
        compute_attention(X, Cache->Q, Cache->K, Cache->V, Cache);

        // Output computation
        compute_output(llama3_model->layers[i], X, Cache);

        // Add pre-normalized input
        add_norm(X, Cache->PN_X);

        // Post-attention normalization
        _deviceMemcpy_fp16_tensor(Cache->PN_X, X);
        compute_layer_norm(llama3_model->layers[i]->post_attention_layernorm, X);

        // Feedforward
        compute_feedforward(X, llama3_model->layers[i], Cache);

        // Add pre-normalized input
        add_norm(X, Cache->PN_X);
    }

    compute_layer_norm(llama3_model->norm, X);

    int output = compute_lm_head(llama3_model->lm_head, X, Cache);

    return output;
}

/* ************************** Convert Tokens to Embeddings ************************** */
void tokens_to_embeddings(Tensor *X, Llama3 *llama3_model, int *d_tokens) {
    // Order threads into blocks
    int total_threads = *(X->mem_len);
    int blocks = (total_threads + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    kernel_tokens_to_embeddings<<<blocks, MAX_THREADS_PER_BLOCK>>>(
        X->d_fp16_tensor, d_tokens, llama3_model->embed_tokens->d_fp16_tensor,
        h_NUM_TOKENS);

    return;
}

__global__ void kernel_tokens_to_embeddings(__half *X, int *tokens, __half *Embed, int num_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_elements = num_tokens * 4096;

    if (idx >= total_elements) return;

    int token_idx = idx / 4096;
    int embed_idx = idx % 4096;

    X[(token_idx * 4096) + embed_idx] =
        Embed[(tokens[token_idx + 1] * 4096) + embed_idx];

    return;
}

/* ******************************* Layer Normalization ******************************* */
// Helpers
Tensor *_create_intermediary_prenorm_tensor_copy() {
    Tensor *Y = (Tensor *)malloc(sizeof(Tensor));

    int *d_ndim;
    int *d_mem_len;
    int *d_shape;
    __half *d_fp16_tensor;

    Y->ndim = (int *)malloc(sizeof(int));
    *(Y->ndim) = 2;

    Y->mem_len = (int *)malloc(sizeof(int));
    *(Y->mem_len) = 2048 * 4096;

    Y->shape = (int *)malloc(sizeof(int) * 2);
    Y->shape[0] = 2048;
    Y->shape[1] = 4096;

    // Allocate CUDA memory
    cudaMalloc((void **)&d_ndim, sizeof(int));
    cudaMalloc((void **)&d_mem_len, sizeof(int));
    cudaMalloc((void **)&d_shape, sizeof(int) * (*(Y->ndim)));
    cudaMalloc((void **)&d_fp16_tensor, sizeof(__half) * (*(Y->mem_len)));

    // Copy data to device
    cudaMemcpy(d_ndim, Y->ndim, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mem_len, Y->mem_len, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, Y->shape, sizeof(int) * (*(Y->ndim)), cudaMemcpyHostToDevice);
    cudaMemset(d_fp16_tensor, __float2half(0.0f), sizeof(__half) * (*(Y->mem_len)));

    // Assign device pointers
    Y->d_ndim = d_ndim;
    Y->d_mem_len = d_mem_len;
    Y->d_shape = d_shape;
    Y->d_fp16_tensor = d_fp16_tensor;

    return Y;
}

void _deviceMemcpy_fp16_tensor(Tensor *Y, Tensor *X) {
    if (*(X->mem_len) > *(Y->mem_len)) {
        printf("X and Y are not alike Tensors: (_deviceMemcpy_fp16_tensor)");
        exit(1);
    };

    cudaMemcpy(
        Y->d_fp16_tensor,
        X->d_fp16_tensor,
        sizeof(__half) * (*(X->mem_len)),
        cudaMemcpyDeviceToDevice);

    return;
}

// Compute RMS Norm
void compute_layer_norm(Tensor *RMSNorm, Tensor *X) {
    dim3 block(32, 32);
    dim3 grid(h_NUM_TOKENS);

    kernel_compute_rms_norm<<<grid, block>>>(
        X->d_fp16_tensor, RMSNorm->d_fp16_tensor, h_NUM_TOKENS);

    return;
}

__global__ void kernel_compute_rms_norm(__half *X, __half *RMSNorm, int num_tokens) {
    __shared__ float shared_mem[1024];

    int token_idx = blockIdx.x;
    int vw_embed_idx = threadIdx.y * blockDim.x + threadIdx.x;

    if (token_idx >= num_tokens) return;
    if (vw_embed_idx >= 1024) return;

    /*
        - Coalesced load into shared memory of 1024 window with vectorized retrieval
        - A 1024 thread block is used to retrieve 4096 elements. Each thread retrieves consecutive
            indicies. Instead of looping and having 4 separate memory access transactions for each
            window retrieval per thread, a singular call loading 4 __half's as 1 uint64_t allows for
            4 indicies to be retrieved virtually as one data type.
    */
    c_half4 data = ((c_half4 *)X)[token_idx * 1024 + vw_embed_idx];
    shared_mem[vw_embed_idx] = __half2float(data.x) * __half2float(data.x) +
                               __half2float(data.y) * __half2float(data.y) +
                               __half2float(data.z) * __half2float(data.z) +
                               __half2float(data.w) * __half2float(data.w);
    __syncthreads();

    /*
        - Parallel reduction along y-axis (maximize warp usage without warp divergence)
        - For a 32 x 32 block dimension, the 1st warp will sum with the 16th warp and
            recursively reduce
    */
    for (int offset = 512; offset >= 32; offset /= 2) {
        if (vw_embed_idx < offset) {
            shared_mem[vw_embed_idx] += shared_mem[offset + vw_embed_idx];
        }
        __syncthreads();
    }

    /*
        - Parallel reduction for 1 warp (divergent warp behavior) without using shared memory
        - Warp level primitive usage
        - Instead of utilizing shared memory to store intermediate reduction sums, inter-thread
            memory access enables faster reduction
        - For a given warp, the following will still not diverge with 0xffffff mask enabling the
            same instruction for every thread in the warp
        - Offset enables reduction to happen with left most indices lasting the longest. Least
            significant indices still perform addition but add no value to context
    */
    if (vw_embed_idx < 32) {
        float val = shared_mem[vw_embed_idx];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (vw_embed_idx == 0) shared_mem[0] = val;
    }
    __syncthreads();

    /*
        - Load rms norm for tensor and perform normalization for 1024 window
        - Similar technique to when loading data from global memory
    */
    float rms = sqrtf(1e-5 + (shared_mem[0] / 4096.0f));
    c_half4 norm_gain = ((c_half4 *)RMSNorm)[vw_embed_idx];

    // Perform RMS calculations and store
    data.x = __float2half(__half2float(data.x) * __half2float(norm_gain.x) / rms);
    data.y = __float2half(__half2float(data.y) * __half2float(norm_gain.y) / rms);
    data.z = __float2half(__half2float(data.z) * __half2float(norm_gain.z) / rms);
    data.w = __float2half(__half2float(data.w) * __half2float(norm_gain.w) / rms);

    ((c_half4 *)X)[token_idx * 1024 + vw_embed_idx] = data;

    return;
}

// Compute addition (skip connection)
void add_norm(Tensor *X, Tensor *PN_X) {
    dim3 block(1024);
    dim3 grid(4, h_NUM_TOKENS);

    kernel_add_norm<<<grid, block>>>(
        X->d_fp16_tensor, PN_X->d_fp16_tensor, h_NUM_TOKENS);

    return;
}

__global__ void kernel_add_norm(__half *X, __half *PN_X, int num_tokens) {
    int token_idx = blockIdx.y;
    int embed_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx >= num_tokens) return;
    if (embed_idx >= 4096) return;

    int offset = token_idx * 4096 + embed_idx;
    float a = __half2float(X[offset]);
    float b = __half2float(PN_X[offset]);
    X[offset] = __float2half(a + b);

    return;
}

/* ***************************** General Matrix Multiplication **************************** */
__global__ void kernel_standard_tiled_gemm(
    __half *O, __half *X, __half *Transform, int m, int n, int k, int tile_size) {
    /*
        - m represents the independent dimension of the input matrix
        - n represents the independent dimenion of the transformation matrix
        - k represents the common dimension of the 2 matrices
        - Within each kernel, the output is computed as: O = matmul(X, Transform)
        - Transposing the transformation tensor is not required as virtual indexing allows
          for intended navigation along rows and columns of either tensors
        - Order of variables within kernels obey order of computation
    */
    // Kernel start
    //
    extern __shared__ half2 _shared_mem[];
    half2 *X_shmem = _shared_mem;
    half2 *T_shmem = _shared_mem + tile_size * tile_size;

    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;

    int rowmaj_col_offset = blockIdx.x * tile_size + threadIdx.y;

    // Loop over tiles
    float value = 0.0f;
    int half2_k = (k + 1) / 2;
    for (int t = 0; t < (half2_k + tile_size - 1) / tile_size; ++t) {
        // Load tile of X into shared memory
        if (row < m) {
            int X_idx = row * half2_k + t * tile_size + threadIdx.x;
            X_shmem[threadIdx.y * tile_size + threadIdx.x] = ((half2 *)X)[X_idx];
        } else {
            X_shmem[threadIdx.y * tile_size + threadIdx.x] = __float2half2_rn(0.0f);
        }

        // Load tile of Transform into shared memory
        if (col < n) {
            int T_idx = rowmaj_col_offset * half2_k + t * tile_size + threadIdx.x;
            T_shmem[threadIdx.x * tile_size + threadIdx.y] = ((half2 *)Transform)[T_idx];
        } else {
            T_shmem[threadIdx.x * tile_size + threadIdx.y] = __float2half2_rn(0.0f);
        }
        __syncthreads();

        for (int i = 0; i < tile_size; ++i) {
            value += __half2float(__low2half(X_shmem[threadIdx.y * tile_size + i])) * __half2float(__low2half(T_shmem[i * tile_size + threadIdx.x]));
            value += __half2float(__high2half(X_shmem[threadIdx.y * tile_size + i])) * __half2float(__high2half(T_shmem[i * tile_size + threadIdx.x]));
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (row < m && col < n) {
        O[row * n + col] = __float2half(value);
    }

    return;
}

__device__ float SiLU(float x) {
    return x / (1 + expf(-x));
}

__global__ void kernel_standard_tiled_gemm_ffn_swiglu(
    __half *O, __half *X, __half *Wup, __half *Wgate, int m, int n, int k, int tile_size) {
    /*
        - m represents the independent dimension of the input matrix
        - n represents the independent dimenion of the transformation matrix
        - k represents the common dimension of the 2 matrices
        - Within each kernel, the output is computed as: O = matmul(X, Transform)
        - Transposing the transformation tensor is not required as virtual indexing allows
          for intended navigation along rows and columns of either tensors
        - Order of variables within kernels obey order of computation
    */
    // Kernel start
    //
    extern __shared__ half2 _shared_mem[];
    half2 *X_shmem = _shared_mem;
    half2 *Up_shmem = _shared_mem + tile_size * tile_size;
    half2 *Gate_shmem = _shared_mem + 2 * tile_size * tile_size;

    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;

    int rowmaj_col_offset = blockIdx.x * tile_size + threadIdx.y;

    // Loop over tiles
    float value_up = 0.0f;
    float value_gate = 0.0f;
    int half2_k = (k + 1) / 2;
    for (int t = 0; t < (half2_k + tile_size - 1) / tile_size; ++t) {
        // Load tile of X into shared memory
        if (row < m) {
            int X_idx = row * half2_k + t * tile_size + threadIdx.x;
            X_shmem[threadIdx.y * tile_size + threadIdx.x] = ((half2 *)X)[X_idx];
        } else {
            X_shmem[threadIdx.y * tile_size + threadIdx.x] = __float2half2_rn(0.0f);
        }

        // Load tile of Transform into shared memory
        if (col < n) {
            int T_idx = rowmaj_col_offset * half2_k + t * tile_size + threadIdx.x;

            Up_shmem[threadIdx.x * tile_size + threadIdx.y] = ((half2 *)Wup)[T_idx];
            Gate_shmem[threadIdx.x * tile_size + threadIdx.y] = ((half2 *)Wgate)[T_idx];
        } else {
            Up_shmem[threadIdx.x * tile_size + threadIdx.y] = __float2half2_rn(0.0f);
            Gate_shmem[threadIdx.x * tile_size + threadIdx.y] = __float2half2_rn(0.0f);
        }
        __syncthreads();

        for (int i = 0; i < tile_size; ++i) {
            value_up += __half2float(__low2half(X_shmem[threadIdx.y * tile_size + i])) * __half2float(__low2half(Up_shmem[i * tile_size + threadIdx.x]));
            value_up += __half2float(__high2half(X_shmem[threadIdx.y * tile_size + i])) * __half2float(__high2half(Up_shmem[i * tile_size + threadIdx.x]));

            value_gate += __half2float(__low2half(X_shmem[threadIdx.y * tile_size + i])) * __half2float(__low2half(Gate_shmem[i * tile_size + threadIdx.x]));
            value_gate += __half2float(__high2half(X_shmem[threadIdx.y * tile_size + i])) * __half2float(__high2half(Gate_shmem[i * tile_size + threadIdx.x]));
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (row < m && col < n) {
        O[row * n + col] = __float2half(SiLU(value_gate) * value_up);
    }

    return;
}

/* ***************************** Attention Tensor Computation **************************** */
Tensor *_create_intermediary_attention_tensor(Tensor *Linear) {
    Tensor *Attention_Tensor = (Tensor *)malloc(sizeof(Tensor));

    int *d_ndim;
    int *d_mem_len;
    int *d_shape;
    __half *d_fp16_tensor;

    Attention_Tensor->ndim = (int *)malloc(sizeof(int));
    *(Attention_Tensor->ndim) = 2;

    Attention_Tensor->mem_len = (int *)malloc(sizeof(int));
    *(Attention_Tensor->mem_len) = 2048 * Linear->shape[0];

    Attention_Tensor->shape = (int *)malloc(sizeof(int) * 2);
    Attention_Tensor->shape[0] = 2048;
    Attention_Tensor->shape[1] = Linear->shape[0];

    // Allocate CUDA memory
    cudaMalloc((void **)&d_ndim, sizeof(int));
    cudaMalloc((void **)&d_mem_len, sizeof(int));
    cudaMalloc((void **)&d_shape, sizeof(int) * 2);
    cudaMalloc((void **)&d_fp16_tensor, sizeof(__half) * (*(Attention_Tensor->mem_len)));

    // Copy data to device
    cudaMemcpy(d_ndim, Attention_Tensor->ndim, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mem_len, Attention_Tensor->mem_len, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, Attention_Tensor->shape, sizeof(int) * 2, cudaMemcpyHostToDevice);
    cudaMemset(d_fp16_tensor, __float2half(0.0f), sizeof(__half) * (*(Attention_Tensor->mem_len)));

    // Assign device pointers
    Attention_Tensor->d_ndim = d_ndim;
    Attention_Tensor->d_mem_len = d_mem_len;
    Attention_Tensor->d_shape = d_shape;
    Attention_Tensor->d_fp16_tensor = d_fp16_tensor;

    return Attention_Tensor;
}

void compute_qkv_tensors(
    Tensor *Q, Tensor *K, Tensor *V,
    Llama3Layer *L3_Layer, Tensor *X) {
    // Declare common variables
    size_t shared_mem_size = 2 * TILE_SIZE * (TILE_SIZE + 1) * sizeof(float);
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid;

    // Query computation
    grid = dim3(
        (L3_Layer->self_attn_q_proj->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE);

    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        Q->d_fp16_tensor, X->d_fp16_tensor, L3_Layer->self_attn_q_proj->d_fp16_tensor,
        h_NUM_TOKENS, L3_Layer->self_attn_q_proj->shape[0], 4096, TILE_SIZE);

    // Key computation
    grid = dim3(
        (L3_Layer->self_attn_k_proj->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE);

    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        K->d_fp16_tensor, X->d_fp16_tensor, L3_Layer->self_attn_k_proj->d_fp16_tensor,
        h_NUM_TOKENS, L3_Layer->self_attn_k_proj->shape[0], 4096, TILE_SIZE);

    // Value computation
    grid = dim3(
        (L3_Layer->self_attn_v_proj->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE);

    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        V->d_fp16_tensor, X->d_fp16_tensor, L3_Layer->self_attn_v_proj->d_fp16_tensor,
        h_NUM_TOKENS, L3_Layer->self_attn_v_proj->shape[0], 4096, TILE_SIZE);

    return;
}

void compute_output(Llama3Layer *L3_Layer, Tensor *X, CudaCache *Cache) {
    // Declare common variables
    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid;

    // Output computation
    grid = dim3(
        (L3_Layer->self_attn_o_proj->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE);
    _deviceMemcpy_fp16_tensor(Cache->Q, X);

    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        X->d_fp16_tensor, Cache->Q->d_fp16_tensor, L3_Layer->self_attn_o_proj->d_fp16_tensor,
        h_NUM_TOKENS, L3_Layer->self_attn_o_proj->shape[0], 4096, TILE_SIZE);

    return;
}

/* ************************* Rotary Positional Embedding (RoPE) ************************* */
void rope_scaling(Tensor *Q, Tensor *K) {
    dim3 block;
    dim3 grid;

    // RoPE on Q
    block = dim3(1024);
    grid = dim3(2, h_NUM_TOKENS);
    kernel_rope_scaling<<<grid, block>>>(Q->d_fp16_tensor, 2048, h_NUM_TOKENS);

    // RoPE on K
    block = dim3(256);
    grid = dim3(2, h_NUM_TOKENS);
    kernel_rope_scaling<<<grid, block>>>(K->d_fp16_tensor, 512, h_NUM_TOKENS);

    return;
}

__global__ void kernel_rope_scaling(__half *tensor, int transformed_embed_size, int num_tokens) {
    /*
    - For Q [tokens, 4096], transformed_embed_size = 2048
      Block size: 1024 threads
      Grid: 2 blocks (2048 / 1024 = 2) per token
    - For K [tokens, 1024], transformed_embed_size = 512
      Block size: 256 threads
      Grid: 2 blocks (512 / 256 = 2) per token
    */

    // Calculate the token index and embedding index
    int token_idx = blockIdx.y;
    int embed_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary checks
    if (embed_idx >= transformed_embed_size) return;
    if (token_idx >= num_tokens) return;

    // Cast tensor to __half2 pointer using C-style cast
    __half2 *tensor_half2 = (__half2 *)tensor;

    // Access the __half2 value for the current token and embedding index
    __half2 h2_val = tensor_half2[token_idx * transformed_embed_size + embed_idx];

    // Split __half2 into two __half values
    __half h_even = __low2half(h2_val);
    __half h_odd = __high2half(h2_val);

    // Convert __half to float for computation
    float even = __half2float(h_even);
    float odd = __half2float(h_odd);

    // Frequency calculation with floating-point division
    float freq = 1.0f / powf(500000.0f, 2.0f * (float)(embed_idx % 64) / 128.0f);
    float angle = (float)token_idx * freq;

    // Compute cosine and sine components
    float cos_comp = cosf(angle);
    float sin_comp = sinf(angle);

    // Apply rotation to the embedding
    float ret_even = (cos_comp * even) - (sin_comp * odd);
    float ret_odd = (sin_comp * even) + (cos_comp * odd);

    // Convert the rotated values back to __half
    __half h_ret_even = __float2half(ret_even);
    __half h_ret_odd = __float2half(ret_odd);

    // Combine the two __half values back into a single __half2
    __half2 h2_result = __halves2half2(h_ret_even, h_ret_odd);

    // Store the result back into the tensor
    tensor_half2[token_idx * transformed_embed_size + embed_idx] = h2_result;

    return;
}

/* **************************** Grouped Multi-Query Attention **************************** */
void compute_attention(Tensor *X, Tensor *Q, Tensor *K, Tensor *V, CudaCache *Cache) {
    // Attention score computation
    int nheads = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE,
        nheads);

    size_t shared_mem = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    kernel_compute_masked_gmq_attention_scores_tiled_matmul<<<grid, block, shared_mem>>>(
        Cache->d_attention_score_cache, Q->d_fp16_tensor, K->d_fp16_tensor,
        h_NUM_TOKENS, h_NUM_TOKENS, 128, TILE_SIZE, nheads);

    block = dim3(1024);
    grid = dim3(h_NUM_TOKENS, nheads);

    shared_mem = (2048 + 1024) * sizeof(float);
    kernel_masking_softmax<<<grid, block, shared_mem>>>(
        Cache->d_attention_score_cache, h_NUM_TOKENS);

    block = dim3(TILE_SIZE, TILE_SIZE);
    grid = dim3(
        (128 + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE,
        nheads);

    shared_mem = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    kernel_compute_resolved_value_from_attention_score_tiled_matmul<<<grid, block, shared_mem>>>(
        X->d_fp16_tensor, Cache->d_attention_score_cache, V->d_fp16_tensor,
        h_NUM_TOKENS, 128, h_NUM_TOKENS, nheads, TILE_SIZE);

    return;
}

__global__ void kernel_compute_masked_gmq_attention_scores_tiled_matmul(
    float *attention_scores, __half *Q, __half *K,
    int m, int n, int k, int tile_size, int nheads) {
    /*
        - Each head operates independently of other heads.
        - `m`: Number of tokens (rows of Q).
        - `n`: Number of tokens (columns of K).
        - `k`: Head dimension (common dimension).
        - `nheads`: Number of attention heads.
    */

    extern __shared__ float shared_mem[];
    float *Q_shmem = shared_mem;
    float *K_shmem = shared_mem + (tile_size * tile_size);

    int q_head_idx = blockIdx.z;
    int kv_head_idx = q_head_idx / 4;
    int kv_heads = nheads / 4;

    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;

    float value = 0.0f;
    for (int t = 0; t < (k + tile_size - 1) / tile_size; ++t) {
        if (row < m && (t * tile_size + threadIdx.x) < k) {
            int Q_idx = row * (nheads * k) + (q_head_idx * k) + t * tile_size + threadIdx.x;
            Q_shmem[threadIdx.y * tile_size + threadIdx.x] = __half2float(Q[Q_idx]);
        } else {
            Q_shmem[threadIdx.y * tile_size + threadIdx.x] = 0.0f;
        }

        if (col < n && (t * tile_size + threadIdx.y) < k) {
            int K_idx = col * (kv_heads * k) + (kv_head_idx * k) + t * tile_size + threadIdx.y;
            K_shmem[threadIdx.y * tile_size + threadIdx.x] = __half2float(K[K_idx]);
        } else {
            K_shmem[threadIdx.y * tile_size + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute partial sums
        for (int i = 0; i < tile_size; ++i) {
            value += Q_shmem[threadIdx.y * tile_size + i] * K_shmem[i * tile_size + threadIdx.x];
        }
        __syncthreads();
    }

    // Write result to memory
    if (row < m && col < n) {
        attention_scores[q_head_idx * m * n + row * n + col] = value / sqrtf(k);
    }

    return;
}

__global__ void kernel_masking_softmax(float *attention_scores, int num_tokens) {
    extern __shared__ float shared_mem[];
    float *vec = shared_mem;
    float *buffer = shared_mem + 2048;

    int token_idx_y = blockIdx.x;
    int head_idx = blockIdx.y;

    if (token_idx_y >= num_tokens) return;
    if (head_idx >= 32) return;

    int token_idx_x;
    float max_val = -FLT_MAX;

    for (int i = 0; i < (num_tokens + blockDim.x - 1) / blockDim.x; ++i) {
        token_idx_x = i * blockDim.x + threadIdx.x;

        if (token_idx_x < num_tokens) {
            if (token_idx_x <= token_idx_y) {
                float score = attention_scores[(head_idx * num_tokens * num_tokens) +
                                               (token_idx_y * num_tokens) +
                                               token_idx_x];
                vec[token_idx_x] = score;

                if (score > max_val) {
                    max_val = score;
                }
            } else {
                vec[token_idx_x] = -FLT_MAX;
            }
        } else {
            vec[token_idx_x] = -FLT_MAX;
        }
        __syncthreads();
    }

    buffer[threadIdx.x] = max_val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            buffer[threadIdx.x] = fmaxf(buffer[threadIdx.x], buffer[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    float global_max = buffer[0];
    __syncthreads();

    float exp_sum = 0.0f;
    for (int i = 0; i < (num_tokens + blockDim.x - 1) / blockDim.x; ++i) {
        token_idx_x = i * blockDim.x + threadIdx.x;

        if (token_idx_x < num_tokens) {
            if (token_idx_x <= token_idx_y) {
                // Subtract the maximum value for numerical stability
                float stabilized_score = vec[token_idx_x] - global_max;
                float exp_score = expf(stabilized_score);
                vec[token_idx_x] = exp_score;
                exp_sum += exp_score;
            } else {
                vec[token_idx_x] = 0.0f;
            }
        } else {
            vec[token_idx_x] = 0.0f;
        }
        __syncthreads();
    }

    buffer[threadIdx.x] = exp_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            buffer[threadIdx.x] += buffer[threadIdx.x + offset];
        }
        __syncthreads();
    }

    float softmax_den = buffer[0];
    __syncthreads();

    for (int i = 0; i < (num_tokens + blockDim.x - 1) / blockDim.x; ++i) {
        token_idx_x = i * blockDim.x + threadIdx.x;

        if (token_idx_x < num_tokens) {
            if (token_idx_x <= token_idx_y) {
                attention_scores[(head_idx * num_tokens * num_tokens) +
                                 (token_idx_y * num_tokens) +
                                 token_idx_x] = vec[token_idx_x] / softmax_den;
            } else {
                attention_scores[(head_idx * num_tokens * num_tokens) +
                                 (token_idx_y * num_tokens) +
                                 token_idx_x] = 0.0f;
            }
        }
        __syncthreads();
    }

    return;
}

__global__ void kernel_compute_resolved_value_from_attention_score_tiled_matmul(
    __half *output, float *attention_scores, __half *V,
    int m, int n, int k, int nheads, int tile_size) {
    /*
        - Each head operates independently of other heads.
        - `m`: Number of tokens (rows of attention scores).
        - `n`: Head dimension
        - `k`: Number of tokens (common dimension).
        - `TILE_SIZE`: Tile size for shared memory.
    */

    extern __shared__ float shared_mem[];
    float *attention_shmem = shared_mem;
    float *V_shmem = shared_mem + tile_size * tile_size;

    int q_head_idx = blockIdx.z;
    int kv_head_idx = q_head_idx / 4;
    int kv_heads = nheads / 4;

    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;

    float value = 0.0f;
    for (int t = 0; t < (k + tile_size - 1) / tile_size; ++t) {
        // Load attention_scores into shared memory
        if (row < m && (t * tile_size + threadIdx.x) < k) {
            int attn_idx = q_head_idx * m * k + row * k + (t * tile_size + threadIdx.x);
            attention_shmem[threadIdx.y * tile_size + threadIdx.x] = attention_scores[attn_idx];
        } else {
            attention_shmem[threadIdx.y * tile_size + threadIdx.x] = 0.0f;
        }

        // Load V into shared memory
        if (col < n && (t * tile_size + threadIdx.y) < k) {
            int V_idx = (t * tile_size * n * kv_heads) + (threadIdx.y * n * kv_heads) + kv_head_idx * n + col;
            V_shmem[threadIdx.y * tile_size + threadIdx.x] = __half2float(V[V_idx]);
        } else {
            V_shmem[threadIdx.y * tile_size + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute partial sums
        for (int i = 0; i < tile_size; ++i) {
            if ((t * tile_size + i) < k) {
                value += attention_shmem[threadIdx.y * tile_size + i] * V_shmem[i * tile_size + threadIdx.x];
            }
        }
        __syncthreads();
    }

    // Write the result to the output tensor
    if (row < m && col < n) {
        int output_idx = row * nheads * n + q_head_idx * n + col;
        output[output_idx] = __float2half(value);
    }
}

/* ********************************* Feed Forward Network ********************************* */
void compute_feedforward(Tensor *X, Llama3Layer *L3_Layer, CudaCache *Cache) {
    // Declare common variables
    size_t shared_mem_size = 3 * TILE_SIZE * TILE_SIZE * sizeof(float);
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid;

    // Gate projection computation
    grid = dim3(
        (L3_Layer->mlp_gate_proj->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE);

    kernel_standard_tiled_gemm_ffn_swiglu<<<grid, block, shared_mem_size>>>(
        Cache->d_feedforward_cache_upgate, X->d_fp16_tensor,
        L3_Layer->mlp_up_proj->d_fp16_tensor, L3_Layer->mlp_gate_proj->d_fp16_tensor,
        h_NUM_TOKENS, L3_Layer->mlp_gate_proj->shape[0], 4096, TILE_SIZE);

    // Final output feedforward output computation
    shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    grid = dim3(
        (L3_Layer->mlp_down_proj->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE);

    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        X->d_fp16_tensor, Cache->d_feedforward_cache_upgate, L3_Layer->mlp_down_proj->d_fp16_tensor,
        h_NUM_TOKENS, L3_Layer->mlp_down_proj->shape[0], L3_Layer->mlp_down_proj->shape[1], TILE_SIZE);

    return;
}

__global__ void kernel_compute_swiglu(
    __half *output, __half *gate, __half *up,
    int embed_dim, int num_tokens) {
    // Kernel start
    //
    int embed_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;

    if (token_idx >= num_tokens) return;
    if (embed_idx >= embed_dim) return;

    int index = token_idx * embed_dim + embed_idx;

    float gate_val = __half2float(gate[index]);
    float up_val = __half2float(up[index]);

    output[index] = __float2half(SiLU(gate_val) * up_val);

    return;
}

/* ********************************* Language Model Head ********************************* */
int compute_lm_head(Tensor *LM_Head, Tensor *X, CudaCache *Cache) {
    // Declare common variables
    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid;

    // Query computation
    grid = dim3(
        (LM_Head->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        1);

    __half *last_tensor = X->d_fp16_tensor + ((h_NUM_TOKENS - 1) * 4096);

    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        Cache->d_token_probdist, last_tensor, LM_Head->d_fp16_tensor,
        1, LM_Head->shape[0], 4096, TILE_SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(
        Cache->h_token_probdist,
        Cache->d_token_probdist,
        VOCAB_SIZE * sizeof(__half),
        cudaMemcpyDeviceToHost);

    return sample_next_token(Cache->f_token_probdist, Cache->h_token_probdist);
}

int sample_next_token(float *tensor, __half *half_tensor) {
    ProbIndex *prob_dist_array = temperature_softmax(tensor, half_tensor, TEMPERATURE);

    int selected_token;
    selected_token = top_k_sampling(prob_dist_array, TOP_K);
    selected_token = top_p_sampling(prob_dist_array, TOP_P);

    free(prob_dist_array);
    return selected_token;
}

ProbIndex *temperature_softmax(float *tensor, __half *half_tensor, float temperature) {
    ProbIndex *prob_dist_array = (ProbIndex *)malloc(VOCAB_SIZE * sizeof(ProbIndex));

    float cumsum = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        float scaled_logit = __half2float(half_tensor[i]) / temperature;

        tensor[i] = exp(scaled_logit);
        cumsum += tensor[i];
    }

    for (int i = 0; i < VOCAB_SIZE; ++i) {
        tensor[i] /= cumsum;

        prob_dist_array[i].probability = tensor[i];
        prob_dist_array[i].index = i;
    }

    qsort(prob_dist_array, VOCAB_SIZE, sizeof(ProbIndex), _compare_desc);
    return prob_dist_array;
}

int _compare_desc(const void *a, const void *b) {
    const ProbIndex *prob_a = (const ProbIndex *)a;
    const ProbIndex *prob_b = (const ProbIndex *)b;

    if (prob_a->probability < prob_b->probability) {
        return 1;
    } else if (prob_a->probability > prob_b->probability) {
        return -1;
    } else {
        return 0;
    }
}

int top_k_sampling(ProbIndex *prob_dist_array, int k) {
    if (k <= 0) {
        // Invalid k, fallback to random sampling
        int random_index = rand() % VOCAB_SIZE;
        return prob_dist_array[random_index].index;
    }

    if (k > VOCAB_SIZE) {
        k = VOCAB_SIZE;
    }

    // Calculate the sum of the top k probabilities
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += prob_dist_array[i].probability;
    }

    // Renormalize the top k probabilities
    for (int i = 0; i < k; ++i) {
        prob_dist_array[i].probability /= sum;
    }

    // Generate a random float between 0 and 1
    float rand_val = (float)rand() / (float)RAND_MAX;

    // Select the token based on the cumulative probability
    float cumulative = 0.0f;
    for (int i = 0; i < k; ++i) {
        cumulative += prob_dist_array[i].probability;
        if (rand_val < cumulative) {
            return prob_dist_array[i].index;
        }
    }

    // Fallback in case of floating point precision issues
    return prob_dist_array[0].index;
}

int top_p_sampling(ProbIndex *prob_dist_array, float p) {
    if (p <= 0.0f) {
        // Invalid p, fallback to random sampling
        int random_index = rand() % VOCAB_SIZE;
        return prob_dist_array[random_index].index;
    }

    if (p > 1.0f) {
        p = 1.0f;
    }

    // Determine the cutoff where cumulative probability exceeds p
    float cumulative = 0.0f;
    int cutoff = 0;
    for (; cutoff < VOCAB_SIZE; ++cutoff) {
        cumulative += prob_dist_array[cutoff].probability;
        if (cumulative >= p) {
            break;
        }
    }

    // Calculate the sum of probabilities up to the cutoff
    float sum = cumulative;
    if (cutoff >= VOCAB_SIZE) {
        cutoff = VOCAB_SIZE - 1;
        sum = cumulative;
    }

    // Renormalize the probabilities up to the cutoff
    for (int i = 0; i <= cutoff; ++i) {
        prob_dist_array[i].probability /= sum;
    }

    // Generate a random float between 0 and 1
    float rand_val = (float)rand() / (float)RAND_MAX;

    // Select the token based on the cumulative probability
    cumulative = 0.0f;
    for (int i = 0; i <= cutoff; ++i) {
        cumulative += prob_dist_array[i].probability;
        if (rand_val < cumulative) {
            return prob_dist_array[i].index;
        }
    }

    // Fallback in case of floating point precision issues
    return prob_dist_array[0].index;
}
