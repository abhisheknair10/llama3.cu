#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama3/llama3.cuh"

/**
 * @brief Structure representing CUDA cache for the LLaMA 3 model.
 *
 * The `CudaCache` structure holds various tensors and device memory allocations required
 * for performing efficient inference with the LLaMA 3 model. This includes memory for
 * attention scores, feedforward network components, and storage for the next token to be generated.
 */
typedef struct {
    Tensor *PN_X;

    Tensor *Q;
    Tensor *K;
    Tensor *V;

    float *d_attention_score_cache;

    __half *d_feedforward_cache_upgate;

    __half *d_token_probdist;
    __half *h_token_probdist;

    float *f_token_probdist;
} CudaCache;

/**
 * @brief Structure representing a 4-component half-precision vector.
 *
 * The `c_half4` structure is used to store four half-precision floating-point values,
 * typically representing a vector in computations such as transformations or embeddings.
 */
typedef struct {
    __half x, y, z, w;
} c_half4;

typedef struct {
    float probability;
    int index;
} ProbIndex;

/* ********************************* Inference Code ********************************* */

/**
 * @brief Performs inference using the LLaMA 3 model.
 *
 * This function executes the inference pipeline for the LLaMA 3 model, processing the input tokens
 * and updating the model's state using the provided CUDA cache. It leverages GPU acceleration
 * to handle complex computations efficiently.
 *
 * @param llama3_model Pointer to the `Llama3` model instance to be used for inference.
 * @param X Pointer to the input tensor `X` containing the initial embeddings.
 * @param d_tokens Pointer to the device memory array of input tokens.
 * @param h_tokens Pointer to the host memory array of input tokens.
 * @param Cache Pointer to the `CudaCache` structure holding necessary CUDA resources.
 *
 * @return int Status code indicating the success (0) or failure (non-zero) of the inference operation.
 *
 * @note Ensure that all input pointers are properly initialized and that the CUDA cache is set up
 *       before calling this function.
 */
int inference(Llama3 *llama3_model, Tensor *X, int *d_tokens, int *h_tokens, CudaCache *Cache);

/* ************************** Convert Tokens to Embeddings ************************** */

/**
 * @brief Converts token indices to their corresponding embeddings.
 *
 * This function maps an array of token indices to their embedding vectors using the LLaMA 3 model's
 * embedding layer. The resulting embeddings are stored in the input tensor `X`.
 *
 * @param X Pointer to the tensor where the resulting embeddings will be stored.
 * @param llama3_model Pointer to the `Llama3` model instance containing the embedding layer.
 * @param d_tokens Pointer to the device memory array of token indices to be converted.
 *
 * @note The tensor `X` must be pre-allocated with appropriate dimensions to hold the embeddings.
 */
void tokens_to_embeddings(Tensor *X, Llama3 *llama3_model, int *d_tokens);

/**
 * @brief CUDA kernel to convert tokens to embeddings.
 *
 * This CUDA kernel maps each token index to its corresponding embedding vector and stores the result
 * in the output tensor `X`. It is designed to run in parallel on the GPU for efficient computation.
 *
 * @param X Pointer to the device memory where the embeddings will be stored.
 * @param tokens Pointer to the device memory array of token indices.
 * @param Embed Pointer to the device memory containing the embedding matrix.
 * @param num_tokens The number of tokens to be converted.
 *
 * @note Ensure that the embedding matrix `Embed` is properly initialized and that `X` has sufficient
 *       memory allocated to store all embeddings.
 */
__global__ void kernel_tokens_to_embeddings(__half *X, int *tokens, __half *Embed, int num_tokens);

/* ******************************* Layer Normalization ******************************* */

/**
 * @brief Creates a copy of the intermediary prenormalization tensor.
 *
 * This function allocates and initializes a new tensor that serves as an intermediary copy for
 * prenormalization operations. It is used to preserve the original tensor state before normalization.
 *
 * @return Tensor* Pointer to the newly created intermediary prenormalization tensor.
 *                 Returns `NULL` if memory allocation fails.
 *
 * @note The caller is responsible for freeing the allocated tensor to prevent memory leaks.
 */
Tensor *_create_intermediary_prenorm_tensor_copy();

/**
 * @brief Copies a half-precision tensor from device to device memory.
 *
 * This function performs a memory copy of a half-precision (`__half`) tensor from source tensor `X`
 * to destination tensor `Y` on the CUDA device. It is used to duplicate tensor data efficiently.
 *
 * @param Y Pointer to the destination tensor where data will be copied.
 * @param X Pointer to the source tensor from which data will be copied.
 *
 * @note Both tensors must be allocated on the device and have compatible dimensions.
 */
void _deviceMemcpy_fp16_tensor(Tensor *Y, Tensor *X);

/**
 * @brief Computes the layer normalization on the input tensor.
 *
 * This function applies layer normalization to the input tensor `X` using the root mean square
 * normalization tensor `RMSNorm`. It normalizes the data to stabilize and accelerate the training
 * process.
 *
 * @param RMSNorm Pointer to the tensor containing normalization parameters.
 * @param X Pointer to the input tensor to be normalized.
 *
 * @note Ensure that both `RMSNorm` and `X` are properly initialized and allocated.
 */
void compute_layer_norm(Tensor *RMSNorm, Tensor *X);

/**
 * @brief CUDA kernel to compute RMS (Root Mean Square) normalization.
 *
 * This CUDA kernel calculates the RMS normalization for each token in the input tensor `X` and
 * stores the result in `RMSNorm`. It operates in parallel on the GPU for efficient computation.
 *
 * @param X Pointer to the device memory containing the input tensor.
 * @param RMSNorm Pointer to the device memory where the RMS normalization results will be stored.
 * @param num_tokens The number of tokens to process for normalization.
 *
 * @note Ensure that the input tensor `X` and output tensor `RMSNorm` are properly allocated on the device.
 */
__global__ void kernel_compute_rms_norm(__half *X, __half *RMSNorm, int num_tokens);

/**
 * @brief Adds normalized data to the primary tensor.
 *
 * This function adds the normalized tensor `PN_X` to the input tensor `X`. It integrates the
 * normalized data into the main tensor for further processing in the model.
 *
 * @param X Pointer to the input tensor.
 * @param PN_X Pointer to the normalized tensor to be added.
 *
 * @note Both tensors must have compatible dimensions.
 */
void add_norm(Tensor *X, Tensor *PN_X);

/**
 * @brief CUDA kernel to add normalized data to the primary tensor.
 *
 * This CUDA kernel performs element-wise addition of the normalized tensor `PN_X` to the input
 * tensor `X`. It is designed to run in parallel on the GPU for efficient computation.
 *
 * @param X Pointer to the device memory of the input tensor.
 * @param PN_X Pointer to the device memory of the normalized tensor to be added.
 * @param num_tokens The number of tokens (elements) to process.
 *
 * @note Ensure that both tensors `X` and `PN_X` are allocated on the device and have sufficient memory.
 */
__global__ void kernel_add_norm(__half *X, __half *PN_X, int num_tokens);

/* ***************************** General Matrix Multiplication **************************** */

/**
 * @brief CUDA kernel for standard tiled general matrix multiplication (GEMM).
 *
 * This kernel performs matrix multiplication using a tiled approach to optimize memory access patterns
 * and improve computational efficiency on the GPU. It multiplies matrices `X` and `Transform`, storing
 * the result in matrix `O`.
 *
 * @param O Pointer to the device memory where the output matrix will be stored.
 * @param X Pointer to the device memory of the first input matrix.
 * @param Transform Pointer to the device memory of the second input matrix.
 * @param m The number of rows in matrix `X`.
 * @param n The number of columns in matrix `Transform`.
 * @param k The number of columns in matrix `X` and rows in matrix `Transform`.
 * @param TILE_SIZE The size of the tile used for the tiled multiplication.
 *
 * @note Ensure that all matrices are properly allocated on the device and that their dimensions are compatible.
 */
__global__ void kernel_standard_tiled_gemm(
    __half *O, __half *X, __half *Transform, int m, int n, int k, int TILE_SIZE);

/* ***************************** Attention Tensor Computation **************************** */

/**
 * @brief Creates an intermediary attention tensor based on the provided linear transformation tensor.
 *
 * This function allocates and initializes a new tensor that serves as an intermediary in the attention
 * computation process. It is derived from the provided linear transformation tensor.
 *
 * @param Linear Pointer to the linear transformation tensor used to create the attention tensor.
 * @return Tensor* Pointer to the newly created intermediary attention tensor.
 *                 Returns `NULL` if memory allocation fails.
 *
 * @note The caller is responsible for managing the lifecycle of the returned tensor to prevent memory leaks.
 */
Tensor *_create_intermediary_attention_tensor(Tensor *Linear);

/**
 * @brief Computes the Query (Q), Key (K), and Value (V) tensors for the attention mechanism.
 *
 * This function calculates the Q, K, and V tensors by applying the respective linear transformations
 * defined in the `Llama3Layer` to the input tensor `X`. These tensors are essential for the attention
 * computation in transformer architectures.
 *
 * @param Q Pointer to the tensor where the Query vectors will be stored.
 * @param K Pointer to the tensor where the Key vectors will be stored.
 * @param V Pointer to the tensor where the Value vectors will be stored.
 * @param L3_Layer Pointer to the `Llama3Layer` containing the linear transformation parameters.
 * @param X Pointer to the input tensor to be transformed.
 *
 * @note Ensure that `Q`, `K`, and `V` tensors are properly allocated with dimensions matching the model's requirements.
 */
void compute_qkv_tensors(
    Tensor *Q, Tensor *K, Tensor *V,
    Llama3Layer *L3_Layer, Tensor *X);

/**
 * @brief Computes the output tensor from the attention mechanism.
 *
 * This function processes the attention outputs using the provided `Llama3Layer` and updates the
 * main tensor `X` accordingly. It leverages the CUDA cache for optimized performance.
 *
 * @param L3_Layer Pointer to the `Llama3Layer` containing necessary transformation parameters.
 * @param X Pointer to the main tensor to be updated with the attention output.
 * @param Cache Pointer to the `CudaCache` structure holding CUDA resources.
 *
 * @note Ensure that all input pointers are properly initialized and that the CUDA cache is set up.
 */
void compute_output(Llama3Layer *L3_Layer, Tensor *X, CudaCache *Cache);

/* ************************* Rotary Positional Embedding (RoPE) ************************* */

/**
 * @brief Applies Rotary Positional Embedding (RoPE) scaling to the Query and Key tensors.
 *
 * This function modifies the Query (Q) and Key (K) tensors by applying RoPE scaling, which incorporates
 * positional information into the embeddings. This technique enhances the model's ability to capture
 * positional relationships within the input sequence.
 *
 * @param Q Pointer to the Query tensor to be scaled.
 * @param K Pointer to the Key tensor to be scaled.
 *
 * @note Ensure that both `Q` and `K` tensors are properly allocated and initialized before calling this function.
 */
void rope_scaling(Tensor *Q, Tensor *K);

/**
 * @brief CUDA kernel to apply RoPE scaling to a tensor.
 *
 * This CUDA kernel applies Rotary Positional Embedding (RoPE) scaling to the input tensor by modifying
 * its values based on the transformed embedding size and the number of tokens. It operates in parallel
 * on the GPU for efficient computation.
 *
 * @param tensor Pointer to the device memory of the tensor to be scaled.
 * @param transformed_embed_size The size of the transformed embeddings.
 * @param num_tokens The number of tokens in the tensor.
 *
 * @note Ensure that the input tensor is properly allocated on the device and that `transformed_embed_size`
 *       and `num_tokens` are set correctly.
 */
__global__ void kernel_rope_scaling(__half *tensor, int transformed_embed_size, int num_tokens);

/* **************************** Grouped Multi-Query Attention **************************** */

/**
 * @brief Computes the attention mechanism using Query, Key, and Value tensors.
 *
 * This function performs the attention computation by processing the Query (Q), Key (K), and Value (V)
 * tensors. It utilizes the CUDA cache for optimized performance and supports grouped multi-query attention.
 *
 * @param X Pointer to the main tensor being processed.
 * @param Q Pointer to the Query tensor.
 * @param K Pointer to the Key tensor.
 * @param V Pointer to the Value tensor.
 * @param Cache Pointer to the `CudaCache` structure holding CUDA resources.
 *
 * @note Ensure that all input tensors are properly allocated and initialized before calling this function.
 */
void compute_attention(Tensor *X, Tensor *Q, Tensor *K, Tensor *V, CudaCache *Cache);

/**
 * @brief CUDA kernel to compute masked grouped multi-query attention scores using tiled matrix multiplication.
 *
 * This kernel calculates the attention scores by performing tiled matrix multiplication on the Query and Key tensors.
 * It applies masking to ensure that only valid attention scores are considered. The results are stored in `attention_scores`.
 *
 * @param attention_scores Pointer to the device memory where attention scores will be stored.
 * @param Q Pointer to the Query tensor in device memory.
 * @param K Pointer to the Key tensor in device memory.
 * @param m The number of rows in matrix `Q`.
 * @param n The number of columns in matrix `K`.
 * @param k The shared dimension size between `Q` and `K`.
 * @param TILE_SIZE The size of the tile used for the tiled matrix multiplication.
 * @param nheads The number of attention heads.
 *
 * @note Ensure that all input tensors are properly allocated on the device and that their dimensions are compatible.
 */
__global__ void kernel_compute_masked_gmq_attention_scores_tiled_matmul(
    float *attention_scores, __half *Q, __half *K,
    int m, int n, int k, int TILE_SIZE, int nheads);

/**
 * @brief CUDA kernel to apply masking and softmax to attention scores.
 *
 * This kernel applies masking to the attention scores to prevent the model from attending to certain positions.
 * It then applies the softmax function to normalize the scores, converting them into probabilities.
 *
 * @param attention_scores Pointer to the device memory containing attention scores to be masked and normalized.
 * @param num_tokens The number of tokens for which attention scores are computed.
 *
 * @note Ensure that `attention_scores` is properly allocated on the device and contains valid score values before calling this kernel.
 */
__global__ void kernel_masking_softmax(float *attention_scores, int num_tokens);

/**
 * @brief CUDA kernel to compute the resolved value from attention scores using tiled matrix multiplication.
 *
 * This kernel multiplies the attention scores with the Value (V) tensor to compute the final output of the attention
 * mechanism. It utilizes a tiled matrix multiplication approach for efficient computation on the GPU.
 *
 * @param output Pointer to the device memory where the attention output will be stored.
 * @param attention_scores Pointer to the device memory containing the normalized attention scores.
 * @param V Pointer to the Value tensor in device memory.
 * @param m The number of rows in the attention scores matrix.
 * @param n The number of columns in the Value matrix.
 * @param k The shared dimension size between attention scores and `V`.
 * @param nheads The number of attention heads.
 * @param TILE_SIZE The size of the tile used for the tiled matrix multiplication.
 *
 * @note Ensure that all input tensors are properly allocated on the device and that their dimensions are compatible.
 */
__global__ void kernel_compute_resolved_value_from_attention_score_tiled_matmul(
    __half *output, float *attention_scores, __half *V,
    int m, int n, int k, int nheads, int TILE_SIZE);

/* ********************************* Feed Forward Network ********************************* */

/**
 * @brief Computes the feedforward network within a transformer layer.
 *
 * This function applies the feedforward network to the input tensor `X` using the parameters from the
 * specified `Llama3Layer`. It updates the tensor `X` with the results, leveraging the CUDA cache for optimized performance.
 *
 * @param X Pointer to the input tensor to be processed by the feedforward network.
 * @param L3_Layer Pointer to the `Llama3Layer` containing the feedforward network parameters.
 * @param Cache Pointer to the `CudaCache` structure holding CUDA resources.
 *
 * @return void
 *
 * @note Ensure that `X` and `L3_Layer` are properly initialized and that the CUDA cache is set up before calling this function.
 */
void compute_feedforward(Tensor *X, Llama3Layer *L3_Layer, CudaCache *Cache);

/**
 * @brief CUDA kernel to compute the SwiGLU activation function.
 *
 * This kernel applies the SwiGLU activation function to the gate and up tensors, producing the output tensor.
 * SwiGLU is a variant of the Gated Linear Unit (GLU) that incorporates a Swish activation function for improved performance.
 *
 * @param output Pointer to the device memory where the activation output will be stored.
 * @param gate Pointer to the device memory of the gate tensor.
 * @param up Pointer to the device memory of the up tensor.
 * @param embed_dim The embedding dimension size.
 * @param num_tokens The number of tokens to process.
 *
 * @note Ensure that all input tensors are properly allocated on the device and that their dimensions are compatible.
 */
__global__ void kernel_compute_swiglu(
    __half *output, __half *gate, __half *up,
    int embed_dim, int num_tokens);

/* ********************************* Language Model Head ********************************* */

/**
 * @brief Computes the language model head to generate output logits.
 *
 * This function applies the language model head to the input tensor `X`, producing the final output logits
 * used for predicting the next token in the sequence. It utilizes the CUDA cache for optimized performance.
 *
 * @param LM_Head Pointer to the tensor where the language model head output will be stored.
 * @param X Pointer to the input tensor containing the processed embeddings.
 * @param Cache Pointer to the `CudaCache` structure holding CUDA resources.
 *
 * @return int Status code indicating the success (0) or failure (non-zero) of the language model head computation.
 *
 * @note Ensure that `X` and `LM_Head` are properly allocated and initialized before calling this function.
 */
int compute_lm_head(Tensor *LM_Head, Tensor *X, CudaCache *Cache);

int sample_next_token(float *tensor, __half *half_tensor);

ProbIndex *temperature_softmax(float *tensor, __half *half_tensor, float temperature);

int _compare_desc(const void *a, const void *b);

int top_k_sampling(ProbIndex *prob_dist_array, int k);

int top_p_sampling(ProbIndex *prob_dist_array, float p);

/* ************************************** Cuda Cache ************************************** */

/**
 * @brief Initializes the CUDA cache for the specified LLaMA 3 model.
 *
 * This function allocates and sets up a `CudaCache` structure, which holds various tensors and
 * device memory allocations necessary for performing efficient inference with the LLaMA 3 model.
 * The cache includes memory for attention scores, feedforward network components, and storage
 * for the next token to be generated. Proper initialization of this cache is crucial for
 * optimizing GPU memory usage and ensuring smooth execution of the model's inference pipeline.
 *
 * @param llama3_model Pointer to the `Llama3` model instance for which the CUDA cache is being initialized.
 *
 * @return CudaCache* Pointer to the newly initialized `CudaCache` structure.
 *                    Returns `NULL` if the initialization fails due to memory allocation errors or other issues.
 *
 * @note Ensure that the `llama3_model` is properly initialized before calling this function.
 *       After usage, remember to free the allocated `CudaCache` resources to prevent memory leaks.
 */
CudaCache *init_cache(Llama3 *llama3_model);
