#include <cuda_fp16.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON/cJSON.h"
#include "llama3/llama3.cuh"
#include "tokenizer.cuh"

#define CHECK_CUDA_ERROR()                                       \
    {                                                            \
        cudaError_t err = cudaGetLastError();                    \
        if (err != cudaSuccess) {                                \
            printf("CUDA error: %s in file '%s' in line %i\n",   \
                   cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    }

Llama3Tokenizer *load_tokenizer() {
    // Allocate memory for the tokenizer
    Llama3Tokenizer *llama3_tokenizer = (Llama3Tokenizer *)malloc(sizeof(Llama3Tokenizer));
    if (llama3_tokenizer == NULL) {
        printf("Error: Memory allocation failed for Llama3 Tokenizer\n");
        exit(1);
    }

    // Initialize the root TrieNode
    llama3_tokenizer->root = init_trienode();

    // Initialize decoder
    llama3_tokenizer->decode = (char **)malloc(sizeof(char *) * 128256);

    // Read the tokenizer JSON file
    char *buffer = read_tokenizer_json(llama3_tokenizer, "model_weights/modified_tokenizer.json");

    // Parse the JSON buffer
    cJSON *json_buffer = cJSON_Parse(buffer);
    if (json_buffer == NULL) {
        printf("Error: while parsing tokenizer.json structure\n");
        free(buffer);
        free(llama3_tokenizer->root);
        free(llama3_tokenizer);
        exit(1);
    }

    // Free the buffer after parsing
    free(buffer);

    cJSON *added_tokens = cJSON_GetObjectItemCaseSensitive(json_buffer, "added_tokens");
    if (added_tokens == NULL) {
        printf("Error: Could not find 'model' in the JSON\n");
        cJSON_Delete(json_buffer);
        free(llama3_tokenizer->root);
        free(llama3_tokenizer);
        exit(1);
    }

    // Get the "model" object from the JSON
    cJSON *model = cJSON_GetObjectItemCaseSensitive(json_buffer, "model");
    if (model == NULL) {
        printf("Error: Could not find 'model' in the JSON\n");
        cJSON_Delete(json_buffer);
        free(llama3_tokenizer->root);
        free(llama3_tokenizer);
        exit(1);
    }

    // Get the "vocab" object from the JSON
    cJSON *vocab = cJSON_GetObjectItemCaseSensitive(model, "vocab");
    if (vocab == NULL || !cJSON_IsObject(vocab)) {
        printf("Error: Could not find 'vocab' in the model or it's not an object\n");
        cJSON_Delete(json_buffer);
        free(llama3_tokenizer->root);
        free(llama3_tokenizer);
        exit(1);
    }

    cJSON *curr_element = NULL;

    // Traverse the vocab and print characters
    cJSON_ArrayForEach(curr_element, vocab) {
        _build_trie(llama3_tokenizer->root, curr_element->string, curr_element->valueint);
        _build_decoder(llama3_tokenizer->decode, curr_element->string, curr_element->valueint);
    }

    cJSON_ArrayForEach(curr_element, added_tokens) {
        cJSON *id = cJSON_GetObjectItemCaseSensitive(curr_element, "id");
        cJSON *content = cJSON_GetObjectItemCaseSensitive(curr_element, "content");
        // printf("%d: %s\n", id->valueint, content->valuestring);

        _build_trie(llama3_tokenizer->root, content->valuestring, id->valueint);
        _build_decoder(llama3_tokenizer->decode, content->valuestring, id->valueint);
    }

    // Free the parsed JSON object
    cJSON_Delete(json_buffer);

    return llama3_tokenizer;
}

int *tokenize(Llama3Tokenizer *tokenizer, char *input_str) {
    // 1st and 2nd index reserved for
    //  - Number of tokens
    //  - array formatter
    const int max_tokens = 2048 + 1;

    TrieNode *curr;
    int input_len = strlen(input_str);

    int *tokens = (int *)malloc(sizeof(int) * max_tokens);
    if (tokens == NULL) {
        printf("Error: Memory allocation failed for tokens\n");
        exit(1);
    }

    int token_count = 0;
    int i = 0;
    while (i < input_len && token_count < 2048) {
        curr = tokenizer->root;
        int last_token = -1;
        int last_token_len = 0;

        for (int j = i; j < input_len; j++) {
            TrieNode *next_node;
            if (_find_child(curr, input_str[j], &next_node) == -1) {
                break;
            }

            curr = next_node;
            if (curr->token != -1) {
                last_token = curr->token;
                last_token_len = j - i + 1;
            }
        }

        if (last_token == -1) {
            printf("Error: Unable to tokenize input at position %d\n", i);
            exit(1);
        }

        tokens[++token_count] = last_token;
        i += last_token_len;
    }

    if (tokens == NULL) {
        printf("Error: Memory reallocation failed for tokens\n");
        exit(1);
    }

    tokens[0] = token_count + 1;

    return tokens;
}

int *tokens_to_cuda(int *tokens, int embed_size, Tensor *token_tensor) {
    /* *************** Token Tensor Init *************** */
    // Set metadata in CPU
    token_tensor->ndim = (int *)malloc(sizeof(int));
    *(token_tensor->ndim) = 2;

    token_tensor->mem_len = (int *)malloc(sizeof(int));
    *(token_tensor->mem_len) = embed_size * 2048;

    token_tensor->shape = (int *)malloc(sizeof(int) * (*(token_tensor->ndim)));
    token_tensor->shape[0] = 2048;
    token_tensor->shape[1] = embed_size;

    /* *************** Token Tensor Init CUDA *************** */
    // SPECIAL: fp16 tensor being allocated but only used later
    _cudaMalloc_fp16(token_tensor);
    cudaMemset(token_tensor->d_fp16_tensor, __float2half(0.0f), sizeof(__half) * (*(token_tensor->mem_len)));

    int *d_ndim;
    int *d_mem_len;
    int *d_shape;

    cudaMalloc((void **)&d_ndim, sizeof(int));
    cudaMalloc((void **)&d_mem_len, sizeof(int));
    cudaMalloc((void **)&d_shape, sizeof(int) * (*(token_tensor->ndim)));

    cudaMemcpy(d_ndim, token_tensor->ndim, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mem_len, token_tensor->mem_len, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, token_tensor->shape, sizeof(int) * (*(token_tensor->ndim)), cudaMemcpyHostToDevice);

    // Set pointers to CUDA pointers
    token_tensor->d_ndim = d_ndim;
    token_tensor->d_mem_len = d_mem_len;
    token_tensor->d_shape = d_shape;

    /* *************** Move Actual Tensor to CUDA *************** */
    // Copy over tokens
    int *d_tokens;
    cudaMalloc((void **)&d_tokens, sizeof(int) * token_tensor->shape[0]);

    return d_tokens;
}

char *read_tokenizer_json(Llama3Tokenizer *tokenizer, const char *filename) {
    long fileSize = _get_file_size(filename);
    if (fileSize == -1) {
        printf("Error: Could not determine file size\n");
        exit(1);
    }

    // Allocate buffer to hold the file contents
    char *buffer = (char *)malloc(sizeof(char) * (fileSize + 1));
    if (buffer == NULL) {
        printf("Error: Memory allocation failed for buffer\n");
        exit(1);
    }

    // Open the file and read its content
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Error: Could not open file %s\n", filename);
        free(buffer);
        exit(1);
    }

    fread(buffer, sizeof(char), fileSize, fp);
    buffer[fileSize] = '\0';  // Null-terminate the buffer
    fclose(fp);

    return buffer;
}

long _get_file_size(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Failed to open file");
        return -1;
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fclose(file);

    return fileSize;
}

TrieNode *init_trienode() {
    TrieNode *node = (TrieNode *)malloc(sizeof(TrieNode));
    if (node == NULL) {
        printf("Error: Memory allocation failed for TrieNode\n");
        return NULL;
    }

    node->_max_childs = 1;
    node->child_len = 0;
    node->token = -1;

    node->child_char = (char *)malloc(sizeof(char) * node->_max_childs);
    if (node->child_char == NULL) {
        printf("Error: Memory allocation failed for child_char array\n");
        free(node);
        return NULL;
    }

    node->child_node = (TrieNode **)malloc(sizeof(TrieNode *) * node->_max_childs);
    if (node->child_node == NULL) {
        printf("Error: Memory allocation failed for child_node array\n");
        free(node->child_char);
        free(node);
        return NULL;
    }

    return node;
}

void _build_trie(TrieNode *root, char *token_string, int token_int) {
    TrieNode *curr = root;

    // Traverse each character in the token string
    for (char *ch = token_string; *ch != '\0'; ch++) {
        TrieNode *next_node;

        if (_find_child(curr, *ch, &next_node) == -1) {
            if (curr->child_len == curr->_max_childs) {
                _memexpand_child_nodes(curr);
            }

            curr->child_char[curr->child_len] = *ch;
            curr->child_node[curr->child_len] = init_trienode();
            curr->child_len += 1;

            curr = curr->child_node[curr->child_len - 1];
        } else {
            curr = next_node;
        }
    }
    curr->token = token_int;

    return;
}

int _find_child(TrieNode *node, char str, TrieNode **result) {
    for (int i = 0; i < node->child_len; i++) {
        if (node->child_char[i] == str) {
            *result = node->child_node[i];
            return i;
        }
    }
    return -1;
}

void _memexpand_child_nodes(TrieNode *node) {
    // Expand to twice memory size
    node->_max_childs *= 2;

    /*
        Realloc char childs and trie node childs
    */

    char *new_child_char = (char *)realloc(node->child_char, sizeof(char) * node->_max_childs);
    if (new_child_char == NULL) {
        printf("Error: Memory allocation failed during child_char expansion\n");
        exit(1);
    }

    node->child_char = new_child_char;

    TrieNode **new_child_node = (TrieNode **)realloc(node->child_node, sizeof(TrieNode *) * node->_max_childs);
    if (new_child_node == NULL) {
        printf("Error: Memory allocation failed during child_node expansion\n");
        exit(1);
    }
    node->child_node = new_child_node;

    return;
}

void _build_decoder(char **decoder, char *token_string, int token_int) {
    size_t length = strlen(token_string) + 1;
    decoder[token_int] = (char *)malloc(length);
    if (decoder[token_int] == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for decoder[%d].\n", token_int);
        exit(EXIT_FAILURE);
    }

    // Copy the token string into the decoder array
    strcpy(decoder[token_int], token_string);
}