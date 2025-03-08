#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON/cJSON.h"
#include "llama3/llama3.cuh"

/**
 * @brief Structure representing a node in a Trie data structure.
 *
 * The TrieNode structure is used to store a single character, along with a list of child nodes. Each node can
 * have multiple children, allowing for efficient tokenization by character traversal.
 */
typedef struct TrieNode {
    int _max_childs;
    int child_len;
    char *child_char;
    struct TrieNode **child_node;
    int token;
} TrieNode;

/**
 * @brief Structure representing the LLaMA 3 tokenizer.
 *
 * This structure holds a pointer to the root TrieNode, which represents the starting point for tokenization.
 */
typedef struct {
    TrieNode *root;
    char **decode;
} Llama3Tokenizer;

/**
 * @brief Loads the tokenizer by initializing the root TrieNode and reading the tokenization rules from a file.
 *
 * @return Llama3Tokenizer* Pointer to the initialized tokenizer.
 */
Llama3Tokenizer *load_tokenizer();

/**
 * @brief Tokenizes a given input string using the provided tokenizer.
 *
 * This function traverses the Trie structure to tokenize the input string, returning an array of integer tokens.
 *
 * @param tokenizer Pointer to the Llama3Tokenizer used for tokenization.
 * @param input_str Input string to be tokenized.
 * @return int* Array of integer tokens corresponding to the input string.
 */
int *tokenize(Llama3Tokenizer *tokenizer, char *input_str);

/**
 * @brief Transfers token data to CUDA memory and initializes the corresponding tensor.
 *
 * This function takes an array of integer tokens, initializes the tensor into which it will be stored,
 * and transfers the tensor data and metadata to CUDA device memory for processing.
 *
 * @param tokens Pointer to the array of integer tokens. The first element of the array
 *               should contain the number of tokens.
 * @param embed_size The embedding size of the model, used for setting tensor dimensions.
 * @param token_tensor Pointer to a Tensor structure, which will hold metadata and
 *                     CUDA device memory references after the transfer.
 *
 * @return int* Pointer to the array of tokens stored in CUDA device memory.
 */
int *tokens_to_cuda(int *tokens, int embed_size, Tensor *token_tensor);

/**
 * @brief Reads the tokenizer rules from a JSON file and constructs the Trie structure.
 *
 * The function parses a JSON file to obtain tokenization rules, such as token strings and their associated integer values.
 * It then builds the Trie structure to reflect these rules.
 *
 * @param tokenizer Pointer to the Llama3Tokenizer to populate.
 * @param filename Path to the JSON file containing the tokenization rules.
 * @return char* Returns the raw content of the JSON file.
 */
char *read_tokenizer_json(Llama3Tokenizer *tokenizer, const char *filename);

/**
 * @brief Gets the file size of the specified file.
 *
 * This function opens the file in read mode and calculates its size in bytes.
 *
 * @param filename Path to the file.
 * @return long The size of the file in bytes.
 */
long _get_file_size(const char *filename);

/**
 * @brief Initializes and returns a new TrieNode.
 *
 * This function allocates memory for a TrieNode and sets the initial values, such as no children and no token.
 *
 * @return TrieNode* Pointer to the newly initialized TrieNode.
 */
TrieNode *init_trienode();

/**
 * @brief Builds the Trie structure by adding a token string and its associated integer value.
 *
 * The function takes a root TrieNode and a token string and inserts the string into the Trie.
 * Each character of the token string is added as a child node, and the final character is assigned the token value.
 *
 * @param node Pointer to the root TrieNode.
 * @param token_string The token string to be inserted into the Trie.
 * @param token_int The integer token value associated with the string.
 */
void _build_trie(TrieNode *node, char *token_string, int token_int);

/**
 * @brief Finds the child node corresponding to a given character.
 *
 * This function searches through the children of a TrieNode to find a child node that matches the given character.
 *
 * @param node Pointer to the TrieNode to search within.
 * @param str Character to find among the children.
 * @param result Pointer to a TrieNode pointer, which will be set to the found child node.
 * @return int 1 if the child was found, 0 otherwise.
 */
int _find_child(TrieNode *node, char str, TrieNode **result);

/**
 * @brief Expands the capacity for child nodes in a TrieNode.
 *
 * If the child capacity of a TrieNode has been reached, this function dynamically increases the capacity
 * to allow for more children to be added.
 *
 * @param node Pointer to the TrieNode whose child capacity will be expanded.
 */
void _memexpand_child_nodes(TrieNode *node);

/**
 * @brief Adds a token and its corresponding string to the decoder array.
 *
 * This function assigns the provided `token_string` to the `decoder` array at the index specified by `token_int`.
 * It is used to build the decoder mapping, which translates integer tokens back to their original string representations.
 * This is essential for decoding tokenized data back into human-readable text.
 *
 * @param decoder Double pointer to the decoder array where the token string will be stored.
 *                The array should be pre-allocated with sufficient space to accommodate the token indices.
 * @param token_string The string representation of the token to be added to the decoder.
 *                     This string should correspond to the token's textual form used during tokenization.
 * @param token_int The integer token value that corresponds to the `token_string`.
 *                  This value serves as the index in the `decoder` array where the `token_string` will be stored.
 *
 * @note It is assumed that `token_int` is within the valid range of the `decoder` array indices.
 *       Ensure that the `decoder` array has been appropriately initialized and allocated before calling this function.
 */
void _build_decoder(char **decoder, char *token_string, int token_int);
