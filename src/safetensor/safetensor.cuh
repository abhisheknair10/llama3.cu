#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON/cJSON.h"
#include "llama3/llama3.cuh"

/**
 * @brief Struct representing a SafeTensor file for loading model weights.
 */
typedef struct {
    FILE *fp;
    char *header;
    uint64_t header_size;
} SafeTensorFile;

/**
 * @brief Loads SafeTensor weights into the Llama3 model from the specified file.
 *
 * @param llama3_model Pointer to the Llama3 model where weights will be loaded.
 * @param filename Path to the SafeTensor file containing the model weights.
 */
void load_safetensor_weights(Llama3 *llama3_model, const char *filename);

/**
 * @brief Loads the header section of the SafeTensor file.
 *
 * This function reads the header section of the SafeTensor file into the SafeTensorFile struct.
 *
 * @param STF Pointer to the SafeTensorFile struct representing the opened SafeTensor file.
 */
void safetensor_load_header(SafeTensorFile *STF);

/**
 * @brief Reads the SafeTensor header and updates the Llama3 model.
 *
 * This function processes the SafeTensor file header and extracts metadata necessary to
 * load the model weights into the Llama3 model.
 *
 * @param STF Pointer to the SafeTensorFile struct.
 * @param llama3_model Pointer to the Llama3 model where weights will be loaded.
 */
void safetensor_read_header(SafeTensorFile *STF, Llama3 *llama3_model);

/**
 * @brief Loads a specific layer's weights into the Llama3 model.
 *
 * This function uses a cJSON object representing layer metadata and loads the corresponding
 * tensor weights from the SafeTensor file.
 *
 * @param json Pointer to a cJSON object containing metadata for the layer.
 * @param STF Pointer to the SafeTensorFile from which weights will be read.
 * @param llama3_model Pointer to the Llama3 model where the layer's weights will be loaded.
 */
void llama3_load_layer(cJSON *json, SafeTensorFile *STF, Llama3 *llama3_model);

/**
 * @brief Retrieves the decoder layer number from the SafeTensor key.
 *
 * This function parses the layer key string to determine the corresponding Llama3 decoder
 * layer number.
 *
 * @param layer_key Pointer to the string representing the layer key in the SafeTensor file.
 * @param index The index of the character where the decoder layer number starts in the string.
 * @return The decoder layer number as an integer.
 */
int get_llama3_decoder_layer_num(char *layer_key, int index);

/**
 * @brief Frees the resources associated with the SafeTensor file handler.
 *
 * This function deallocates the memory used for the SafeTensor file header and closes the file.
 *
 * @param STF Pointer to the SafeTensorFile struct to be freed.
 */
void free_safetensor_handler(SafeTensorFile *STF);
