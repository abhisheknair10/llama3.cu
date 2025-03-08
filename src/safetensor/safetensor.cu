#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON/cJSON.h"
#include "llama3/llama3.cuh"
#include "safetensor.cuh"

#define WARN "\033[1;33m"
#define GREY "\033[2m"
#define RESET "\033[0m"

// Load SafeTensor entry function
void load_safetensor_weights(Llama3 *llama3_model, const char *filename) {
    SafeTensorFile STF;
    STF.fp = fopen(filename, "rb");

    if (STF.fp == NULL) {
        printf("Error: An Error Occurred while opening %s\n", filename);
        exit(1);
    }

    safetensor_load_header(&STF);
    safetensor_read_header(&STF, llama3_model);

    free_safetensor_handler(&STF);

    return;
}

void safetensor_load_header(SafeTensorFile *STF) {
    // read into header size
    uint64_t header_size;
    fread(&header_size, sizeof(uint64_t), 1, STF->fp);

    STF->header_size = header_size;

    STF->header = (char *)malloc(STF->header_size);
    if (STF->header == NULL) {
        printf("Error: An Error Occurred while allocating memory for the safetensor file header data\n");
        exit(1);
    }
    
    // Read into header buffer
    fread(STF->header, 1, STF->header_size, STF->fp);

    return;
}

void safetensor_read_header(SafeTensorFile *STF, Llama3 *llama3_model) {
    cJSON *json = cJSON_ParseWithLength(STF->header, STF->header_size);
    // printf("%s\n", STF->header);

    if (json == NULL) {
        printf("Error: Error parsing JSON\n");
        return;
    }

    cJSON *current_element = NULL;
    
    // Load tensor for each layer and component
    cJSON_ArrayForEach(current_element, json) {
        if (cJSON_IsObject(current_element)) {
            cJSON *dtype = cJSON_GetObjectItemCaseSensitive(current_element, "dtype");
            cJSON *shape = cJSON_GetObjectItemCaseSensitive(current_element, "shape");
            cJSON *data_offsets = cJSON_GetObjectItemCaseSensitive(current_element, "data_offsets");

            // Check if all three keys exist
            if (dtype && shape && data_offsets) {
                llama3_load_layer(current_element, STF, llama3_model);
                printf(WARN "[CPU]" RESET GREY " Loaded: %s\n" RESET, current_element->string);
            }
        }
    }

    cJSON_Delete(json);
    return;
}

void llama3_load_layer(cJSON *curr_element, SafeTensorFile *STF, Llama3 *llama3_model) {
    int layer_num = get_llama3_decoder_layer_num(curr_element->string, 2);

    // Extract keys from JSON object
    cJSON *dtype = cJSON_GetObjectItemCaseSensitive(curr_element, "dtype");
    cJSON *shape = cJSON_GetObjectItemCaseSensitive(curr_element, "shape");
    cJSON *data_offsets = cJSON_GetObjectItemCaseSensitive(curr_element, "data_offsets");

    int ndim = 0;
    cJSON *shape_curr = shape->child;
    while (shape_curr) {
        ndim += 1;
        shape_curr = shape_curr->next;
    }

    // Assign component being operated
    Tensor *component = NULL;
    if (strstr(curr_element->string, "input_layernorm")) {
        component = llama3_model->layers[layer_num]->input_layernorm;
    } else if (strstr(curr_element->string, "down_proj")) {
        component = llama3_model->layers[layer_num]->mlp_down_proj;
    } else if (strstr(curr_element->string, "gate_proj")) {
        component = llama3_model->layers[layer_num]->mlp_gate_proj;
    } else if (strstr(curr_element->string, "up_proj")) {
        component = llama3_model->layers[layer_num]->mlp_up_proj;
    } else if (strstr(curr_element->string, "post_attention_layernorm")) {
        component = llama3_model->layers[layer_num]->post_attention_layernorm;
    } else if (strstr(curr_element->string, "k_proj")) {
        component = llama3_model->layers[layer_num]->self_attn_k_proj;
    } else if (strstr(curr_element->string, "o_proj")) {
        component = llama3_model->layers[layer_num]->self_attn_o_proj;
    } else if (strstr(curr_element->string, "q_proj")) {
        component = llama3_model->layers[layer_num]->self_attn_q_proj;
    } else if (strstr(curr_element->string, "v_proj")) {
        component = llama3_model->layers[layer_num]->self_attn_v_proj;
    } else if (strstr(curr_element->string, "embed_tokens")) {
        component = llama3_model->embed_tokens;
    } else if (strstr(curr_element->string, "model.norm")) {
        component = llama3_model->norm;
    } else if (strstr(curr_element->string, "lm_head")) {
        component = llama3_model->lm_head;
    }

    if (component == NULL) {
        printf("Error: Component not allocated or not initialized properly.\n");
        exit(1);
    }

    // Allocate(?)/assign metadata and data
    component->ndim = (int *)malloc(sizeof(int));
    *(component->ndim) = ndim;
    component->shape = (int *)malloc(sizeof(int) * ndim);
    if (component->shape == NULL) {
        printf("Error: An Error Occurred while allocating memory for the component shape\n");
        exit(1);
    }

    shape_curr = shape->child;
    int mem_len = 1;
    for (int i = 0; i < ndim; i++) {
        component->shape[i] = shape_curr->valueint;
        mem_len *= (int)shape_curr->valueint;
        shape_curr = shape_curr->next;
    }

    component->mem_len = (int *)malloc(sizeof(int));
    *(component->mem_len) = mem_len;
    component->bf16_tensor = (uint16_t *)malloc(sizeof(uint16_t) * mem_len);

    if (component->bf16_tensor == NULL) {
        printf("Error: An Error Occurred while allocating memory for the component Tensor\n");
        exit(1);
    }

    long offset = (long)8 + (long)STF->header_size + ((long)(data_offsets->child->valuedouble));

    fseek(STF->fp, offset, SEEK_SET);
    size_t num = fread(component->bf16_tensor, sizeof(uint16_t), mem_len, STF->fp);

    return;
}

int get_llama3_decoder_layer_num(char *layer_key, int index) {
    const char *delimiter = ".";
    char *token;

    // Duplicate the layer_key to avoid modifying the original string
    char *layer_key_copy = strdup(layer_key);
    if (layer_key_copy == NULL) {
        printf("Error: An Error Occurred while duplicating layer_key\n");
        exit(1);
    }

    token = strtok(layer_key_copy, delimiter);

    int i = 0;
    while (token != NULL) {
        if (i == index) {
            int layer_num = atoi(token);
            free(layer_key_copy);
            return layer_num;
        }

        token = strtok(NULL, delimiter);
        i++;
    }

    free(layer_key_copy);
    return -1;
}

void free_safetensor_handler(SafeTensorFile *STF) {
    if (STF == NULL) {
        return;
    }

    if (STF->fp != NULL) {
        fclose(STF->fp);
    }

    if (STF->header != NULL) {
        free(STF->header);
    }

    return;
}