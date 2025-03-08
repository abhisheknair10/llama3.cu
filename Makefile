# Variables
CC = clang
NVCC = nvcc
CFLAGS = -std=gnu17 -I/usr/local/cuda/include
NVCCFLAGS = -std=c++11 -I/usr/local/cuda/include
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart
SRC_DIR = ./src
OUTPUT_DIR = ./output

# Find all .c and .cu files in src directory and subdirectories
C_SRC_FILES = $(shell find $(SRC_DIR) -name '*.c')
CU_SRC_FILES = $(shell find $(SRC_DIR) -name '*.cu')

# Generate the corresponding .o files in the output directory
C_OBJS = $(patsubst $(SRC_DIR)/%, $(OUTPUT_DIR)/%, $(C_SRC_FILES:.c=.o))
CU_OBJS = $(patsubst $(SRC_DIR)/%, $(OUTPUT_DIR)/%, $(CU_SRC_FILES:.cu=.o))

# Collect all subdirectories under src for include paths
INCLUDES = $(shell find $(SRC_DIR) -type d -exec echo -I{} \;)

# All object files
OBJS = $(C_OBJS) $(CU_OBJS)

# Final target
all: $(OUTPUT_DIR)/output.out

# Run the program
run: all $(OUTPUT_DIR)/output.out
	if [ ! -f model_weights/tokenizer_modified.json ]; then \
		python3 setup-tokenizer.py; \
	fi
	clear && ./$(OUTPUT_DIR)/output.out

# Link the final executable
$(OUTPUT_DIR)/output.out: $(OBJS)
	$(NVCC) $(OBJS) $(INCLUDES) $(LDFLAGS) -o $@

# Compile .c files to object files
$(OUTPUT_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)  # Create directories as needed
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Compile .cu files to object files
$(OUTPUT_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)  # Create directories as needed
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

# Clean target
clean:
	rm -rf $(OUTPUT_DIR)

# Define phony targets
.PHONY: all run clean