# Llama3.cu - A Llama 3 (8B) CUDA Inference Engine

<div align="center">
  <img src="https://github.com/abhisheknair10/Llama3.cu/blob/main/llama.jpeg" alt="inference" width="600">
</div>

##

Llama3.cu is a CUDA native implementation of the LLaMA3 architecture for causal language modeling. Core principles of the transformer architecture from the papers [Attention is All You Need](https://arxiv.org/abs/1706.03762) and [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) are implemented using custom CUDA kernel definitions, enabling scalable parallel processing on Nvidia GPUs.

The models are expected to be downloaded off of HuggingFace. They are stored as BF16 parameter weights in a .safetensor file, which during load time to the CUDA device, is converted to FP16 via a FP32 proxy. Hence, a CUDA device with a minimum of 24GB VRAM must be used.

## Setup and Usage

### Minimum Requirements:

```bash
- 24GB+ VRAM CUDA Device
- HuggingFace account
- Operating System: UNIX or WSL
- CUDA Toolkit (7.5+)
```

### Run Inference

1. Run the **[setup-docker.sh](https://github.com/abhisheknair10/Llama3.cu/blob/main/setup-docker.sh)** file to setup your Virtual/Physical Machine to run Docker with access to Nvidia GPUs. Once the shell script has finished executing, make sure to log out of the terminal, and then log back in to run **[run-docker.sh](https://github.com/abhisheknair10/Llama3.cu/blob/main/run-docker.sh)**.

```bash
# Setup Docker
chmod +x setup-docker.sh
./setup-docker.sh
```

```bash
# Restart terminal and run
chmod +x run-docker.sh
./run-docker.sh
```

2. For this inference engine to work, a SafeTensor formatted file(s) of the Llama3-8b model needs to be stored in the ./model_weights/ folder. Head to the [HuggingFace - meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) repo to get access to the model. Additionally, [Generate a Hugging Face Token](https://huggingface.co/settings/tokens) so that the next step can successfully download the weights files.

3. Once the Docker container has started up, run the following command to store the Hugging Face token as an environment variable, replacing **<your_token>** with the token you generated.

```bash
export HF_TOKEN=<your_token>
```

4. Next, run the following command to download the model parameters into the target directory.

```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./model_weights/ --token $HF_TOKEN
```

5. Run Make 🎉.

```bash
make run
```

<div align="center">
  <img src="https://github.com/abhisheknair10/Llama3.cu/blob/main/inference.png" alt="inference" width="600">
</div>

## Acknowledgments

Non exhaustive list of sources:

1. [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762)

1. [**LLaMA: Open and Efficient Foundation Language Models**](https://arxiv.org/abs/2302.13971)

1. [**RoPE: Rotary Position Embedding for Robust, Efficient Transformer Models**](https://arxiv.org/abs/2104.09864)

1. This project makes use of the [cJSON library by DaveGamble](https://github.com/DaveGamble/cJSON), which is licensed under the MIT License.
