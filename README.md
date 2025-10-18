This project uses a Large Language Model (LLM) to generate a dataset used to train a Multi-Layer Perceptron (MLP) network.

This process is known as "Distilling" to take outputs from an LLM to train a smaller network.

Distilling the logic of an LLM down to an MLP provides complex decision-making at a fraction of the original compute cost.

## How
- Download the LLM GGUF: [download](https://huggingface.co/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/blob/main/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`)
- Launch llama.cpp server: `llama-server --port 8081 --mlock --ctx-size 512 --batch-size 512 --n-gpu-layers -1 -m Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`
- Launch the dataset generator: `php llm.php 8081`
- Once it's generated enough lines for the dataset use it to train the MLP: `python fit.py`

## Files
- `llm.php` - Uses [llama.cpp](https://github.com/ggml-org/llama.cpp) to generate the [training_data.txt](training_data.txt).
- `fit.py` - Uses the [training_data.txt](training_data.txt) to train an MLP Dense network using [Tensorflow](https://www.tensorflow.org/).

## Prerequisites
```
apt install php-cli php-curl
pip install numpy tensorflow
```

## Setup llama.cpp with Vulkan backend
```
apt update
apt install -y build-essential git cmake ninja-build pkg-config vulkan-tools mesa-vulkan-drivers libvulkan-dev
apt install -y glslc glslang-tools spirv-tools vulkan-tools
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build -j
```

## Notes
- Tensorflow generally trains small MLPs faster on a CPU than a GPU.
- LLM's via llama.cpp generally run faster on a GPU using the Vulkan backend.
- LLM used: [Qwen3-30B-A3B-Instruct-2507-Q4_K_M](https://huggingface.co/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF).
- Demo available at [romance.html](https://colinrizzman.github.io/romance)
