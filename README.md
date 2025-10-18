## Files
- `llm.py` - Uses [llama.cpp](https://github.com/ggml-org/llama.cpp) to generate the [training_data.txt](training_data.txt).
- `fit.py` - Uses the [training_data.txt](training_data.txt) to train an MLP Dense network using [Tensorflow](https://www.tensorflow.org/).

## Prerequisites
```
apt install php-cli php-curl
pip install numpy tensorflow
```

## Setup llama.cpp with Vulkan backend
```
apt update
apt install -y build-essential git cmake ninja-build pkg-config vulkan-tools mesa-vulkan-drivers
apt install -y glslc glslang-tools spirv-tools vulkan-tools
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build -j
```

## Notes
- Tensorflow generally trains small MLP's faster on a CPU than a GPU.
- LLM's via llama.cpp generally run faster on a GPU using the Vulkan backend.
- Demo available at [romance.html](https://colinrizzman.github.io/romance)
