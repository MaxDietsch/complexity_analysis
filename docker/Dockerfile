ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# Fetch the key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MIM
RUN pip install openmim

# Install MMPretrain
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmpretrain.git

WORKDIR /mmpretrain
RUN mim install --no-cache-dir -e .

# Copy the inference script and any other necessary files
COPY ../data mmpretrain/data
COPY ../runtime mmpretrain/runtime
COPY ../schedule mmpretrain/schedule
COPY ../model mmpretrain/model
COPY ../weights /mmpretrain/weights
COPY ../tools_created /mmpretrain/tools_created
COPY ../config mmpretrain/config

# Run the inference script with the specified path parameter
CMD ["python", "/mmpretrain/tools_created/inference_light.py", "/path/to/your/image"]

# To keep the container running after executing the script, you can add a tail command
# This is useful for debugging or if you need the container to stay alive for some reason
# CMD ["sh", "-c", "python /mmpretrain/tools_created/inference_light.py /path/to/your/image && tail -f /dev/null"]

