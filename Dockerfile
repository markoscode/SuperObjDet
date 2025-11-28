# Dockerfile for WS-Mask2Former inference service
# Build: docker build -t ws-mask2former .
# Test:  docker run --gpus all ws-mask2former python inference_server.py --print-labels

# PyTorch 1.10 + CUDA 11.3 - known to work with mask2former CUDA ops
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Critical fix for Detectron2 v0.6 build on PyTorch 1.10
# Prevents: AttributeError: module 'distutils' has no attribute 'version'
RUN pip install setuptools==59.5.0

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Fix expired NVIDIA GPG key for Ubuntu 18.04
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    opencv-python-headless \
    pillow \
    scipy \
    timm==0.6.13 \
    fvcore \
    omegaconf

# Install detectron2 v0.6 (compatible with PyTorch 1.10)
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# Set working directory
WORKDIR /workspace

# Copy mask2former module first to compile CUDA ops
COPY mask2former /workspace/mask2former
COPY configs /workspace/configs

# Compile MultiScaleDeformableAttention CUDA op
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
WORKDIR /workspace/mask2former/modeling/pixel_decoder/ops
RUN python setup.py build install

# Back to workspace root
WORKDIR /workspace

# Set PYTHONPATH
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# Copy model weights and config
COPY sem_ade_output_2/config.yaml /workspace/sem_ade_output_2/config.yaml
COPY sem_ade_output_2/model_final.pth /workspace/sem_ade_output_2/model_final.pth

# Copy inference script and test image
COPY inference_server.py /workspace/inference_server.py
COPY input.jpg /workspace/input.jpg

# Default command
CMD ["python", "inference_server.py", "--print-labels"]
