# Base image on the official NVIDIA-PyTorch image (Optimized for PyTorch & NVIDIA GPUs)

FROM nvcr.io/nvidia/pytorch:24.02-py3

# Install system dependencies

RUN apt-get update && apt-get install -y \ 
    libosmesa6-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev

# Install python packages (from requirements.txt)

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set the working directory

WORKDIR /workspace





