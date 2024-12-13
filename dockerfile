# 使用 NVIDIA CUDA 镜像作为基础镜像
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

WORKDIR /app


# 安装 git 并增加 Git 缓冲区大小
RUN apt-get update && apt-get install -y git
RUN git config --global http.postBuffer 524288000

# 更新包列表并安装依赖，包括 Python3 和 pip
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-distutils \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
RUN apt-get update && apt-get install -y git && \
apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置 python3 指向 python3.9
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

RUN git config --global --add safe.directory /app

# 复制当前代码到容器中的 /app 目录
COPY . /app

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple './torch-1.13.1+cu116-cp38-cp38-linux_x86_64.whl'
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple './torchvision-0.14.1+cu116-cp38-cp38-linux_x86_64.whl'
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple './torchaudio-0.13.1+cu116-cp38-cp38-linux_x86_64.whl'

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt || \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt || \
    pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple -r requirements.txt || \
    pip install --no-cache-dir -i https://pypi.mirrors.ustc.edu.cn/simple/ -r requirements.txt


ENV PYTHONUNBUFFERED=1