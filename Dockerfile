# 基于官方的 Julia 镜像
FROM docker.m.daocloud.io/julia:latest

# 设置工作目录
WORKDIR /app

# 将 Project.toml 和 Manifest.toml 文件复制到工作目录
COPY Project.toml Manifest.toml ./

# 安装 lualatex 所需的依赖包
RUN apt-get update && apt-get install -y \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-latex-recommended \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-science \
    lmodern \
    pdf2svg \
    libgtk-3-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装 Project.toml 中的依赖包
RUN julia --project=. -e 'import Pkg; Pkg.instantiate()'

# 设置默认的启动命令（可以根据需要修改）
CMD ["julia", "--project=.", "runner/runner2.jl"]

# docker run --name pomdp -it -v .:/app pomdp:1.2 /bin/bash

# xvfb-run julia --threads 24 --project=. runner/runner2.jl