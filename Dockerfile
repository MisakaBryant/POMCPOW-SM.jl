# 基于官方的 Julia 镜像
FROM docker.1ms.run/julia:1.12.5

# 设置工作目录
WORKDIR /app

# 将 Project.toml 和 Manifest.toml 文件复制到工作目录
COPY Project.toml Manifest.toml ./

# 切换为清华源
# 安装 lualatex 所需的依赖包
RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y \
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
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装 Project.toml 中的依赖包
RUN julia --project=. -e 'import Pkg; Pkg.instantiate()'

# docker run --name pomcpow-sm -it -v .:/app pomcpow-sm:1.0 /bin/bash

# xvfb-run julia --threads 24 --project=. runner/runner2.jl