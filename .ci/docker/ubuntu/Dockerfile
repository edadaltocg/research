ARG OS_VERSION

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu${OS_VERSION}

ARG OS_VERSION

ENV DEBIAN_FRONTEND noninteractive

# Install common dependencies
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install clang
ARG CLANG_VERSION
COPY ./common/install_clang.sh install_clang.sh
RUN bash ./install_clang.sh && rm install_clang.sh

# Install gcc
ARG GCC_VERSION
COPY ./common/install_gcc.sh install_gcc.sh
RUN bash ./install_gcc.sh && rm install_gcc.sh

# Setup user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install uv and other dependencies
ARG PYTHON_VERSION
ENV PYTHON_VERSION=$PYTHON_VERSION
COPY ./common/utils.sh utils.sh
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod +x /install.sh && /install.sh --bin-dir /usr/local/bin && rm /install.sh

# Install Python using uv
RUN uv python install ${PYTHON_VERSION}

RUN rm -f utils.sh

USER ci-user
CMD ["bash"]
