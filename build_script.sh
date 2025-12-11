#!/bin/bash

set -e
set -x

######## User Configurations ########
# Source directory
adios_src_dir=.
# Build directory
adios_build_dir=./adios2-build
adios_install_dir=$(pwd)/install

mkdir -p ${adios_build_dir}
mkdir -p ${adios_install_dir}

export PKG_CONFIG_PATH=~/Software/miniforge3/envs/py311torch/lib/pkgconfig:$PKG_CONFIG_PATH

cmake -S ${adios_src_dir} -B ${adios_build_dir} \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
  -DTorch_DIR="~/Software/miniforge3/envs/py311torch/lib/python3.11/site-packages/torch/share/cmake/Torch" \
  -DADIOS2_USE_BZip2=OFF \
  -DADIOS2_USE_HDF5=OFF \
  -DADIOS2_USE_GPU_Support=ON \
  -DADIOS2_USE_CUDA=ON \
  -DCMAKE_PREFIX_PATH="$HOME/Projects/CAESAR_ALL/CAESAR_C/install/" \
  -DCMAKE_INSTALL_PREFIX=${adios_install_dir}

cmake --build ${adios_build_dir} -- -j$(nproc)
cmake --install ${adios_build_dir}