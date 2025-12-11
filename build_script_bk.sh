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

cmake -S ${adios_src_dir} -B ${adios_build_dir} \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
  -DADIOS2_USE_HDF5=OFF \
  -DADIOS2_USE_CAESAR=ON \
  -DCMAKE_PREFIX_PATH="$HOME/Projects/CAESAR_ALL/CAESAR_C/install/;$HOME/Software/miniforge3/envs/py311torch/lib/python3.11/site-packages/torch/share/cmake" \
  -DCMAKE_INSTALL_PREFIX=${adios_install_dir}

cmake --build ${adios_build_dir} -- -j$(nproc)
cmake --install ${adios_build_dir}