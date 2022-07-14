#!/bin/bash

# build.sh

export ANACONDA_PATH=$(dirname $(dirname $(which conda)))

rm -rf build
mkdir -p build
cd build
cmake ..
make -j12 VERBOSE=1
cd ..
mv build/_mgmmf_cpp.so mgmmf/_mgmmf_cpp.so
rm -rf build
