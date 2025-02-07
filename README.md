# slime
3D SPH-based Fluid Simulation using CUDA and OpenGL

## Current Status
The core features have been implemented, and the project is currently undergoing debugging and improvement, focusing on fine-tuning SPH simulation accuracy (adjusting constants) and enhancing marching cubes rendering performance.

## Features
- SPH-based Fluid Simulation
- Surface Extraction using Marching Cubes Algorithm
- GPU Parallel Processing Using CUDA (CUDA-OpenGL Interop)
- Spatial Hashing

## Future Updates
- Lighting

## How to clone with submodules
```bash
git clone --recursive https://github.com/harutea/slime.git
```
Or,
```bash
git clone https://github.com/harutea/slime.git
cd slime
git submodule update --recursive --init
```

## How to build
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
