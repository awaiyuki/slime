# slime
3D SPH-based Fluid Simulation using CUDA and OpenGL

## Current Status
The core features have been implemented, but there are still plenty of bugs to work through. The project is in the debugging and improvement, focusing on fine-tuning SPH simulation accuracy (by adjusting constants) and boosting rendering performance.

## Features
- SPH-based Fluid Simulation
- Surface Extraction using Marching Cubes Algorithm
- GPU Parallel Processing Using CUDA (CUDA-OpenGL Interop)
- Spatial Hashing for Neighbor Search (Best case: $`O(1)`$, Worst case: $`O(n)`$)

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
