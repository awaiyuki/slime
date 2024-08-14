# slime
3D Slime (Viscoelastic Fluid) Simulation using OpenGL (on progress)

## Features
- SPH-based Fluid Simulation
- Simulation Parallelized Using CUDA
- Surface Extraction using Marching Cubes Algorithm

## Clone with submodules
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
cmake .. -G "Unix Makefiles"
make
```
