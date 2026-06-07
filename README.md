# slime

Real-time 3D SPH fluid simulation with a smooth Marching Cubes surface.

## Features

- SPH fluid simulation with fixed time steps
- Apple Metal compute backend on macOS
- Portable CPU fallback on Windows and Linux
- Smooth `64 x 64 x 64` Marching Cubes surface
- Continuous density field smoothing and opaque depth rendering
- OpenGL rendering with GLFW

The macOS build runs the particle simulation on Metal and extracts the
Marching Cubes surface on the CPU. The original CUDA implementation remains in
the repository but is not part of the default portable build.

## Requirements

- CMake 3.18 or newer
- A C++17 compiler
- Git submodules initialized
- macOS: Xcode command-line tools and a Metal-capable Mac

## Clone

```bash
git clone --recursive https://github.com/harutea/slime.git
cd slime
```

For an existing clone:

```bash
git submodule update --recursive --init
```

## Build And Run

macOS and Linux:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/slime
```

Windows:

```powershell
cmake -S . -B build
cmake --build build --config Release
.\build\Release\slime.exe
```

## Controls

- `W`, `A`, `S`, `D`: Move camera
- Mouse: Look around
- Scroll: Zoom
- `C`: Toggle background color
- `Escape`: Release cursor
- `F`: Capture cursor

## Current Limitations

- The Metal SPH kernel currently uses an all-pairs neighbor search.
- Marching Cubes extraction currently runs on the CPU.
- The portable CPU backend uses fewer particles than the original CUDA path.
