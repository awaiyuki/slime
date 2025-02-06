#ifndef MARCHING_CUBES_CUH
#define MARCHING_CUBES_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <iostream>
#include <glm/glm.hpp>
#include "marching_cubes_parallel.cuh"

namespace slime {
class MarchingCubes {
private:
  int gridSize;

  float3 *d_vertices;
  VertexData vertexData, *d_vertexDataPtr;
  cudaGraphicsResource_t cudaVBOResource;

public:
  MarchingCubes(int _gridSize);
  ~MarchingCubes();

  void march(float *scalarField, float surfaceLevel);
};
} // namespace slime
#endif