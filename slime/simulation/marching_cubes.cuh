#ifndef MARCHING_CUBES_H
#define MARCHING_CUBES_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <iostream>
#include "marching_cubes_tables.h"
#include "marching_cubes_parallel.cu"

namespace slime {
struct VertexData {
  glm::vec3 *vertices;
  int size;
};
class MarchingCubes {
private:
  int gridSize;
  VertexData vertexData, *d_vertexDataPtr;

public:
  MarchingCubes(int _gridSize);
  ~MarchingCubes();

  VertexData march(float *scalarField, float surfaceLevel);
};
} // namespace slime
#endif