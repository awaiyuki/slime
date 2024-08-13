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
class MarchingCubes {
private:
  float *d_scalarField;
  int gridSize;

public:
  struct Triangle {
    float3 v1, v2, v3;
  };

  MarchingCubes(int gridSize);
  ~MarchingCubes();

  std::vector<MarchingCubes::Triangle> march(float *scalarField,
                                             float surfaceLevel);
};
} // namespace slime
#endif