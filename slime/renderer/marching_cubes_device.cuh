#ifndef MARCHING_CUBES_DEVICE_CUH
#define MARCHING_CUBES_DEVICE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <slime/utility/cuda_math.cuh>

namespace slime {
struct VertexData {
  float3 *vertices;
  int size;
};
extern __constant__ int d_triangulation[256][16];
extern __constant__ int d_cornerIndexFromEdge[12][2];

extern __device__ float3 interpolateVertices(float *d_scalarField, int gridSize,
                                             float surfaceLevel, int va[3],
                                             int vb[3]);

extern __global__ void marchParallel(float *d_scalarField, int gridSize,
                                     float surfaceLevel,
                                     slime::VertexData *d_vertexDataPtr,
                                     int *d_counter);

} // namespace slime

#endif