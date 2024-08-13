#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "marching_cubes.cuh"
#include <glm/glm.hpp>

namespace slime {
__constant__ int d_triangulation[256][16];
__constant__ int d_cornerIndexFromEdge[12][2];

extern __device__ float3 interpolateVertices(float *scalarFieldDevice,
                                             int gridSize, float surfaceLevel,
                                             int va[3], int vb[3]);

extern __global__ void marchParallel(float *scalarFieldDevice, int gridSize,
                                     float surfaceLevel, glm::vec3 *d_vertices);

} // namespace slime