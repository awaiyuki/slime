#include "marching_cubes_parallel.cuh"
#include "marching_cubes_tables.h"

using namespace slime;

__constant__ int slime::d_triangulation[256][16];
__constant__ int slime::d_cornerIndexFromEdge[12][2];

__device__ const int diff[8][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 0, 1 },
    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 1, 0 },
    { 1, 1, 1 }, { 0, 1, 1 }
};
__device__ float3 slime::interpolateVertices(float *d_scalarField, int gridSize,
                                             float surfaceLevel, int va[3],
                                             int vb[3]) {
  float scalarA =
      d_scalarField[va[2] * gridSize * gridSize + va[1] * gridSize + va[0]];
  float scalarB =
      d_scalarField[vb[2] * gridSize * gridSize + vb[1] * gridSize + vb[0]];
  float t = (surfaceLevel - scalarA) / (scalarB - scalarA);
  //printf("%d %d %f %f %f\n", va[0], vb[0], scalarA, scalarB, t);
  return make_float3(va[0], va[1], va[2]) +
         t * (make_float3(vb[0], vb[1], vb[2]) -
              make_float3(va[0], va[1], va[2]));
}

__global__ void slime::marchParallel(float *d_scalarField, int gridSize,
                                     float surfaceLevel,
                                     slime::VertexData *d_vertexDataPtr, int *d_counter) {

    

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int z = threadIdx.z + blockDim.z * blockIdx.z;
  printf("%f \n",
      d_scalarField[z * gridSize * gridSize + y * gridSize + x]);

  float3 currentPosition = make_float3(x, y, z);
  float3 cubeVertices[8];
  int cubeVertexCoordInt[8][3];

  for (int i = 0; i < 8; i++) {
    cubeVertices[i] =
        currentPosition + make_float3(diff[i][0], diff[i][1], diff[i][2]);
    cubeVertexCoordInt[i][0] = x + diff[i][0];
    cubeVertexCoordInt[i][1] = y + diff[i][1];
    cubeVertexCoordInt[i][2] = z + diff[i][2];
  }

  uint8_t tableKey = 0;
  for (int i = 0; i < 8; i++) {
    if (d_scalarField[(z + diff[i][2]) * gridSize * gridSize +
                      (y + diff[i][1]) * gridSize + (x + diff[i][0])] <
        surfaceLevel) { // correct?
      tableKey |= 1 << i;
    }
  }
  int* edges = d_triangulation[tableKey];

  for (int i = 0; i < 16; i += 3) {
    if (edges[i] == -1)
      continue;
    float3 v1Float3 = interpolateVertices(
        d_scalarField, gridSize, surfaceLevel,
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i]][0]],
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i]][1]]);

    glm::vec3 v1(v1Float3.x, v1Float3.y, v1Float3.z);

    float3 v2Float3 = interpolateVertices(
        d_scalarField, gridSize, surfaceLevel,
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i + 1]][0]],
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i + 1]][1]]);

    glm::vec3 v2(v2Float3.x, v2Float3.y, v2Float3.z);

    float3 v3Float3 = interpolateVertices(
        d_scalarField, gridSize, surfaceLevel,
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i + 2]][0]],
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i + 2]][1]]);

    glm::vec3 v3(v3Float3.x, v3Float3.y, v3Float3.z);

    // Need to correct!
    d_vertexDataPtr->vertices[*d_counter] = v1;
    d_vertexDataPtr->vertices[*d_counter + 1] = v2;
    d_vertexDataPtr->vertices[*d_counter + 2] = v3;

    atomicAdd(d_counter, 3);
    d_vertexDataPtr->size = *d_counter;
    // std::cout << "extract surface, triangle.v1[0]: " <<
    // triangle.v1[0]
    //           << std::endl;
  }
}