#include "marching_cubes_device.cuh"
#include "marching_cubes_tables.h"
#include <stdio.h>
#define EPSILON 1e-6

using namespace slime;

__constant__ int slime::d_triangulation[256][16];
__constant__ int slime::d_cornerIndexFromEdge[12][2];

__device__ const int diff[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1},
                                   {0, 1, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1}};
__device__ float3 slime::interpolateVertices(float *d_scalarField, int gridSize,
                                             float surfaceLevel, int va[3],
                                             int vb[3]) {
  float scalarA =
      d_scalarField[va[2] * gridSize * gridSize + va[1] * gridSize + va[0]];
  float scalarB =
      d_scalarField[vb[2] * gridSize * gridSize + vb[1] * gridSize + vb[0]];
  float scale = 1.0f / gridSize;
  float t = (surfaceLevel - scalarA) / (scalarB - scalarA + EPSILON);
  // printf("%d %d %f %f %f\n", va[0], vb[0], scalarA, scalarB, t);
  return (make_float3(va[0], va[1], va[2]) +
          t * (make_float3(vb[0], vb[1], vb[2]) -
               make_float3(va[0], va[1], va[2]))) *
         scale;
}

__global__ void slime::marchParallel(float *d_scalarField, int gridSize,
                                     float surfaceLevel,
                                     slime::VertexData *d_vertexDataPtr,
                                     int *d_counter) {

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int z = threadIdx.z + blockDim.z * blockIdx.z;
  //  printf("%f \n",
  //   d_scalarField[z * gridSize * gridSize + y * gridSize + x]);

  if (x >= gridSize - 1 || y >= gridSize - 1 || z >= gridSize - 1)
    return;
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

  int tableKey = 0;
  for (int i = 0; i < 8; i++) {
    if (d_scalarField[(z + diff[i][2]) * gridSize * gridSize +
                      (y + diff[i][1]) * gridSize + (x + diff[i][0])] <
        surfaceLevel) {
      tableKey |= 1 << i;
    }
  }

  int *edges = d_triangulation[tableKey];

  for (int i = 0; i < 16; i += 3) {
    // if (edges[i] == -1) {
    //   // temporary solution

    //   // 이 부분도 실제로는 그리는 게 문제인듯

    //   const int cellIndex = z * gridSize * gridSize + y * gridSize + x;
    //   d_vertexDataPtr->vertices[15 * cellIndex + i] =
    //       make_float3(0.0f, 0.0f, 0.0f);
    //   d_vertexDataPtr->vertices[15 * cellIndex + i + 1] =
    //       make_float3(0.0f, 0.0f, 0.0f);
    //   d_vertexDataPtr->vertices[15 * cellIndex + i + 2] =
    //       make_float3(0.0f, 0.0f, 0.0f);
    //   continue;
    // }
    float3 v1Float3 = interpolateVertices(
        d_scalarField, gridSize, surfaceLevel,
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i]][0]],
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i]][1]]);

    float3 v2Float3 = interpolateVertices(
        d_scalarField, gridSize, surfaceLevel,
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i + 1]][0]],
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i + 1]][1]]);

    float3 v3Float3 = interpolateVertices(
        d_scalarField, gridSize, surfaceLevel,
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i + 2]][0]],
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i + 2]][1]]);

    // printf("%d\n", *d_counter);
    const int cellIndex = z * gridSize * gridSize + y * gridSize + x;
    d_vertexDataPtr->vertices[*d_counter + i] = v1Float3;
    d_vertexDataPtr->vertices[*d_counter + i + 1] = v2Float3;
    d_vertexDataPtr->vertices[*d_counter + i + 2] = v3Float3;
    atomicAdd(d_counter, 3);

    // std::cout << "extract surface, triangle.v1[0]: " <<
    // triangle.v1[0]
    //           << std::endl;
  }
}