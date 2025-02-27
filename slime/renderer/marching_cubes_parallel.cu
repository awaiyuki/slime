#include "marching_cubes_parallel.cuh"
#include "marching_cubes_tables.h"
#include <stdio.h>
#define EPSILON 1e-6

using namespace slime;

__constant__ int slime::d_triangulation[256][16];
__constant__ int slime::d_cornerIndexFromEdge[12][2];

__device__ const int diff[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                   {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};

__device__ float3 slime::interpolateVertices(float *d_scalarField, int gridSize,
                                             float surfaceLevel, int va[3],
                                             int vb[3]) {
  float scalarA =
      d_scalarField[va[2] * gridSize * gridSize + va[1] * gridSize + va[0]];
  float scalarB =
      d_scalarField[vb[2] * gridSize * gridSize + vb[1] * gridSize + vb[0]];
  float t = (fabs(scalarB - scalarA) > EPSILON)
                ? (surfaceLevel - scalarA) / (scalarB - scalarA)
                : 0.5f;
  // printf("%d %d %f %f %f\n", va[0], vb[0], scalarA, scalarB, t);
  const float3 posA = make_float3(va[0], va[1], va[2]) / float(gridSize) -
                      make_float3(0.5f, 0.5f, 0.5f);
  const float3 posB = make_float3(vb[0], vb[1], vb[2]) / float(gridSize) -
                      make_float3(0.5f, 0.5f, 0.5f);
  return (posA + t * (posB - posA));
}

__global__ void slime::g_march(float *d_scalarField, int gridSize,
                               float surfaceLevel,
                               slime::VertexData *d_vertexDataPtr,
                               int *d_counter) {

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int z = threadIdx.z + blockDim.z * blockIdx.z;

  if (x >= gridSize - 1 || y >= gridSize - 1 || z >= gridSize - 1)
    return;
  int cubeVertexCoord[8][3];

  for (int i = 0; i < 8; i++) {
    cubeVertexCoord[i][0] = x + diff[i][0];
    cubeVertexCoord[i][1] = y + diff[i][1];
    cubeVertexCoord[i][2] = z + diff[i][2];
  }

  int tableKey = 0;
  for (int i = 0; i < 8; i++) {
    float scalarValue =
        d_scalarField[(z + diff[i][2]) * gridSize * gridSize +
                      (y + diff[i][1]) * gridSize + (x + diff[i][0])];
    if (scalarValue > surfaceLevel) {
      tableKey |= 1 << i;
    }
  }

  int *edges = d_triangulation[tableKey];

  for (int i = 0; i < 16; i += 3) {
    if (edges[i] == -1) {
      break;
    }
    float3 v1 = interpolateVertices(
        d_scalarField, gridSize, surfaceLevel,
        cubeVertexCoord[d_cornerIndexFromEdge[edges[i]][0]],
        cubeVertexCoord[d_cornerIndexFromEdge[edges[i]][1]]);

    float3 v2 = interpolateVertices(
        d_scalarField, gridSize, surfaceLevel,
        cubeVertexCoord[d_cornerIndexFromEdge[edges[i + 1]][0]],
        cubeVertexCoord[d_cornerIndexFromEdge[edges[i + 1]][1]]);

    float3 v3 = interpolateVertices(
        d_scalarField, gridSize, surfaceLevel,
        cubeVertexCoord[d_cornerIndexFromEdge[edges[i + 2]][0]],
        cubeVertexCoord[d_cornerIndexFromEdge[edges[i + 2]][1]]);

    int offset = atomicAdd(d_counter, 3); // increase number of vertices
    printf("d_counter: %d\n", *d_counter);
    d_vertexDataPtr->vertices[offset] = v1;
    d_vertexDataPtr->vertices[offset + 1] = v2;
    d_vertexDataPtr->vertices[offset + 2] = v3;
  }
}

__global__ void slime::g_copyVertexDataToVBO(float *d_positions,
                                             slime::VertexData *d_vertexDataPtr,
                                             const int numVertices) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= numVertices)
    return;
  // if (idx < 10) {
  //   printf("Thread %d: copying vertex %f, %f, %f\n", idx,
  //          d_vertexDataPtr->vertices[idx].x,
  //          d_vertexDataPtr->vertices[idx].y,
  //          d_vertexDataPtr->vertices[idx].z);
  // }
  d_positions[3 * idx] = d_vertexDataPtr->vertices[idx].x;
  d_positions[3 * idx + 1] = d_vertexDataPtr->vertices[idx].y;
  d_positions[3 * idx + 2] = d_vertexDataPtr->vertices[idx].z;
}