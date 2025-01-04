#include "marching_cubes.cuh"
#include "marching_cubes_tables.h"

#include <bitset>
#include <iostream>

using namespace slime;
using namespace std;

__device__ int *d_counter;

MarchingCubes::MarchingCubes(int _gridSize) : gridSize(_gridSize) {

  vertexData.vertices = new float3[gridSize * gridSize * gridSize * 15];
  vertexData.size = 0;

  float3 *d_vertices;
  cudaMalloc((void **)&d_vertices,
             sizeof(float3) * gridSize * gridSize * gridSize * 15);

  VertexData tempVertexData;
  tempVertexData.size = 0;
  tempVertexData.vertices = d_vertices;
  cudaMalloc((void **)&d_vertexDataPtr, sizeof(VertexData));
  cudaMemcpy(d_vertexDataPtr, &tempVertexData, sizeof(VertexData),
             cudaMemcpyHostToDevice);

  /* copy constant arrays */
  cudaMemcpyToSymbol(d_triangulation, MarchingCubesTables::triangulation,
                     sizeof(int) * 256 * 16);
  cudaMemcpyToSymbol(d_cornerIndexFromEdge,
                     MarchingCubesTables::cornerIndexFromEdge,
                     sizeof(int) * 12 * 2);

  cudaMalloc((void **)&d_counter, sizeof(int));
}

MarchingCubes::~MarchingCubes() {
  if (vertexData.vertices) {
    delete[] vertexData.vertices;
  }
  if (d_vertexDataPtr) {
    cudaFree(d_vertexDataPtr);
  }
  if (d_counter) {
    cudaFree(d_counter);
  }
}

VertexData MarchingCubes::march(float *d_scalarField, float surfaceLevel) {

  cudaMemset(d_counter, 0, sizeof(int));

  const int threadSize = 8;
  dim3 dimBlock(threadSize, threadSize, threadSize);
  const int blockSize = (gridSize + threadSize - 1) / threadSize;
  dim3 dimGrid(blockSize, blockSize, blockSize);

  // cudaMemset(d_vertexDataPtr->vertices, 0.0, sizeof(float3) * gridSize *
  // gridSize * gridSize * 15);
  marchParallel<<<dimGrid, dimBlock>>>(d_scalarField, gridSize, surfaceLevel,
                                       d_vertexDataPtr, d_counter);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("marchParallel error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  /* get triangles from device and return */
  VertexData tempVertexData;
  cudaMemcpy(&tempVertexData, d_vertexDataPtr, sizeof(VertexData),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(vertexData.vertices, tempVertexData.vertices,
             sizeof(float3) * gridSize * gridSize * gridSize * 15,
             cudaMemcpyDeviceToHost);
  int h_counter;
  cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  // vertexData.size = gridSize * gridSize * gridSize * 15;
  cout << h_counter << endl;
  vertexData.size = h_counter;
  //   cudaMemcpy(&vertexData.size, d_counter, sizeof(int),
  //   cudaMemcpyDeviceToHost);
  cout << vertexData.size << endl;
  // cout << vertexData.vertices[0].x << endl;
  return vertexData;
};