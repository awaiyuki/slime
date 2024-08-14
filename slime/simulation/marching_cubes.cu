#include "marching_cubes.cuh"

#include <bitset>
#include <iostream>

using namespace slime;
using namespace std;

MarchingCubes::MarchingCubes(int _gridSize) : gridSize(_gridSize) {

  vertexData.vertices = new glm::vec3[gridSize * gridSize * gridSize];
  vertexData.size = 0;

  glm::vec3 *d_vertices;
  cudaMalloc((void **)&d_vertices,
             sizeof(glm::vec3) * gridSize * gridSize * gridSize);

  VertexData d_vertexData;
  d_vertexData.size = 0;
  d_vertexData.vertices = d_vertices;
  cudaMalloc((void **)&d_vertexDataPtr, sizeof(VertexData));
  cudaMemcpy(d_vertexDataPtr, &d_vertexData, sizeof(VertexData),
             cudaMemcpyHostToDevice);
}

MarchingCubes::~MarchingCubes() {
  // cudaFree(d_vertices);
}

VertexData MarchingCubes::march(float *d_scalarField, float surfaceLevel) {

  /* copy constant arrays */
  cudaMemcpyToSymbol(d_triangulation, MarchingCubesTables::triangulation,
                     sizeof(int) * 256 * 16);
  cudaMemcpyToSymbol(d_cornerIndexFromEdge,
                     MarchingCubesTables::cornerIndexFromEdge,
                     sizeof(int) * 12 * 2);

  const int threadSize = 128;
  dim3 dimBlock(threadSize, threadSize, threadSize);
  const int blockSize = (gridSize + threadSize - 1) / threadSize;
  dim3 dimGrid(blockSize, blockSize, blockSize);
  marchParallel<<<dimGrid, dimBlock>>>(d_scalarField, gridSize, surfaceLevel,
                                       d_vertexDataPtr);
  cudaDeviceSynchronize();
  /* get triangles from device and return */
  VertexData tempVertexData;
  cudaMemcpy(&tempVertexData, d_vertexDataPtr, sizeof(vertexData),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(vertexData.vertices, tempVertexData.vertices,
             sizeof(glm::vec3) * tempVertexData.size, cudaMemcpyDeviceToHost);
  vertexData.size = tempVertexData.size;

  cout << vertexData.size << endl;
  cout << vertexData.vertices[0].x;
  return vertexData;
};