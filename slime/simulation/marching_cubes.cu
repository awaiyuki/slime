#include "marching_cubes.cuh"

#include <bitset>
#include <iostream>

using namespace slime;
using namespace std;

MarchingCubes::MarchingCubes(int gridSize) {
  cudaMalloc((void **)&d_scalarField,
             sizeof(float) * gridSize * gridSize * gridSize);
}

MarchingCubes::~MarchingCubes() {}

std::vector<MarchingCubes::Triangle> MarchingCubes::march(float *scalarField,
                                                          float surfaceLevel) {

  /* triangles to scalarField, device */
  cudaMemcpy(d_scalarField, scalarField,
             sizeof(float) * gridSize * gridSize * gridSize,
             cudaMemcpyHostToDevice);
  cudaMemcpy(...);
  /* copy constant arrays */

  const int threadSize = 128;
  dim3 dimBlock(threadSize, threadSize, threadSize);
  const int blockSize = (gridSize + threadSize - 1) / threadSize;
  dim3 dimGrid(blockSize, blockSize, blockSize);
  marchParallel<<<dimGrid, dimBlock>>>(d_scalarField, gridSize, surfaceLevel,
                                       d_triangles);
  cudaDeviceSynchronize();
  /* get triangles from device and return */
  cudaMemcpy(...);
};