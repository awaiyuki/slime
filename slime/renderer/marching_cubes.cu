#include "marching_cubes.cuh"
#include "marching_cubes_tables.h"
#include <slime/constants/marching_cubes_constants.h>
#include <slime/utility/cuda_debug.cuh>
#include <bitset>
#include <iostream>

using namespace slime;
using namespace slime::MarchingCubesConstants;
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

void MarchingCubes::march(cudaGraphicsResource_t cudaVBOResource,
                          float *d_scalarField, float surfaceLevel) {

  cudaMemset(d_counter, 0, sizeof(int));

  const int threadSize = THREAD_SIZE_IN_MARCH;
  dim3 dimBlock(threadSize, threadSize, threadSize);
  const int blockSize = (gridSize + threadSize - 1) / threadSize;
  dim3 dimGrid(blockSize, blockSize, blockSize);

  // cudaMemset(d_vertexDataPtr->vertices, 0.0, sizeof(float3) * gridSize *
  // gridSize * gridSize * 15);
  g_march<<<dimGrid, dimBlock>>>(d_scalarField, gridSize, surfaceLevel,
                                 d_vertexDataPtr, d_counter);

  cudaDeviceSynchronize();
  printCudaError("g_march");

  // Ensure d_counter is updated
  int h_counter;
  cudaError_t err =
      cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("cudaMemcpy error: %s\n", cudaGetErrorString(err));
  } else {
    printf("h_counter = %d\n", h_counter);
  }

  if (h_counter == 0) {
    return;
  }

  /* get triangles from device and return */
  // VertexData tempVertexData;
  // cudaMemcpy(&tempVertexData, d_vertexDataPtr, sizeof(VertexData),
  //            cudaMemcpyDeviceToHost);
  // cudaMemcpy(vertexData.vertices, tempVertexData.vertices,
  //            sizeof(float3) * gridSize * gridSize * gridSize * 15,
  //            cudaMemcpyDeviceToHost);
  // int h_counter;
  // cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  // // vertexData.size = gridSize * gridSize * gridSize * 15;
  // cout << h_counter << endl;
  // vertexData.size = h_counter;
  // //   cudaMemcpy(&vertexData.size, d_counter, sizeof(int),
  // //   cudaMemcpyDeviceToHost);
  // cout << vertexData.size << endl;
  // cout << vertexData.vertices[0].x << endl;

  /* CUDA-OpenGL interop */
  cudaGraphicsMapResources(1, &cudaVBOResource, 0);
  float *d_positions;
  size_t size;
  cudaGraphicsResourceGetMappedPointer((void **)&d_positions, &size,
                                       cudaVBOResource);
  printCudaError("cudaGraphicsResourceGetMappedPointer in marching cubes");

  cout << "cudavboresource size: " << size << endl;

  int totalElements = gridSize * gridSize * gridSize * 15;
  int numThreads = THREAD_SIZE_IN_COPY_VERTEX_DATA;
  int numBlocks = (h_counter + numThreads - 1) / numThreads;

  g_copyVertexDataToVBO<<<numBlocks, numThreads>>>(d_positions, d_vertexDataPtr,
                                                   h_counter);
  cudaDeviceSynchronize();
  printCudaError("copyVertexDataToVBODevice");

  cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);
};