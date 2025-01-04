#ifndef CUDA_DEBUG_CUH
#define CUDA_DEBUG_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace slime {
void printCudaError(const char *errorTitle) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s: %s\n", errorTitle, cudaGetErrorString(err));
  }
}
} // namespace slime
#endif