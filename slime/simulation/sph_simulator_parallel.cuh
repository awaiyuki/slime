
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include "sph_simulator.cuh"

namespace slime {
extern __device__ float poly6KernelDevice(glm::vec3 r, float h);
extern __device__ float spikyKernelDevice(glm::vec3 r, float h);
extern __device__ float gradientSpikyKernelDevice(glm::vec3 r, float h);
extern __device__ float viscosityKernelDevice(glm::vec3 r, float h);
extern __device__ float laplacianViscosityKernelDevice(glm::vec3 r, float h);

extern __global__ void updateScalarFieldDevice(float *colorFieldDevice,
                                               Particle *particlesDevice,
                                               int gridSize);

extern __global__ void computeDensityDevice(Particle *particlesDevice);

extern __global__ void computePressureDevice(Particle *particlesDevice);

extern __global__ void computePressureForceDevice(Particle *particlesDevice,
                                                  double deltaTime);

extern __global__ void computeViscosityForceDevice(Particle *particlesDevice,
                                                   double deltaTime);

extern __global__ void computeGravityDevice(Particle *particlesDevice,
                                            double deltaTime);

extern __global__ void computeWallConstraintDevice(Particle *particlesDevice,
                                                   double deltaTime);

extern __global__ void computePositionParallel(Particle *particlesDevice,
                                               double deltaTime)
} // namespace slime