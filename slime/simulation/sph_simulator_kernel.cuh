
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include "sph_simulator.h"

namespace slime {
extern __device__ float poly6KernelDevice(glm::vec3 r, float h);

extern __global__ void updateScalarFieldDevice(float *colorFieldDevice,
                                               Particle *particlesDevice,
                                               int gridSize);

extern __device__ void computeDensityDevice();

extern __device__ void computePressureForceDevice(double deltaTime);

extern __device__ void computeViscosityForceDevice(double deltaTime);

extern __device__ void computeGravityDevice(double deltaTime);

extern __device__ void computeWallConstraintDevice(double deltaTime);
} // namespace slime