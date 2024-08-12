
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

extern __global__ void slime::updateParticlesDevice(Particle *particlesDevice,
                                                    double deltaTime);

extern __device__ void computeDensityDevice(Particle *particlesDevice);

extern __device__ void computePressureForceDevice(Particle *particlesDevice,
                                                  double deltaTime);

extern __device__ void computeViscosityForceDevice(Particle *particlesDevice,
                                                   double deltaTime);

extern __device__ void computeGravityDevice(Particle *particlesDevice,
                                            double deltaTime);

extern __device__ void computeWallConstraintDevice(Particle *particlesDevice,
                                                   double deltaTime);
} // namespace slime