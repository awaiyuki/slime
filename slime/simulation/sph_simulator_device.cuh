#ifndef SPH_SIMULATOR_DEVICE_CUH
#define SPH_SIMULATOR_DEVICE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "sph_simulator.cuh"

namespace slime {

/* TODO: update naming of the global kernels(remove 'Device' and add something)
 */

extern __global__ void updateScalarFieldDevice(float *colorFieldDevice,
                                               Particle *particlesDevice,
                                               int gridSize,
                                               float maxColorQuantity);

extern __global__ void computeDensityDevice(Particle *particlesDevice);

extern __global__ void computePressureDevice(Particle *particlesDevice);

extern __global__ void computePressureForceDevice(Particle *particlesDevice,
                                                  double deltaTime);

extern __global__ void computeViscosityForceDevice(Particle *particlesDevice,
                                                   double deltaTime);

extern __global__ void computeSurfaceTensionDevice(Particle *particlesDevice,
                                                   double deltaTime);

extern __global__ void computeGravityDevice(Particle *particlesDevice,
                                            double deltaTime);

extern __global__ void computeWallConstraintDevice(Particle *particlesDevice,
                                                   double deltaTime);

extern __global__ void computePositionDevice(Particle *particlesDevice,
                                             double deltaTime);

extern __global__ void updateSpatialHashDevice(Particle *particlesDevice,
                                               unsigned int *hashKeys,
                                               unsigned int *hashIndices);
extern __global__ void updateHashBucketDevice(unsigned int *hashKeys,
                                              unsigned int *hashIndices);

extern __global__ void copyPositionToVBODevice(float *d_positions,
                                               Particle *particlesDevice);
} // namespace slime

#endif