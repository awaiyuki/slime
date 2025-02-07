#ifndef SPH_SIMULATOR_DEVICE_CUH
#define SPH_SIMULATOR_DEVICE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "sph_simulator.cuh"

namespace slime {

extern __global__ void g_updateScalarField(float *colorFieldDevice,
                                           Particle *d_particles, int gridSize);

extern __global__ void g_computeDensity(Particle *d_particles,
                                        unsigned int *hashIndices,
                                        unsigned int *bucketStart,
                                        unsigned int *bucketEnd);

extern __global__ void g_computePressure(Particle *d_particles);

extern __global__ void g_computePressureForce(Particle *d_particles,
                                              unsigned int *hashIndices,
                                              unsigned int *bucketStart,
                                              unsigned int *bucketEnd,
                                              double deltaTime);

extern __global__ void g_computeViscosityForce(Particle *d_particles,
                                               unsigned int *hashIndices,
                                               unsigned int *bucketStart,
                                               unsigned int *bucketEnd,
                                               double deltaTime);

extern __global__ void g_computeSurfaceTension(Particle *d_particles,
                                               unsigned int *hashIndices,
                                               unsigned int *bucketStart,
                                               unsigned int *bucketEnd,
                                               double deltaTime);

extern __global__ void g_computeGravity(Particle *d_particles,
                                        double deltaTime);

extern __global__ void g_computeWallConstraint(Particle *d_particles,
                                               double deltaTime);

extern __global__ void g_computePosition(Particle *d_particles,
                                         double deltaTime);

extern __global__ void g_updateSpatialHash(Particle *d_particles,
                                           unsigned int *hashKeys,
                                           unsigned int *hashIndices);
extern __global__ void g_updateHashBucket(unsigned int *hashKeys,
                                          unsigned int *hashIndices,
                                          unsigned int *bucketStart,
                                          unsigned int *bucketEnd);

extern __global__ void g_copyPositionToVBO(float *d_positions,
                                           Particle *d_particles);
} // namespace slime

#endif