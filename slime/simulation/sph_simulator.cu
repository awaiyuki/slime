
#include "sph_simulator.cuh"
#include "sph_simulator_parallel.cuh"
#include <cstring>
#include <cmath>
#include <climits>
#include <iostream>
#include <stdio.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <slime/utility/cuda_debug.cuh>

using namespace slime;
using namespace slime::SPHSimulatorConstants;
using namespace std;

SPHSimulator::SPHSimulator(const unsigned int vbo,
                           const std::string _renderMode)
    : renderMode(_renderMode),
      hashKeys(SPHSimulatorConstants::NUM_PARTICLES, 0),
      hashIndices(SPHSimulatorConstants::NUM_PARTICLES, 0),
      bucketStart(SPHSimulatorConstants::NUM_PARTICLES, -1),
      bucketEnd(SPHSimulatorConstants::NUM_PARTICLES, -1) {
  cout << "in SPHSimulator: renderMode=" << renderMode << endl;
  particles.reserve(SPHSimulatorConstants::NUM_PARTICLES);
  const int particlesPerAxis =
      static_cast<int>(ceil(cbrt(SPHSimulatorConstants::NUM_PARTICLES)));
  const float spacing =
      SPHSimulatorConstants::INITIAL_FLUID_WIDTH / particlesPerAxis;
  const float initialMin = -0.5f * SPHSimulatorConstants::INITIAL_FLUID_WIDTH;

  for (int i = 0; i < SPHSimulatorConstants::NUM_PARTICLES; i++) {
    Particle particle;
    particle.id = i;

    const int ix = i % particlesPerAxis;
    const int iy = (i / particlesPerAxis) % particlesPerAxis;
    const int iz = i / (particlesPerAxis * particlesPerAxis);
    const float x = initialMin + (ix + 0.5f) * spacing;
    const float y = initialMin + (iy + 0.5f) * spacing;
    const float z = initialMin + (iz + 0.5f) * spacing;
    particle.position = make_float3(x, y, z);
    particle.velocity = make_float3(0, 0, 0);
    particle.density = SPHSimulatorConstants::REST_DENSITY;
    particle.pressure = 0.0f;

    // cout << "initial position: " << x << y << z << endl;
    particle.mass = SPHSimulatorConstants::PARTICLE_MASS;
    particles.push_back(particle);
  }

  marchingCubes = make_unique<MarchingCubes>(GRID_SIZE);

  std::vector<float> scalarField(GRID_SIZE * GRID_SIZE * GRID_SIZE, 0);

  cudaMalloc((void **)&d_particles,
             sizeof(Particle) * SPHSimulatorConstants::NUM_PARTICLES);
  cudaMalloc((void **)&d_nextVelocities,
             sizeof(float3) * SPHSimulatorConstants::NUM_PARTICLES);
  cudaMalloc((void **)&d_scalarField,
             sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);

  cudaMemcpy(d_particles, particles.data(),
             sizeof(Particle) * SPHSimulatorConstants::NUM_PARTICLES,
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_scalarField, scalarField.data(),
             sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE,
             cudaMemcpyHostToDevice);
  cudaGraphicsGLRegisterBuffer(&cudaVBOResource, vbo, cudaGraphicsMapFlagsNone);

  raw_hashKeys = thrust::raw_pointer_cast(hashKeys.data());
  raw_hashIndices = thrust::raw_pointer_cast(hashIndices.data());
  raw_bucketStart = thrust::raw_pointer_cast(bucketStart.data());
  raw_bucketEnd = thrust::raw_pointer_cast(bucketEnd.data());
}

SPHSimulator::~SPHSimulator() {
  cudaFree(d_particles);
  cudaFree(d_nextVelocities);
  cudaFree(d_scalarField);
}

std::vector<Particle> *SPHSimulator::getParticlesPointer() {
  return &particles;
}

void SPHSimulator::updateParticles(double deltaTime) {

  const int threadSize = THREAD_SIZE_IN_UPDATE_PARTICLES;
  const int blockSize =
      (SPHSimulatorConstants::NUM_PARTICLES + threadSize - 1) / threadSize;

  /* Updating Spatial Hashing */

  g_updateSpatialHash<<<blockSize, threadSize>>>(d_particles, raw_hashKeys,
                                                 raw_hashIndices);
  cudaDeviceSynchronize();

  printCudaError("updateSpatialHash or before");

  // cout << "check2" << endl;
  thrust::sort_by_key(hashKeys.begin(), hashKeys.end(), hashIndices.begin());
  cudaDeviceSynchronize();

  printCudaError("sortbykey");
  // cout << "check3" << endl;

  raw_hashKeys = thrust::raw_pointer_cast(hashKeys.data());
  raw_hashIndices = thrust::raw_pointer_cast(hashIndices.data());

  thrust::fill(bucketStart.begin(), bucketStart.end(), UINT_MAX);
  thrust::fill(bucketEnd.begin(), bucketEnd.end(), UINT_MAX);
  g_updateHashBucket<<<blockSize, threadSize>>>(raw_hashKeys, raw_hashIndices,
                                                raw_bucketStart, raw_bucketEnd);
  cudaDeviceSynchronize();
  printCudaError("updateHashBucket");
  raw_hashKeys = thrust::raw_pointer_cast(hashKeys.data());
  raw_hashIndices = thrust::raw_pointer_cast(hashIndices.data());
  raw_bucketStart = thrust::raw_pointer_cast(bucketStart.data());
  raw_bucketEnd = thrust::raw_pointer_cast(bucketEnd.data());

  /* Updating Particle attributes */

  g_computeDensity<<<blockSize, threadSize>>>(d_particles, raw_hashIndices,
                                              raw_bucketStart, raw_bucketEnd);
  cudaDeviceSynchronize();
  printCudaError("computeDensity");

  g_computePressure<<<blockSize, threadSize>>>(d_particles);

  g_computeForces<<<blockSize, threadSize>>>(
      d_particles, d_nextVelocities, raw_hashIndices, raw_bucketStart,
      raw_bucketEnd, deltaTime);
  cudaDeviceSynchronize();

  printCudaError("computePressureForce");
  g_applyVelocities<<<blockSize, threadSize>>>(d_particles, d_nextVelocities);
  g_computePosition<<<blockSize, threadSize>>>(d_particles, deltaTime);
  cudaDeviceSynchronize();

  g_computeWallConstraint<<<blockSize, threadSize>>>(d_particles, deltaTime);
  cudaDeviceSynchronize();

  if (this->renderMode == "point") {
    /* Copying Particle Positions to VBO positions array */

    cudaGraphicsMapResources(1, &cudaVBOResource, 0);
    float *d_positions;
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void **)&d_positions, &size,
                                         cudaVBOResource);
    cout << "cudavboresource size: " << size << endl;
    g_copyPositionToVBO<<<blockSize, threadSize>>>(d_positions, d_particles);
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);
  }
}

void SPHSimulator::updateScalarField() {
  /* Need to debug */
  cout << "updateScalarField" << endl;
  const int threadSize = THREAD_SIZE_IN_UPDATE_SCALAR_FIELD;
  dim3 dimBlock(threadSize, threadSize, threadSize);
  const int blockSize = (GRID_SIZE + threadSize - 1) / threadSize;
  dim3 dimGrid(blockSize, blockSize, blockSize);

  g_updateScalarField<<<dimGrid, dimBlock>>>(d_scalarField, d_particles,
                                             GRID_SIZE);

  printCudaError("updateScalarField");
  cudaDeviceSynchronize();
}

void SPHSimulator::extractSurface() {
  marchingCubes->march(cudaVBOResource, d_scalarField,
                       SPHSimulatorConstants::SURFACE_LEVEL);
}
