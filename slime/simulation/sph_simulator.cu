
#include "sph_simulator.cuh"
#include "sph_simulator_parallel.cuh"
#include <cstring>
#include <random>
#include <iostream>
#include <stdio.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
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
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(-0.4f, 0.4f); // simulation space: [-1, 1]^3
  particles.reserve(SPHSimulatorConstants::NUM_PARTICLES);
  for (int i = 0; i < SPHSimulatorConstants::NUM_PARTICLES; i++) {
    Particle particle;
    particle.id = i;

    float x = static_cast<float>(dis(gen));
    float y = static_cast<float>(dis(gen));
    float z = static_cast<float>(dis(gen));
    particle.position = make_float3(x, y, z);
    particle.velocity = make_float3(0, 0, 0);

    // cout << "initial position: " << x << y << z << endl;
    particle.mass = SPHSimulatorConstants::PARTICLE_MASS;
    particles.push_back(particle);
  }

  marchingCubes = make_unique<MarchingCubes>(GRID_SIZE);

  std::vector<float> scalarField(GRID_SIZE * GRID_SIZE * GRID_SIZE, 0);

  cudaMalloc((void **)&d_particles,
             sizeof(Particle) * SPHSimulatorConstants::NUM_PARTICLES);
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

  g_computePressureForce<<<blockSize, threadSize>>>(
      d_particles, raw_hashIndices, raw_bucketStart, raw_bucketEnd, deltaTime);
  cudaDeviceSynchronize();

  printCudaError("computePressureForce");
  g_computeViscosityForce<<<blockSize, threadSize>>>(
      d_particles, raw_hashIndices, raw_bucketStart, raw_bucketEnd, deltaTime);
  cudaDeviceSynchronize();
  printCudaError("computeViscosityForce");

  g_computeSurfaceTension<<<blockSize, threadSize>>>(
      d_particles, raw_hashIndices, raw_bucketStart, raw_bucketEnd, deltaTime);
  cudaDeviceSynchronize();
  printCudaError("computeSurfaceTensionForce");

  g_computeGravity<<<blockSize, threadSize>>>(d_particles, deltaTime);
  cudaDeviceSynchronize();

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
