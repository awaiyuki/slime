
#include "sph_simulator.cuh"
#include "sph_simulator_device.cuh"
#include <cstring>
#include <random>
#include <iostream>
#include <stdio.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

using namespace slime;
using namespace slime::SPHSimulatorConstants;
using namespace std;

SPHSimulator::SPHSimulator(const unsigned int vbo)
    : hashKeys(SPHSimulatorConstants::NUM_PARTICLES, 1),
      hashIndices(SPHSimulatorConstants::NUM_PARTICLES, 1) { // correct?
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(-0.1f, 0.1f);

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

  memset(scalarField, 0, sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);

  cudaMalloc((void **)&d_particles,
             sizeof(Particle) * SPHSimulatorConstants::NUM_PARTICLES);
  cudaMalloc((void **)&d_scalarField,
             sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);

  cudaMemcpy(d_particles, particles.data(),
             sizeof(Particle) * SPHSimulatorConstants::NUM_PARTICLES,
             cudaMemcpyHostToDevice);
  cudaGraphicsGLRegisterBuffer(&cudaVBOResource, vbo, cudaGraphicsMapFlagsNone);

  raw_hashKeys = thrust::raw_pointer_cast(hashKeys.data());
  raw_hashIndices = thrust::raw_pointer_cast(hashIndices.data());
}

SPHSimulator::~SPHSimulator() {
  cudaFree(d_particles);
  cudaFree(d_scalarField);
}

std::vector<Particle> *SPHSimulator::getParticlesPointer() {
  return &particles;
}

void SPHSimulator::updateParticles(double deltaTime) {

  const int threadSize = 128;
  const int blockSize =
      (SPHSimulatorConstants::NUM_PARTICLES + threadSize - 1) / threadSize;

  cout << "check1" << endl;
  updateSpatialHashDevice<<<blockSize, threadSize>>>(d_particles, raw_hashKeys,
                                                     raw_hashIndices);
  cudaDeviceSynchronize();

  cout << "check2" << endl;
  thrust::sort_by_key(hashKeys.begin(), hashKeys.end(), hashIndices.begin());

  raw_hashKeys = thrust::raw_pointer_cast(hashKeys.data());
  raw_hashIndices = thrust::raw_pointer_cast(hashIndices.data());
  cout << "check3" << endl;
  updateHashBucketDevice<<<blockSize, threadSize>>>(raw_hashKeys,
                                                    raw_hashIndices);
  cudaDeviceSynchronize();

  computeDensityDevice<<<blockSize, threadSize>>>(d_particles);
  cudaDeviceSynchronize();

  computePressureDevice<<<blockSize, threadSize>>>(d_particles);

  computePressureForceDevice<<<blockSize, threadSize>>>(d_particles, deltaTime);
  cudaDeviceSynchronize();

  computeViscosityForceDevice<<<blockSize, threadSize>>>(d_particles,
                                                         deltaTime);
  cudaDeviceSynchronize();

  computeSurfaceTensionDevice<<<blockSize, threadSize>>>(d_particles,
                                                         deltaTime);
  cudaDeviceSynchronize();

  /*
  computeSurfaceTensionForce<<<blockSize, threadSize>>>(d_particles,
                                                         deltaTime);
  cudaDeviceSynchronize();
  */

  computeGravityDevice<<<blockSize, threadSize>>>(d_particles, deltaTime);
  cudaDeviceSynchronize();

  computePositionDevice<<<blockSize, threadSize>>>(d_particles, deltaTime);
  cudaDeviceSynchronize();

  computeWallConstraintDevice<<<blockSize, threadSize>>>(d_particles,
                                                         deltaTime);
  cudaDeviceSynchronize();

  cout << "check4" << endl;
  /* Copy Particle Positions to VBO positions array */
  cudaGraphicsMapResources(1, &cudaVBOResource, 0);
  float *d_positions;
  size_t size;
  cudaGraphicsResourceGetMappedPointer((void **)&d_positions, &size,
                                       cudaVBOResource);
  copyPositionToVBODevice<<<blockSize, threadSize>>>(d_positions, d_particles);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("cuda kernel error: %s\n", cudaGetErrorString(err));
  }

  cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);
}

void SPHSimulator::updateScalarField() {
  /* Need to debug */

  const int threadSize = 8;
  dim3 dimBlock(threadSize, threadSize, threadSize);
  const int blockSize = (GRID_SIZE + threadSize - 1) / threadSize;
  dim3 dimGrid(blockSize, blockSize, blockSize);

  updateScalarFieldDevice<<<dimGrid, dimBlock>>>(
      d_scalarField, d_particles, GRID_SIZE,
      1954.0); // need to investigate normalization methods.

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("updated_scalarField error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
}

VertexData SPHSimulator::extractSurface() {
  return marchingCubes->march(d_scalarField,
                              SPHSimulatorConstants::SURFACE_LEVEL);
}
