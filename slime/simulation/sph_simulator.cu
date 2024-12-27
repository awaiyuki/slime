
#include "sph_simulator.cuh"
#include "sph_simulator_device.cuh"
#include <cstring>
#include <random>
#include <iostream>
#include <stdio.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

using namespace slime;
using namespace std;

SPHSimulator::SPHSimulator(const unsigned int vbo) {
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.4f, 0.5f);

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

  memset(colorField, 0, sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);

  cudaMalloc((void **)&particlesDevice,
             sizeof(Particle) * SPHSimulatorConstants::NUM_PARTICLES);
  cudaMalloc((void **)&colorFieldDevice,
             sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);

  cudaMemcpy(particlesDevice, particles.data(),
             sizeof(Particle) * SPHSimulatorConstants::NUM_PARTICLES,
             cudaMemcpyHostToDevice);
  cudaGraphicsGLRegisterBuffer(&cudaVBOResource, vbo, cudaGraphicsMapFlagsNone);
}

SPHSimulator::~SPHSimulator() {
  cudaFree(particlesDevice);
  cudaFree(colorFieldDevice);
}

std::vector<Particle> *SPHSimulator::getParticlesPointer() {
  return &particles;
}

void SPHSimulator::updateParticles(double deltaTime) {

  const int threadSize = 128;
  const int blockSize =
      (SPHSimulatorConstants::NUM_PARTICLES + threadSize - 1) / threadSize;

  computeDensityDevice<<<blockSize, threadSize>>>(particlesDevice);
  cudaDeviceSynchronize();

  computePressureDevice<<<blockSize, threadSize>>>(particlesDevice);

  computePressureForceDevice<<<blockSize, threadSize>>>(particlesDevice,
                                                        deltaTime);
  cudaDeviceSynchronize();

  computeViscosityForceDevice<<<blockSize, threadSize>>>(particlesDevice,
                                                         deltaTime);
  cudaDeviceSynchronize();

  computeSurfaceTensionDevice<<<blockSize, threadSize>>>(particlesDevice,
                                                         deltaTime);
  cudaDeviceSynchronize();

  /*
  computeSurfaceTensionForce<<<blockSize, threadSize>>>(particlesDevice,
                                                         deltaTime);
  cudaDeviceSynchronize();
  */

  computeGravityDevice<<<blockSize, threadSize>>>(particlesDevice, deltaTime);
  cudaDeviceSynchronize();

  computePositionParallel<<<blockSize, threadSize>>>(particlesDevice,
                                                     deltaTime);
  cudaDeviceSynchronize();

  computeWallConstraintDevice<<<blockSize, threadSize>>>(particlesDevice,
                                                         deltaTime);
  cudaDeviceSynchronize();

  /* Copy Particle Positions to VBO positions array */
  cudaGraphicsMapResources(1, &cudaVBOResource, 0);
  float *d_positions;
  size_t size;
  cudaGraphicsResourceGetMappedPointer((void **)&d_positions, &size,
                                       cudaVBOResource);
  copyPositionToVBO<<<blockSize, threadSize>>>(d_positions, particlesDevice);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("cuda kernel error: %s\n", cudaGetErrorString(err));
  }

  cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);
}

void SPHSimulator::updateScalarField() {

  const int threadSize = 8;
  dim3 dimBlock(threadSize, threadSize, threadSize);
  const int blockSize = (GRID_SIZE + threadSize - 1) / threadSize;
  dim3 dimGrid(blockSize, blockSize, blockSize);

  updateScalarFieldDevice<<<dimGrid, dimBlock>>>(
      colorFieldDevice, particlesDevice, GRID_SIZE, 1954.0);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("updateScalarFieldDevice error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
}

VertexData SPHSimulator::extractSurface() {
  return marchingCubes->march(colorFieldDevice,
                              SPHSimulatorConstants::SURFACE_LEVEL);
}

void SPHSimulator::extractParticlePositions(unsigned int vbo) {

  // float *devicePtr = nullptr;
  // size_t numBytes;
  // cudaGraphicsMapResources(1, &cudaVBOResource, 0);
  // cudaGraphicsResourceGetMappedPointer((void **)&devicePtr, &numBytes,
  //                                      cudaVBOResource);

  // cudaMemcpy(devicePtr, particlesDevice,
  //            sizeof(Particle) * SPHSimulatorConstants::NUM_PARTICLES,
  //            cudaMemcpyDeviceToDevice);

  // cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);
  // cudaMemcpy(particles.data(), particlesDevice,
  //            sizeof(Particle) * SPHSimulatorConstants::NUM_PARTICLES,
  //            cudaMemcpyDeviceToHost);
  // vector<float> positions;
  // positions.reserve(SPHSimulatorConstants::NUM_PARTICLES * 3);
  // for (const auto &i : particles) {
  //   positions.push_back(i.position.x);
  //   positions.push_back(i.position.y);
  //   positions.push_back(i.position.z);
  // }
  // return positions;
}