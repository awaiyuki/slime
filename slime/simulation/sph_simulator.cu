
#include "sph_simulator.cuh"
#include "sph_simulator_parallel.cuh"
#include <cstring>
#include <random>
#include <iostream>

using namespace slime;
using namespace std;

SPHSimulator::SPHSimulator() {
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
}

SPHSimulator::~SPHSimulator() {
  cudaFree(particlesDevice);
  cudaFree(colorFieldDevice);
}

void SPHSimulator::updateParticles(double deltaTime) {

  const int threadSize = 128;
  const int blockSize =
      (SPHSimulatorConstants::NUM_PARTICLES + threadSize - 1) / threadSize;
  computeDensityDevice<<<blockSize, threadSize>>>(particlesDevice);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("computeDensity error: %s\n", cudaGetErrorString(err));
  }
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
}

void SPHSimulator::updateScalarField() {

  const int threadSize = 8;
  dim3 dimBlock(threadSize, threadSize, threadSize);
  const int blockSize = (GRID_SIZE + threadSize - 1) / threadSize;
  dim3 dimGrid(blockSize, blockSize, blockSize);
  updateScalarFieldDevice<<<dimGrid, dimBlock>>>(colorFieldDevice,
                                                 particlesDevice, GRID_SIZE);
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

std::vector<float> SPHSimulator::extractParticlePositions() {

  cudaMemcpy(particles.data(), particlesDevice,
             sizeof(Particle) * SPHSimulatorConstants::NUM_PARTICLES,
             cudaMemcpyDeviceToHost);
  vector<float> positions;
  positions.reserve(SPHSimulatorConstants::NUM_PARTICLES * 3);
  for (const auto &i : particles) {
    positions.push_back(i.position.x);
    positions.push_back(i.position.y);
    positions.push_back(i.position.z);
  }
  return positions;
}