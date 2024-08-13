
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
  uniform_real_distribution<> dis(0.3f, 0.6f);

  for (int i = 0; i < SPHSimulatorConstants::NUM_PARTICLES; i++) {
    Particle particle;
    particle.id = i;

    float x = static_cast<float>(dis(gen));
    float y = static_cast<float>(dis(gen));
    float z = static_cast<float>(dis(gen));
    particle.position = glm::vec3(x, y, z);
    particle.velocity = glm::vec3(0, 0, 0);

    // cout << "initial position: " << x << y << z << endl;
    particle.mass = 1.0f;
    particles.push_back(particle);
  }

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

  computeDensityDevice<<<1, SPHSimulatorConstants::NUM_PARTICLES>>>(
      particlesDevice);
  cudaDeviceSynchronize();

  computePressureDevice<<<1, SPHSimulatorConstants::NUM_PARTICLES>>>(
      particlesDevice);

  computePressureForceDevice<<<1, SPHSimulatorConstants::NUM_PARTICLES>>>(
      particlesDevice, deltaTime);
  cudaDeviceSynchronize();

  computeViscosityForceDevice<<<1, SPHSimulatorConstants::NUM_PARTICLES>>>(
      particlesDevice, deltaTime);
  cudaDeviceSynchronize();

  computeGravityDevice<<<1, SPHSimulatorConstants::NUM_PARTICLES>>>(
      particlesDevice, deltaTime);
  cudaDeviceSynchronize();

  computeWallConstraintDevice<<<1, SPHSimulatorConstants::NUM_PARTICLES>>>(
      particlesDevice, deltaTime);
  cudaDeviceSynchronize();

  computePositionParallel<<<1, SPHSimulatorConstants::NUM_PARTICLES>>>(
      particlesDevice, deltaTime);
  cudaDeviceSynchronize();

  /* Update the positions of particles */
  i.position += i.velocity * static_cast<float>(deltaTime);
}

void SPHSimulator::updateScalarField() {
  dim3 dimBlock(1, 1, 1);
  dim3 dimGrid(GRID_SIZE, GRID_SIZE, GRID_SIZE); // need to be updated
  updateScalarFieldDevice<<<dimBlock, dimGrid>>>(colorFieldDevice,
                                                 particlesDevice, GRID_SIZE);
  cudaDeviceSynchronize();
  cudaMemcpy(colorField, colorFieldDevice,
             sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE,
             cudaMemcpyDeviceToHost);
}

std::vector<MarchingCubes::Triangle> SPHSimulator::extractSurface() {
  MarchingCubes marchingCubes;
  return marchingCubes.march(colorField, SPHSimulatorConstants::SURFACE_LEVEL);
}

std::vector<glm::vec3> SPHSimulator::extractParticlePositions() {
  vector<glm::vec3> positions;
  for (auto &i : particles) {
    positions.push_back(i.position);
  }
  return positions;
}