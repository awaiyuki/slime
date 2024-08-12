
#include "sph_simulator.cuh"
#include "sph_simulator_kernel.cuh"
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
  updateParticlesDevice<<<1, SPHSimulatorConstants::NUM_PARTICLES>>>(
      particlesDevice, deltaTime);
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

float SPHSimulator::poly6Kernel(glm::vec3 r, float h) {
  float rMagnitude = glm::length(r);
  if (rMagnitude > h)
    return 0.0f;

  return 315.0f / (64.0f * PI * glm::pow(h, 9)) *
         glm::pow(h * h - rMagnitude * rMagnitude, 3);
}

float SPHSimulator::spikyKernel(glm::vec3 r, float h) { return 0.0f; }

float SPHSimulator::gradientSpikyKernel(glm::vec3 r, float h) {
  float rMagnitude = glm::length(r);
  if (rMagnitude > h)
    return 0.0f;

  return -45.0f / (PI * glm::pow(h, 6)) * glm::pow(h - rMagnitude, 2);
}

float SPHSimulator::viscosityKernel(glm::vec3 r, float h) { return 0.0f; }

float SPHSimulator::laplacianViscosityKernel(glm::vec3 r, float h) {
  float rMagnitude = glm::length(r);
  if (rMagnitude > h)
    return 0.0f;

  return 45 / (PI * glm::pow(h, 6)) * (h - rMagnitude);
}

void SPHSimulator::computeDensity() {
  for (auto &i : particles) {
    i.density = 0.0f;
    for (auto &j : particles) {
      if (i == j)
        continue;

      auto r = j.position - i.position;
      i.density +=
          j.mass * poly6Kernel(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
    }
  }
}

void SPHSimulator::computePressureForce(double deltaTime) {
  for (auto &i : particles) {
    i.pressure = SPHSimulatorConstants::GAS_CONSTANT *
                 (i.density - SPHSimulatorConstants::REST_DENSITY);
  }

  for (auto &i : particles) {
    glm::vec3 pressureForce = glm::vec3(0.0f, 0.0f, 0.0f);
    for (auto &j : particles) {
      if (i == j)
        continue;

      if (j.density < EPSILON)
        continue;

      auto r = j.position - i.position;
      pressureForce +=
          -glm::normalize(r) * j.mass * (i.pressure + j.pressure) /
          (2.0f * j.density) *
          gradientSpikyKernel(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
    }
    auto acceleration = pressureForce / i.mass;
    auto deltaVelocity = acceleration * float(deltaTime);
    i.velocity += deltaVelocity;
  }
}

void SPHSimulator::computeViscosityForce(double deltaTime) {
  for (auto &i : particles) {
    glm::vec3 viscosityForce = glm::vec3(0.0f, 0.0f, 0.0f);
    for (auto &j : particles) {
      if (i == j)
        continue;

      if (j.density < EPSILON)
        continue;

      auto r = j.position - i.position;
      viscosityForce +=
          j.mass * (j.velocity - i.velocity) / j.density *
          laplacianViscosityKernel(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
    }
    viscosityForce *= SPHSimulatorConstants::VISCOSITY_COEFFICIENT;

    auto acceleration = viscosityForce / i.mass;
    auto deltaVelocity = acceleration * float(deltaTime);
    i.velocity += deltaVelocity;
  }
}

void SPHSimulator::computeGravity(double deltaTime) {
  for (auto &i : particles) {
    auto acceleration = glm::vec3(0, -0.098f, 0);
    auto deltaVelocity = acceleration * float(deltaTime);
    i.velocity += deltaVelocity;
  }
}

void SPHSimulator::computeWallConstraint(double deltaTime) {

  /* Spring-Damper Collision */

  for (auto &i : particles) {
    const float FLOOR_CONSTRAINT = -3.0f;
    const float CEILING_CONSTRAINT = 3.0f;
    const float SPRING_CONSTANT = 500.0f;
    const float DAMPING = 1.0f;
    if (i.position.x < FLOOR_CONSTRAINT) {
      auto deltaVelocity =
          (SPRING_CONSTANT * (FLOOR_CONSTRAINT - i.position.x) +
           DAMPING * i.velocity.x) *
          float(deltaTime);
      i.velocity.x += deltaVelocity;
    }

    if (i.position.x > CEILING_CONSTRAINT) {
      auto deltaVelocity =
          (SPRING_CONSTANT * (i.position.x - CEILING_CONSTRAINT) +
           DAMPING * i.velocity.x) *
          float(deltaTime);
      i.velocity.x -= deltaVelocity;
    }
    if (i.position.y < FLOOR_CONSTRAINT) {
      auto deltaVelocity =
          (SPRING_CONSTANT * (FLOOR_CONSTRAINT - i.position.y) +
           DAMPING * i.velocity.y) *
          float(deltaTime);
      i.velocity.y += deltaVelocity;
    }

    if (i.position.y > CEILING_CONSTRAINT) {
      auto deltaVelocity =
          (SPRING_CONSTANT * (i.position.y - CEILING_CONSTRAINT) +
           DAMPING * i.velocity.y) *
          float(deltaTime);
      i.velocity.y -= deltaVelocity;
    }
    if (i.position.z < FLOOR_CONSTRAINT) {
      auto deltaVelocity =
          (SPRING_CONSTANT * (FLOOR_CONSTRAINT - i.position.z) +
           DAMPING * i.velocity.z) *
          float(deltaTime);
      i.velocity.z += deltaVelocity;
    }

    if (i.position.z > CEILING_CONSTRAINT) {
      auto deltaVelocity =
          (SPRING_CONSTANT * (i.position.z - CEILING_CONSTRAINT) +
           DAMPING * i.velocity.z) *
          float(deltaTime);
      i.velocity.z -= deltaVelocity;
    }
  }
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