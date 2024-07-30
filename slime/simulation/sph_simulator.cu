
#include "sph_simulator.cuh"
#include <cstring>
#include <random>
#include <iostream>
#include <glm/gtc/constants.hpp>

#define PI 3.141592653589793238462643
#define EPSILON 0.000001

using namespace slime;
using namespace std;

SPHSimulator::SPHSimulator() {
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0f, 1.0f);

  for (int i = 0; i < SPHSimulatorConstants::NUM_PARTICLES; i++) {
    Particle particle;

    float x = static_cast<float>(dis(gen));
    float y = static_cast<float>(dis(gen));
    float z = static_cast<float>(dis(gen));
    particle.position = glm::vec3(x, y, z);

    // cout << "initial position: " << x << y << z << endl;
    particle.mass = 1.0f;
    particles.push_back(particle);
  }
}

SPHSimulator::~SPHSimulator() {}

__device__ float SPHSimulator::poly6Kernel(glm::vec3 r, float h) {
  float rMagnitude = glm::length(r);
  if (rMagnitude > h)
    return 0.0f;

  return 315.0f / (64.0f * PI * glm::pow(h, 9)) *
         glm::pow(h * h - rMagnitude * rMagnitude, 3);
}

__device__ float SPHSimulator::poly6KernelDevice(glm::vec3 r, float h) {
  float rMagnitude = glm::length(r);
  if (rMagnitude > h)
    return 0.0f;

  return 315.0f / (64.0f * PI * glm::pow(h, 9)) *
         glm::pow(h * h - rMagnitude * rMagnitude, 3);
}

__device__ float SPHSimulator::spikyKernel(glm::vec3 r, float h) {}

__device__ float SPHSimulator::gradientSpikyKernel(glm::vec3 r, float h) {
  float rMagnitude = glm::length(r);
  if (rMagnitude > h)
    return 0.0f;

  return -45.0f / (PI * glm::pow(h, 6)) * glm::pow(h - rMagnitude, 2);
}

__device__ float SPHSimulator::viscosityKernel(glm::vec3 r, float h) {}

__device__ float SPHSimulator::laplacianViscosityKernel(glm::vec3 r, float h) {
  float rMagnitude = glm::length(r);
  if (rMagnitude > h)
    return 0.0f;

  return 45 / (PI * glm::pow(h, 6)) * (h - rMagnitude);
}

__device__ void SPHSimulator::computeDensity() {
  for (auto &i : particles) {
    i->density = 0;
    for (auto &j : particles) {
      if (i == j)
        continue;

      auto r = j->position - i->position;
      i->density +=
          j->mass * poly6Kernel(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
    }
    if (abs(i->density) < EPSILON) {
      i->density = 0.000002;
    }
  }
}

__device__ void SPHSimulator::computePressureForce(double deltaTime) {
  for (auto &i : particles) {
    i->pressure = SPHSimulatorConstants::GAS_CONSTANT *
                  (i->density - SPHSimulatorConstants::REST_DENSITY);
  }

  for (auto &i : particles) {
    glm::vec3 pressureForce = glm::vec3(0.0f, 0.0f, 0.0f);
    for (auto &j : particles) {
      if (i == j)
        continue;

      auto r = j->position - i->position;
      pressureForce +=
          -glm::normalize(r) * j->mass * (i->pressure + j->pressure) /
          (2.0f * j->density) *
          gradientSpikyKernel(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
    }
    auto acceleration = pressureForce / i->mass;
    auto deltaVelocity = acceleration * float(deltaTime);
    i->velocity += deltaVelocity;
  }
}

__device__ void SPHSimulator::computeViscosityForce(double deltaTime) {
  for (auto &i : particles) {
    glm::vec3 viscosityForce = glm::vec3(0.0f, 0.0f, 0.0f);
    for (auto &j : particles) {
      if (i == j)
        continue;

      auto r = j->position - i->position;
      viscosityForce +=
          j->mass * (j->velocity - i->velocity) / j->density *
          laplacianViscosityKernel(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
    }
    viscosityForce *= SPHSimulatorConstants::VISCOSITY_COEFFICIENT;

    auto acceleration = viscosityForce / i->mass;
    auto deltaVelocity = acceleration * float(deltaTime);
    i->velocity += deltaVelocity;
  }
}

__device__ void SPHSimulator::computeGravity(double deltaTime) {
  for (auto &i : particles) {
    auto acceleration = glm::vec3(0, -0.098f, 0);
    auto deltaVelocity = acceleration * float(deltaTime);
    i->velocity += deltaVelocity;
  }
}

std::vector<MarchingCubes::Triangle> SPHSimulator::extractSurface() {
  MarchingCubes marchingCubes;
  return marchingCubes.march(colorField, SPHSimulatorConstants::SURFACE_LEVEL);
}

__global__ void updateParticles(Particle *particles, int particleCount,
                                double deltaTime) {
  computeDensity();
  computePressureForce(deltaTime);
  computeViscosityForce(deltaTime);
  computeGravity(deltaTime);

  /* Update the positions of particles */
  for (auto &i : particles) {
    i->position += i->velocity * static_cast<float>(deltaTime);
    /* TODO: Keep particles within grid */
    if (i->position.y < 0.0f) {
      i->position.y = 0.001f;
    }
  }
}

__global__ void SPHSimulator::initScalarField(int gridSize) {
  memset(densityField, 0, sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);
  memset(pressureField, 0, sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);
  memset(viscosityField, 0, sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);
  memset(colorField, 0, sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);
  memset(surfaceTensionField, 0,
         sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);
}

__global__ void SPHSimulator::updateScalarField(Particle *particles,
                                                int particleCount,
                                                int gridSize) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  float colorQuantity = 0.0f;
  for (auto &j : particles) {
    glm::vec3 r =
        glm::vec3(static_cast<float>(x) / static_cast<float>(GRID_SIZE),
                  static_cast<float>(y) / static_cast<float>(GRID_SIZE),
                  static_cast<float>(z) / static_cast<float>(GRID_SIZE)) -
        j->position;
    colorQuantity +=
        j->mass * (1.0 / j->density) *
        poly6KernelDevice(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
    // cout << "test:"
    //      << poly6Kernel(r, SPHSimulatorConstants::SMOOTHING_RADIUS) /
    //             j->density
    //      << endl;
  }
  // cout << "colorQuantity:" << colorQuantity << endl;
  colorField[x][y][z] = colorQuantity;
}
