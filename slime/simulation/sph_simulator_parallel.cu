#include "sph_simulator_parallel.cuh"
#include <slime/constants/sph_simulator_constants.h>
#define PI 3.141592653589793238462643
#define EPSILON 0.000001

using namespace slime;

__global__ float slime::poly6KernelDevice(glm::vec3 r, float h) {
  float rMagnitude = glm::length(r);
  if (rMagnitude > h)
    return 0.0f;

  return 315.0f / (64.0f * PI * glm::pow(h, 9)) *
         glm::pow(h * h - rMagnitude * rMagnitude, 3);
}

__device__ float slime::spikyKernelDevice(glm::vec3 r, float h) { return 0.0f; }

__device__ float slime::gradientSpikyKernelDevice(glm::vec3 r, float h) {
  float rMagnitude = glm::length(r);
  if (rMagnitude > h)
    return 0.0f;

  return -45.0f / (PI * glm::pow(h, 6)) * glm::pow(h - rMagnitude, 2);
}

__device__ float slime::viscosityKernelDevice(glm::vec3 r, float h) {
  return 0.0f;
}

__device__ float slime::laplacianViscosityKernelDevice(glm::vec3 r, float h) {
  float rMagnitude = glm::length(r);
  if (rMagnitude > h)
    return 0.0f;

  return 45 / (PI * glm::pow(h, 6)) * (h - rMagnitude);
}

__global__ void slime::updateScalarFieldDevice(float *colorFieldDevice,
                                               Particle *particlesDevice,
                                               int gridSize) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x >= gridSize || y >= gridSize || z >= gridSize)
    return;

  float colorQuantity = 0.0f;
  for (int j = 0; j < SPHSimulatorConstants::NUM_PARTICLES; j++) {
    glm::vec3 r =
        glm::vec3(static_cast<float>(x) / static_cast<float>(gridSize),
                  static_cast<float>(y) / static_cast<float>(gridSize),
                  static_cast<float>(z) / static_cast<float>(gridSize)) -
        particlesDevice[j].position;
    colorQuantity +=
        particlesDevice[j].mass * (1.0 / particlesDevice[j].density) *
        poly6KernelDevice(r, float(SPHSimulatorConstants::SMOOTHING_RADIUS));
    // cout << "test:"
    //      << poly6Kernel(r, SPHSimulatorConstants::SMOOTHING_RADIUS) /
    //             j.density
    //      << endl;
  }
  // cout << "colorQuantity:" << colorQuantity << endl;
  colorFieldDevice[x * gridSize * gridSize + y * gridSize + z] = colorQuantity;
}

__global__ void slime::computeDensityDevice(Particle *particlesDevice) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  auto &i = particlesDevice[idx];
  i.density = 0.0f;
  for (auto &j : particlesDevice) {
    if (i == j)
      continue;

    auto r = j.position - i.position;
    i.density +=
        j.mass * poly6Kernel(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
  }
}

__global__ void slime::computePressureForceDevice(Particle *particlesDevice,
                                                  double deltaTime) {
  for (auto &i : particlesDevice) {
    i.pressure = SPHSimulatorConstants::GAS_CONSTANT *
                 (i.density - SPHSimulatorConstants::REST_DENSITY);
  }

  for (auto &i : particlesDevice) {
    glm::vec3 pressureForce = glm::vec3(0.0f, 0.0f, 0.0f);
    for (auto &j : particlesDevice) {
      if (i == j)
        continue;

      if (j.density < EPSILON)
        continue;

      auto r = j.position - i.position;
      pressureForce +=
          -glm::normalize(r) * j.mass * (i.pressure + j.pressure) /
          (2.0f * j.density) *
          gradientSpikyKernelDevice(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
    }
    auto acceleration = pressureForce / i.mass;
    auto deltaVelocity = acceleration * float(deltaTime);
    i.velocity += deltaVelocity;
  }
}

__global__ void slime::computeViscosityForceDevice(Particle *particlesDevice,
                                                   double deltaTime) {
  for (auto &i : particlesDevice) {
    glm::vec3 viscosityForce = glm::vec3(0.0f, 0.0f, 0.0f);
    for (auto &j : particlesDevice) {
      if (i == j)
        continue;

      if (j.density < EPSILON)
        continue;

      auto r = j.position - i.position;
      viscosityForce += j.mass * (j.velocity - i.velocity) / j.density *
                        laplacianViscosityKernelDevice(
                            r, SPHSimulatorConstants::SMOOTHING_RADIUS);
    }
    viscosityForce *= SPHSimulatorConstants::VISCOSITY_COEFFICIENT;

    auto acceleration = viscosityForce / i.mass;
    auto deltaVelocity = acceleration * float(deltaTime);
    i.velocity += deltaVelocity;
  }
}

__global__ void slime::computeGravityDevice(Particle *particlesDevice,
                                            double deltaTime) {
  for (auto &i : particlesDevice) {
    auto acceleration = glm::vec3(0, -0.098f, 0);
    auto deltaVelocity = acceleration * float(deltaTime);
    i.velocity += deltaVelocity;
  }
}

__global__ void slime::computeWallConstraintDevice(Particle *particlesDevice,
                                                   double deltaTime) {

  /* Spring-Damper Collision */

  for (auto &i : particlesDevice) {
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