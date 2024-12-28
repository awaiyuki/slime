#include "sph_simulator_device.cuh"
#include <slime/constants/sph_simulator_constants.h>
#include <math.h>
#define PI 3.141592653589793238462643
#define EPSILON 0.000001

using namespace slime;

float slime::poly6Kernel(glm::vec3 r, float h) {
  float rMagnitude = glm::length(r);
  if (rMagnitude > h)
    return 0.0f;

  float result = 315.0f / (64.0f * PI * glm::pow(h, 9)) *
                 glm::pow(h * h - rMagnitude * rMagnitude, 3);
  return result;
}

__host__ __device__ float slime::poly6KernelDevice(float3 r, float h) {
  float rMagnitude = length(r);
  if (rMagnitude > h)
    return 0.0f;

  float result = 315.0f / (64.0f * PI * pow(h, 9)) *
                 pow(h * h - rMagnitude * rMagnitude, 3);
  return result;
}

__device__ float slime::spikyKernelDevice(float3 r, float h) { return 0.0f; }

__device__ float slime::gradientSpikyKernelDevice(float3 r, float h) {
  float rMagnitude = length(r);
  if (rMagnitude > h)
    return 0.0f;

  return -45.0f / (PI * pow(h, 6)) * pow(h - rMagnitude, 2);
}

__device__ float slime::viscosityKernelDevice(float3 r, float h) {
  return 0.0f;
}

__device__ float slime::laplacianViscosityKernelDevice(float3 r, float h) {
  float rMagnitude = length(r);
  if (rMagnitude > h)
    return 0.0f;

  return 45 / (PI * pow(h, 6)) * (h - rMagnitude);
}

__global__ void slime::updateScalarFieldDevice(float *colorFieldDevice,
                                               Particle *particlesDevice,
                                               int gridSize,
                                               float maxColorQuantity) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  // printf("test\n");
  if (x >= gridSize || y >= gridSize || z >= gridSize)
    return;

  float colorQuantity = 0.0f;
  for (int j = 0; j < SPHSimulatorConstants::NUM_PARTICLES; j++) {
    float3 r =
        make_float3((static_cast<float>(x) / static_cast<float>(gridSize)),
                    (static_cast<float>(y) / static_cast<float>(gridSize)),
                    (static_cast<float>(z) / static_cast<float>(gridSize))) -
        particlesDevice[j].position;
    if (particlesDevice[j].density < EPSILON)
      continue;
    colorQuantity += particlesDevice[j].mass *
                     (1.0 / particlesDevice[j].density) *
                     poly6KernelDevice(r, 2.0 / static_cast<float>(gridSize));

    // colorQuantity += particlesDevice[j].density *
    //                  poly6KernelDevice(r, 2.0 /
    //                  static_cast<float>(gridSize));
  }
  // cout << "colorQuantity:" << colorQuantity << endl;
  //  printf("cq: %f\n", colorQuantity / maxColorQuantity);
  colorFieldDevice[z * gridSize * gridSize + y * gridSize + x] =
      colorQuantity / maxColorQuantity;

  /* Test */
  // colorFieldDevice[15 * gridSize * gridSize + 15 * gridSize + 15] = 0.7;
  // colorFieldDevice[15 * gridSize * gridSize + 16 * gridSize + 16] = 0.7;
  // colorFieldDevice[15 * gridSize * gridSize + 15 * gridSize + 16] = 0.7;
  // colorFieldDevice[15 * gridSize * gridSize + 16 * gridSize + 15] = 0.7;
  // colorFieldDevice[16 * gridSize * gridSize + 15 * gridSize + 15] = 0.7;
  // colorFieldDevice[16 * gridSize * gridSize + 16 * gridSize + 15] = 0.7;
  // colorFieldDevice[16 * gridSize * gridSize + 15 * gridSize + 16] = 0.7;
  // colorFieldDevice[16 * gridSize * gridSize + 16 * gridSize + 16] = 0.7;

  // colorFieldDevice[10 * gridSize * gridSize + 10 * gridSize + 10] = 0.7;
  // colorFieldDevice[10 * gridSize * gridSize + 11 * gridSize + 11] = 0.7;
  // colorFieldDevice[10 * gridSize * gridSize + 10 * gridSize + 11] = 0.7;
  // colorFieldDevice[10 * gridSize * gridSize + 11 * gridSize + 10] = 0.7;
  // colorFieldDevice[17 * gridSize * gridSize + 10 * gridSize + 10] = 0.7;
  // colorFieldDevice[17 * gridSize * gridSize + 17 * gridSize + 10] = 0.7;
  // colorFieldDevice[17 * gridSize * gridSize + 10 * gridSize + 17] = 0.7;
  // colorFieldDevice[17 * gridSize * gridSize + 17 * gridSize + 17] = 0.7;
}

__global__ void slime::computeDensityDevice(Particle *particlesDevice) {

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];

  i.density = 0.0f;
  for (int idx_j = 0; idx_j < SPHSimulatorConstants::NUM_PARTICLES; idx_j++) {
    auto &j = particlesDevice[idx_j];

    if (idx == idx_j)
      continue;

    auto r = j.position - i.position;
    i.density +=
        j.mass * poly6KernelDevice(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
  }
}

__global__ void slime::computePressureDevice(Particle *particlesDevice) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];
  i.pressure = SPHSimulatorConstants::GAS_CONSTANT *
               (i.density - SPHSimulatorConstants::REST_DENSITY);
}
__global__ void slime::computePressureForceDevice(Particle *particlesDevice,
                                                  double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];

  float3 pressureForce = make_float3(0.0f, 0.0f, 0.0f);
  for (int idx_j = 0; idx_j < SPHSimulatorConstants::NUM_PARTICLES; idx_j++) {
    auto &j = particlesDevice[idx_j];

    if (idx == idx_j)
      continue;

    if (j.density < EPSILON)
      continue;

    auto r = j.position - i.position;
    pressureForce +=
        (-1) * normalize(r) * j.mass * (i.pressure + j.pressure) /
        (2.0f * j.density) *
        gradientSpikyKernelDevice(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
  }
  auto acceleration = pressureForce / i.mass;
  auto deltaVelocity = acceleration * float(deltaTime);
  i.velocity += deltaVelocity;
}

__global__ void slime::computeViscosityForceDevice(Particle *particlesDevice,
                                                   double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];
  float3 viscosityForce = make_float3(0.0f, 0.0f, 0.0f);
  for (int idx_j = 0; idx_j < SPHSimulatorConstants::NUM_PARTICLES; idx_j++) {
    auto &j = particlesDevice[idx_j];
    if (idx == idx_j)
      continue;

    if (j.density < EPSILON)
      continue;

    auto r = j.position - i.position;
    viscosityForce += j.mass * (j.velocity - i.velocity) / j.density *
                      laplacianViscosityKernelDevice(
                          r, SPHSimulatorConstants::SMOOTHING_RADIUS);
  }
  viscosityForce *= 0.1f; // VISCOSITY_COEFFICIENT

  auto acceleration = viscosityForce / i.mass;
  auto deltaVelocity = acceleration * float(deltaTime);
  i.velocity += deltaVelocity;
}

__global__ void slime::computeSurfaceTensionDevice(Particle *particlesDevice,
                                                   double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];

  // TODO: implement surface tension

  return;
}

__global__ void slime::computeGravityDevice(Particle *particlesDevice,
                                            double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];
  auto acceleration = make_float3(0, -0.098f, 0);
  auto deltaVelocity = acceleration * float(deltaTime);
  i.velocity += deltaVelocity;
}

__global__ void slime::computeWallConstraintDevice(Particle *particlesDevice,
                                                   double deltaTime) {
  /* Handling Collision */
  /* Simulation Space: x, y, z in [-0.5, 0.5] */

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];
  const float FLOOR_CONSTRAINT = 0.0f;
  const float CEILING_CONSTRAINT = 1.0f;
  const float SPRING_CONSTANT = 10.0f;
  const float DAMPING = 1.0f;
  if (i.position.x < FLOOR_CONSTRAINT) {
    auto deltaVelocity = (SPRING_CONSTANT * (FLOOR_CONSTRAINT - i.position.x) +
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
    auto deltaVelocity = (SPRING_CONSTANT * (FLOOR_CONSTRAINT - i.position.y) +
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
    auto deltaVelocity = (SPRING_CONSTANT * (FLOOR_CONSTRAINT - i.position.z) +
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

__global__ void slime::computePositionDevice(Particle *particlesDevice,
                                             double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];
  i.position += i.velocity * static_cast<float>(deltaTime);
}

__global__ void slime::updateSpatialHash(Particle *particlesDevice) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];
}

__global__ void slime::copyPositionToVBO(float *d_positions,
                                         Particle *particlesDevice) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];

  d_positions[3 * idx] = i.position.x;
  d_positions[3 * idx + 1] = i.position.y;
  d_positions[3 * idx + 2] = i.position.z;
}