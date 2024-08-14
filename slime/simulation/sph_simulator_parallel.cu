#include "sph_simulator_parallel.cuh"
#include <slime/constants/sph_simulator_constants.h>
#include <math.h>

#define PI 3.141592653589793238462643
#define EPSILON 0.000001

using namespace slime;

__device__ __host__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __host__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __host__ float3 operator*(const float& a, const float3& b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}
__device__ __host__ float3 operator*( const float3& b, const float& a)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}
__device__ __host__ float3 operator/(const float3& b, const float& a)
{
    return make_float3(b.x/a, b.y/a, b.z/a);
}
__device__ __host__ inline float3& operator*=(float3& a, const float& s)
{
    a.x *= s;
    a.y *= s;
    a.z *= s;
    return a;
}
__device__ __host__ inline float3& operator+=(float3& a, const float& s)
{
    a.x += s;
    a.y += s;
    a.z += s;
    return a;
}
__device__ __host__ inline float3& operator+=(float3& a, float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
__device__ __host__ inline float length(float3& a)
{
    return sqrt(a.x * a.x * +a.y * a.y + a.z * a.z);
}
__device__ __host__ inline float3& normalize(float3& a)
{
    return a / length(a);
}


__device__ float slime::poly6KernelDevice(float3 r, float h) {
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
                                               int gridSize) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x >= gridSize || y >= gridSize || z >= gridSize)
    return;

  float colorQuantity = 0.0f;
  for (int j = 0; j < SPHSimulatorConstants::NUM_PARTICLES; j++) {
    float3 r =
        make_float3(static_cast<float>(x) / static_cast<float>(gridSize),
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
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES) return;
  auto &i = particlesDevice[idx];

  i.density = 0.0f;
  for (int idx_j = 0; idx_j < SPHSimulatorConstants::NUM_PARTICLES; idx_j++) {
    auto &j = particlesDevice[idx_j];

    if (i == j)
      continue;

    auto r = j.position - i.position;
    i.density +=
        j.mass * poly6KernelDevice(r, SPHSimulatorConstants::SMOOTHING_RADIUS);

  }
}

__global__ void slime::computePressureDevice(Particle *particlesDevice) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES) return;
  auto &i = particlesDevice[idx];
  i.pressure = SPHSimulatorConstants::GAS_CONSTANT *
               (i.density - SPHSimulatorConstants::REST_DENSITY);
}
__global__ void slime::computePressureForceDevice(Particle *particlesDevice,
                                                  double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES) return;
  auto &i = particlesDevice[idx];

  float3 pressureForce = make_float3(0.0f, 0.0f, 0.0f);
  for (int idx_j = 0; idx_j < SPHSimulatorConstants::NUM_PARTICLES; idx_j++) {
    auto &j = particlesDevice[idx_j];

    if (i == j)
      continue;

    if (j.density < EPSILON)
      continue;

    auto r = j.position - i.position;
    pressureForce +=
        (-1)* normalize(r)* j.mass* (i.pressure + j.pressure) /
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
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES) return;
  auto &i = particlesDevice[idx];
  float3 viscosityForce = make_float3(0.0f, 0.0f, 0.0f);
  for (int idx_j = 0; idx_j < SPHSimulatorConstants::NUM_PARTICLES; idx_j++) {
    auto &j = particlesDevice[idx_j];
    if (i == j)
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

__global__ void slime::computeGravityDevice(Particle *particlesDevice,
                                            double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES) return;
  auto &i = particlesDevice[idx];
  auto acceleration = make_float3(0, -0.098f, 0);
  auto deltaVelocity = acceleration * float(deltaTime);
  i.velocity += deltaVelocity;
}

__global__ void slime::computeWallConstraintDevice(Particle *particlesDevice,
                                                   double deltaTime) {

  /* Spring-Damper Collision */

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= SPHSimulatorConstants::NUM_PARTICLES) return;
  auto &i = particlesDevice[idx];
  const float FLOOR_CONSTRAINT = -5.0f;
  const float CEILING_CONSTRAINT = 5.0f;
  const float SPRING_CONSTANT = 500.0f;
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

__global__ void slime::computePositionParallel(Particle* particlesDevice,
    double deltaTime) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= SPHSimulatorConstants::NUM_PARTICLES) return;
    auto& i = particlesDevice[idx];
    i.position += i.velocity * static_cast<float>(deltaTime);
}