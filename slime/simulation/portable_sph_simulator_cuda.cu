#include "portable_sph_simulator.h"
#include "portable_simulation_constants.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace slime::PortableSimulationConstants;

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEpsilon = 1e-6f;
constexpr int kThreadsPerBlock = 256;

struct CudaParticle {
  float3 position;
  float3 velocity;
  float density;
  float pressure;
};

void checkCuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
}

__device__ float3 add(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 subtract(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 multiply(float3 value, float scalar) {
  return make_float3(value.x * scalar, value.y * scalar, value.z * scalar);
}

__device__ float3 divide(float3 value, float scalar) {
  return multiply(value, 1.0f / scalar);
}

__device__ float dotProduct(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float magnitude(float3 value) {
  return sqrtf(dotProduct(value, value));
}

__device__ float3 limitMagnitude(float3 value, float maximum) {
  const float length = magnitude(value);
  if (!isfinite(length))
    return make_float3(0.0f, 0.0f, 0.0f);
  return length > maximum ? multiply(value, maximum / length) : value;
}

__device__ float poly6(float distanceSquared) {
  const float h2 = smoothingRadius * smoothingRadius;
  if (distanceSquared >= h2)
    return 0.0f;
  const float term = h2 - distanceSquared;
  return 315.0f / (64.0f * kPi * powf(smoothingRadius, 9.0f)) *
         term * term * term;
}

__device__ float spikyGradient(float distance) {
  if (distance <= kEpsilon || distance >= smoothingRadius)
    return 0.0f;
  const float term = smoothingRadius - distance;
  return -45.0f / (kPi * powf(smoothingRadius, 6.0f)) * term * term;
}

__device__ float viscosityLaplacian(float distance) {
  if (distance >= smoothingRadius)
    return 0.0f;
  return 45.0f / (kPi * powf(smoothingRadius, 6.0f)) *
         (smoothingRadius - distance);
}

__device__ void applyBoundary(float &position, float &velocity) {
  if (position < -boundary) {
    position = -boundary;
    if (velocity < 0.0f)
      velocity *= -collisionDamping;
  } else if (position > boundary) {
    position = boundary;
    if (velocity > 0.0f)
      velocity *= -collisionDamping;
  }
}

__global__ void computeDensity(CudaParticle *particles) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= particleCount)
    return;

  const float3 position = particles[index].position;
  float density = mass * poly6(0.0f);
  for (std::uint32_t j = 0; j < particleCount; ++j) {
    if (j == index)
      continue;
    const float3 offset = subtract(position, particles[j].position);
    density += mass * poly6(dotProduct(offset, offset));
  }

  density = fmaxf(density, restDensity * 0.25f);
  particles[index].density = density;
  particles[index].pressure =
      pressureStiffness * fmaxf(density - restDensity, 0.0f);
}

__global__ void computeForcesAndIntegrate(const CudaParticle *current,
                                          CudaParticle *next,
                                          float deltaTime) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= particleCount)
    return;

  CudaParticle particle = current[index];
  float3 pressureForce = make_float3(0.0f, 0.0f, 0.0f);
  float3 viscosityForce = make_float3(0.0f, 0.0f, 0.0f);
  for (std::uint32_t j = 0; j < particleCount; ++j) {
    if (j == index)
      continue;
    const CudaParticle neighbor = current[j];
    const float3 offset = subtract(particle.position, neighbor.position);
    const float distance = magnitude(offset);
    if (distance <= kEpsilon || distance >= smoothingRadius)
      continue;

    const float3 direction = divide(offset, distance);
    const float pressureScale =
        -mass * (particle.pressure + neighbor.pressure) /
        (2.0f * neighbor.density) * spikyGradient(distance);
    pressureForce = add(pressureForce, multiply(direction, pressureScale));
    const float viscosityScale =
        mass / neighbor.density * viscosityLaplacian(distance);
    viscosityForce =
        add(viscosityForce,
            multiply(subtract(neighbor.velocity, particle.velocity),
                     viscosityScale));
  }

  float3 acceleration = make_float3(0.0f, gravity, 0.0f);
  acceleration =
      add(acceleration,
          divide(add(pressureForce, multiply(viscosityForce, viscosity)),
                 particle.density));
  acceleration = limitMagnitude(acceleration, maxAcceleration);

  float3 velocity =
      limitMagnitude(multiply(add(particle.velocity,
                                  multiply(acceleration, deltaTime)),
                              velocityDamping),
                     maxSpeed);
  float3 position = add(particle.position, multiply(velocity, deltaTime));

  applyBoundary(position.x, velocity.x);
  applyBoundary(position.y, velocity.y);
  applyBoundary(position.z, velocity.z);

  particle.position = position;
  particle.velocity = velocity;
  next[index] = particle;
}

} // namespace

namespace slime {

struct PortableSPHSimulator::CudaState {
  CudaParticle *buffers[2] = {nullptr, nullptr};
  std::uint32_t currentBuffer = 0;
  std::vector<CudaParticle> hostParticles;
};

PortableSPHSimulator::PortableSPHSimulator()
    : positions_(particleCount), cuda_(std::make_unique<CudaState>()) {
  using namespace PortableSimulationConstants;
  cuda_->hostParticles.resize(particleCount);
  const float3 minimum =
      make_float3(-initialWidth * 0.5f, -0.35f, -initialWidth * 0.5f);
  for (std::uint32_t i = 0; i < particleCount; ++i) {
    const std::uint32_t x = i % particlesPerAxis;
    const std::uint32_t y = (i / particlesPerAxis) % particlesPerAxis;
    const std::uint32_t z = i / (particlesPerAxis * particlesPerAxis);
    const float3 position =
        make_float3(minimum.x + (x + 0.5f) * spacing,
                    minimum.y + (y + 0.5f) * spacing,
                    minimum.z + (z + 0.5f) * spacing);
    cuda_->hostParticles[i] = {
        position, make_float3(0.0f, 0.0f, 0.0f), restDensity, 0.0f};
    positions_[i] = glm::vec3(position.x, position.y, position.z);
  }

  const std::size_t bufferSize = sizeof(CudaParticle) * particleCount;
  checkCuda(cudaMalloc(reinterpret_cast<void **>(&cuda_->buffers[0]),
                       bufferSize),
            "cudaMalloc");
  checkCuda(cudaMalloc(reinterpret_cast<void **>(&cuda_->buffers[1]),
                       bufferSize),
            "cudaMalloc");
  checkCuda(cudaMemcpy(cuda_->buffers[0], cuda_->hostParticles.data(),
                       bufferSize, cudaMemcpyHostToDevice),
            "cudaMemcpy");
  checkCuda(cudaMemcpy(cuda_->buffers[1], cuda_->hostParticles.data(),
                       bufferSize, cudaMemcpyHostToDevice),
            "cudaMemcpy");

  cudaDeviceProp properties{};
  checkCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
  std::cout << "CUDA SPH backend: " << properties.name << ", " << particleCount
            << " particles" << std::endl;
}

PortableSPHSimulator::~PortableSPHSimulator() {
  if (cuda_) {
    cudaFree(cuda_->buffers[0]);
    cudaFree(cuda_->buffers[1]);
  }
}

void PortableSPHSimulator::update(float deltaTime) {
  using namespace PortableSimulationConstants;
  const float timeStep = std::min(deltaTime, maxTimeStep);
  const int blocks =
      static_cast<int>((particleCount + kThreadsPerBlock - 1) /
                       kThreadsPerBlock);
  computeDensity<<<blocks, kThreadsPerBlock>>>(
      cuda_->buffers[cuda_->currentBuffer]);
  computeForcesAndIntegrate<<<blocks, kThreadsPerBlock>>>(
      cuda_->buffers[cuda_->currentBuffer],
      cuda_->buffers[1 - cuda_->currentBuffer], timeStep);
  checkCuda(cudaDeviceSynchronize(), "CUDA simulation");

  cuda_->currentBuffer = 1 - cuda_->currentBuffer;
  const std::size_t bufferSize = sizeof(CudaParticle) * particleCount;
  checkCuda(cudaMemcpy(cuda_->hostParticles.data(),
                       cuda_->buffers[cuda_->currentBuffer], bufferSize,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy");
  for (std::uint32_t i = 0; i < particleCount; ++i) {
    const float3 position = cuda_->hostParticles[i].position;
    positions_[i] = glm::vec3(position.x, position.y, position.z);
  }
}

const std::vector<glm::vec3> &PortableSPHSimulator::positions() const {
  return positions_;
}

} // namespace slime
