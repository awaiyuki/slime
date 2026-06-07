#include "portable_sph_simulator.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <simd/simd.h>
#include <stdexcept>

namespace {

constexpr std::uint32_t kParticlesPerAxis = 16;
constexpr std::uint32_t kParticleCount =
    kParticlesPerAxis * kParticlesPerAxis * kParticlesPerAxis;
constexpr float kInitialWidth = 0.64f;
constexpr float kSpacing = kInitialWidth / kParticlesPerAxis;

struct MetalParticle {
  simd_float4 position;
  simd_float4 velocity;
  float density;
  float pressure;
  simd_float2 padding;
};

struct SimulationParameters {
  std::uint32_t particleCount;
  float deltaTime;
  float smoothingRadius;
  float mass;
  float restDensity;
  float pressureStiffness;
  float viscosity;
  float gravity;
  float maxAcceleration;
  float maxSpeed;
  float boundary;
  float collisionDamping;
};

NSString *shaderSource() {
  return @R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float PI = 3.14159265358979323846f;
constant float EPSILON = 1e-6f;

struct Particle {
  float4 position;
  float4 velocity;
  float density;
  float pressure;
  float2 padding;
};

struct Parameters {
  uint particleCount;
  float deltaTime;
  float smoothingRadius;
  float mass;
  float restDensity;
  float pressureStiffness;
  float viscosity;
  float gravity;
  float maxAcceleration;
  float maxSpeed;
  float boundary;
  float collisionDamping;
};

float poly6(float distanceSquared, float h) {
  float h2 = h * h;
  if (distanceSquared >= h2)
    return 0.0f;
  float term = h2 - distanceSquared;
  return 315.0f / (64.0f * PI * pow(h, 9.0f)) * term * term * term;
}

float spikyGradient(float distance, float h) {
  if (distance <= EPSILON || distance >= h)
    return 0.0f;
  float term = h - distance;
  return -45.0f / (PI * pow(h, 6.0f)) * term * term;
}

float viscosityLaplacian(float distance, float h) {
  if (distance >= h)
    return 0.0f;
  return 45.0f / (PI * pow(h, 6.0f)) * (h - distance);
}

float3 limitMagnitude(float3 value, float maximum) {
  float magnitude = length(value);
  if (!isfinite(magnitude))
    return float3(0.0f);
  return magnitude > maximum ? value * (maximum / magnitude) : value;
}

kernel void computeDensity(
    device Particle *particles [[buffer(0)]],
    constant Parameters &params [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.particleCount)
    return;

  float3 position = particles[index].position.xyz;
  float density = params.mass * poly6(0.0f, params.smoothingRadius);
  for (uint j = 0; j < params.particleCount; ++j) {
    if (j == index)
      continue;
    float3 offset = position - particles[j].position.xyz;
    density += params.mass * poly6(dot(offset, offset), params.smoothingRadius);
  }

  density = max(density, params.restDensity * 0.25f);
  particles[index].density = density;
  particles[index].pressure =
      params.pressureStiffness * max(density - params.restDensity, 0.0f);
}

kernel void computeForcesAndIntegrate(
    device const Particle *current [[buffer(0)]],
    device Particle *next [[buffer(1)]],
    constant Parameters &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.particleCount)
    return;

  Particle particle = current[index];
  float3 pressureForce = float3(0.0f);
  float3 viscosityForce = float3(0.0f);

  for (uint j = 0; j < params.particleCount; ++j) {
    if (j == index)
      continue;
    Particle neighbor = current[j];
    float3 offset = particle.position.xyz - neighbor.position.xyz;
    float distance = length(offset);
    if (distance <= EPSILON || distance >= params.smoothingRadius)
      continue;

    float3 direction = offset / distance;
    pressureForce +=
        -direction * params.mass *
        (particle.pressure + neighbor.pressure) /
        (2.0f * neighbor.density) *
        spikyGradient(distance, params.smoothingRadius);
    viscosityForce +=
        params.mass * (neighbor.velocity.xyz - particle.velocity.xyz) /
        neighbor.density * viscosityLaplacian(distance, params.smoothingRadius);
  }

  float3 acceleration = float3(0.0f, params.gravity, 0.0f);
  acceleration +=
      (pressureForce + params.viscosity * viscosityForce) / particle.density;
  acceleration = limitMagnitude(acceleration, params.maxAcceleration);

  float3 velocity =
      limitMagnitude((particle.velocity.xyz + acceleration * params.deltaTime) *
                         0.999f,
                     params.maxSpeed);
  float3 position = particle.position.xyz + velocity * params.deltaTime;

  for (uint axis = 0; axis < 3; ++axis) {
    if (position[axis] < -params.boundary) {
      position[axis] = -params.boundary;
      if (velocity[axis] < 0.0f)
        velocity[axis] *= -params.collisionDamping;
    } else if (position[axis] > params.boundary) {
      position[axis] = params.boundary;
      if (velocity[axis] > 0.0f)
        velocity[axis] *= -params.collisionDamping;
    }
  }

  particle.position = float4(position, 1.0f);
  particle.velocity = float4(velocity, 0.0f);
  next[index] = particle;
}
)METAL";
}

} // namespace

namespace slime {

struct PortableSPHSimulator::MetalState {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> commandQueue = nil;
  id<MTLComputePipelineState> densityPipeline = nil;
  id<MTLComputePipelineState> forcePipeline = nil;
  id<MTLBuffer> buffers[2] = {nil, nil};
  std::uint32_t currentBuffer = 0;
};

PortableSPHSimulator::PortableSPHSimulator()
    : positions_(kParticleCount), metal_(std::make_unique<MetalState>()) {
  metal_->device = MTLCreateSystemDefaultDevice();
  if (!metal_->device)
    throw std::runtime_error("Metal is unavailable on this Mac");

  NSError *error = nil;
  id<MTLLibrary> library =
      [metal_->device newLibraryWithSource:shaderSource() options:nil error:&error];
  if (!library)
    throw std::runtime_error([[error localizedDescription] UTF8String]);

  metal_->densityPipeline = [metal_->device
      newComputePipelineStateWithFunction:[library
                                             newFunctionWithName:@"computeDensity"]
                                    error:&error];
  metal_->forcePipeline = [metal_->device
      newComputePipelineStateWithFunction:
          [library newFunctionWithName:@"computeForcesAndIntegrate"]
                                    error:&error];
  if (!metal_->densityPipeline || !metal_->forcePipeline)
    throw std::runtime_error([[error localizedDescription] UTF8String]);

  metal_->commandQueue = [metal_->device newCommandQueue];
  const NSUInteger bufferSize = sizeof(MetalParticle) * kParticleCount;
  metal_->buffers[0] =
      [metal_->device newBufferWithLength:bufferSize
                                  options:MTLResourceStorageModeShared];
  metal_->buffers[1] =
      [metal_->device newBufferWithLength:bufferSize
                                  options:MTLResourceStorageModeShared];

  auto *particles = static_cast<MetalParticle *>([metal_->buffers[0] contents]);
  const simd_float3 minimum = {-kInitialWidth * 0.5f, -0.35f,
                               -kInitialWidth * 0.5f};
  for (std::uint32_t i = 0; i < kParticleCount; ++i) {
    const std::uint32_t x = i % kParticlesPerAxis;
    const std::uint32_t y = (i / kParticlesPerAxis) % kParticlesPerAxis;
    const std::uint32_t z = i / (kParticlesPerAxis * kParticlesPerAxis);
    const simd_float3 position =
        minimum + simd_make_float3((x + 0.5f) * kSpacing,
                                   (y + 0.5f) * kSpacing,
                                   (z + 0.5f) * kSpacing);
    particles[i] = {simd_make_float4(position, 1.0f),
                    simd_make_float4(0.0f), 1.0f, 0.0f, simd_make_float2(0.0f)};
    positions_[i] = glm::vec3(position.x, position.y, position.z);
  }
  std::memcpy([metal_->buffers[1] contents], particles, bufferSize);

  std::cout << "Metal SPH backend: "
            << [[metal_->device name] UTF8String] << ", " << kParticleCount
            << " particles" << std::endl;
}

PortableSPHSimulator::~PortableSPHSimulator() = default;

void PortableSPHSimulator::update(float deltaTime) {
  SimulationParameters params{
      kParticleCount, std::min(deltaTime, 1.0f / 60.0f), 0.1f,
      kInitialWidth * kInitialWidth * kInitialWidth / kParticleCount,
      1.0f, 45.0f, 0.12f, -3.0f, 35.0f, 2.5f, 0.96f, 0.25f};

  id<MTLCommandBuffer> commandBuffer = [metal_->commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  const MTLSize threadsPerGroup =
      MTLSizeMake(std::min<NSUInteger>(256, metal_->densityPipeline
                                               .maxTotalThreadsPerThreadgroup),
                  1, 1);
  const MTLSize threads = MTLSizeMake(kParticleCount, 1, 1);

  [encoder setComputePipelineState:metal_->densityPipeline];
  [encoder setBuffer:metal_->buffers[metal_->currentBuffer] offset:0 atIndex:0];
  [encoder setBytes:&params length:sizeof(params) atIndex:1];
  [encoder dispatchThreads:threads threadsPerThreadgroup:threadsPerGroup];

  [encoder setComputePipelineState:metal_->forcePipeline];
  [encoder setBuffer:metal_->buffers[metal_->currentBuffer] offset:0 atIndex:0];
  [encoder setBuffer:metal_->buffers[1 - metal_->currentBuffer]
                offset:0
              atIndex:1];
  [encoder setBytes:&params length:sizeof(params) atIndex:2];
  [encoder dispatchThreads:threads threadsPerThreadgroup:threadsPerGroup];
  [encoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  metal_->currentBuffer = 1 - metal_->currentBuffer;
  const auto *particles = static_cast<const MetalParticle *>(
      [metal_->buffers[metal_->currentBuffer] contents]);
  for (std::uint32_t i = 0; i < kParticleCount; ++i)
    positions_[i] = glm::vec3(particles[i].position.x, particles[i].position.y,
                              particles[i].position.z);
}

const std::vector<glm::vec3> &PortableSPHSimulator::positions() const {
  return positions_;
}

} // namespace slime
