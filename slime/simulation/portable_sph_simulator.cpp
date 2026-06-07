#include "portable_sph_simulator.h"

#include <algorithm>
#include <cmath>

#ifndef __APPLE__

namespace {

constexpr int kParticlesPerAxis = 20;
constexpr int kParticleCount =
    kParticlesPerAxis * kParticlesPerAxis * kParticlesPerAxis;
constexpr float kInitialWidth = 0.60f;
constexpr float kSpacing = kInitialWidth / kParticlesPerAxis;
constexpr float kSmoothingRadius = 0.075f;
constexpr float kMass = kInitialWidth * kInitialWidth * kInitialWidth /
                        static_cast<float>(kParticleCount);
constexpr float kRestDensity = 1.0f;
constexpr float kPressureStiffness = 45.0f;
constexpr float kViscosity = 0.12f;
constexpr float kGravity = -3.0f;
constexpr float kMaxAcceleration = 35.0f;
constexpr float kMaxSpeed = 2.5f;
constexpr float kBoundary = 0.96f;
constexpr float kCollisionDamping = 0.25f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kEpsilon = 1e-6f;

float poly6(float distanceSquared) {
  const float h2 = kSmoothingRadius * kSmoothingRadius;
  if (distanceSquared >= h2)
    return 0.0f;
  const float term = h2 - distanceSquared;
  return 315.0f / (64.0f * kPi * std::pow(kSmoothingRadius, 9.0f)) *
         term * term * term;
}

float spikyGradient(float distance) {
  if (distance <= kEpsilon || distance >= kSmoothingRadius)
    return 0.0f;
  const float term = kSmoothingRadius - distance;
  return -45.0f / (kPi * std::pow(kSmoothingRadius, 6.0f)) * term * term;
}

float viscosityLaplacian(float distance) {
  if (distance >= kSmoothingRadius)
    return 0.0f;
  return 45.0f / (kPi * std::pow(kSmoothingRadius, 6.0f)) *
         (kSmoothingRadius - distance);
}

glm::vec3 limitMagnitude(const glm::vec3 &value, float maximum) {
  const float magnitude = glm::length(value);
  if (!std::isfinite(magnitude))
    return glm::vec3(0.0f);
  if (magnitude > maximum)
    return value * (maximum / magnitude);
  return value;
}

} // namespace

namespace slime {

PortableSPHSimulator::PortableSPHSimulator() {
  particles_.reserve(kParticleCount);
  positions_.resize(kParticleCount);
  nextVelocities_.resize(kParticleCount);

  const glm::vec3 minimum(-kInitialWidth * 0.5f, -0.35f,
                          -kInitialWidth * 0.5f);
  for (int z = 0; z < kParticlesPerAxis; ++z) {
    for (int y = 0; y < kParticlesPerAxis; ++y) {
      for (int x = 0; x < kParticlesPerAxis; ++x) {
        PortableParticle particle;
        particle.position =
            minimum + glm::vec3((x + 0.5f) * kSpacing,
                                (y + 0.5f) * kSpacing,
                                (z + 0.5f) * kSpacing);
        positions_[particles_.size()] = particle.position;
        particles_.push_back(particle);
      }
    }
  }
  grid_.reserve(kParticleCount * 2);
}

PortableSPHSimulator::~PortableSPHSimulator() = default;

PortableSPHSimulator::CellKey
PortableSPHSimulator::cellKey(const glm::ivec3 &cell) const {
  constexpr std::int64_t offset = 1 << 20;
  return ((static_cast<std::int64_t>(cell.x) + offset) << 42) ^
         ((static_cast<std::int64_t>(cell.y) + offset) << 21) ^
         (static_cast<std::int64_t>(cell.z) + offset);
}

glm::ivec3 PortableSPHSimulator::cellFor(const glm::vec3 &position) const {
  return glm::ivec3(glm::floor(position / kSmoothingRadius));
}

void PortableSPHSimulator::rebuildGrid() {
  grid_.clear();
  for (std::uint32_t i = 0; i < particles_.size(); ++i)
    grid_[cellKey(cellFor(particles_[i].position))].push_back(i);
}

void PortableSPHSimulator::computeDensities() {
  for (auto &particle : particles_) {
    float density = kMass * poly6(0.0f);
    const glm::ivec3 cell = cellFor(particle.position);

    for (int z = -1; z <= 1; ++z) {
      for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
          const auto bucket = grid_.find(cellKey(cell + glm::ivec3(x, y, z)));
          if (bucket == grid_.end())
            continue;
          for (const auto index : bucket->second) {
            const glm::vec3 offset =
                particle.position - particles_[index].position;
            const float distanceSquared = glm::dot(offset, offset);
            if (distanceSquared > kEpsilon)
              density += kMass * poly6(distanceSquared);
          }
        }
      }
    }

    particle.density = std::max(density, kRestDensity * 0.25f);
    particle.pressure =
        kPressureStiffness * std::max(particle.density - kRestDensity, 0.0f);
  }
}

void PortableSPHSimulator::computeVelocities(float deltaTime) {
  for (std::uint32_t i = 0; i < particles_.size(); ++i) {
    const auto &particle = particles_[i];
    glm::vec3 pressureForce(0.0f);
    glm::vec3 viscosityForce(0.0f);
    const glm::ivec3 cell = cellFor(particle.position);

    for (int z = -1; z <= 1; ++z) {
      for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
          const auto bucket = grid_.find(cellKey(cell + glm::ivec3(x, y, z)));
          if (bucket == grid_.end())
            continue;
          for (const auto index : bucket->second) {
            if (index == i)
              continue;
            const auto &neighbor = particles_[index];
            const glm::vec3 offset = particle.position - neighbor.position;
            const float distance = glm::length(offset);
            if (distance <= kEpsilon || distance >= kSmoothingRadius)
              continue;

            const glm::vec3 direction = offset / distance;
            pressureForce +=
                -direction * kMass *
                (particle.pressure + neighbor.pressure) /
                (2.0f * neighbor.density) * spikyGradient(distance);
            viscosityForce +=
                kMass * (neighbor.velocity - particle.velocity) /
                neighbor.density * viscosityLaplacian(distance);
          }
        }
      }
    }

    glm::vec3 acceleration(0.0f, kGravity, 0.0f);
    acceleration +=
        (pressureForce + kViscosity * viscosityForce) / particle.density;
    acceleration = limitMagnitude(acceleration, kMaxAcceleration);
    nextVelocities_[i] =
        limitMagnitude((particle.velocity + acceleration * deltaTime) * 0.999f,
                       kMaxSpeed);
  }
}

void PortableSPHSimulator::integrate(float deltaTime) {
  for (std::uint32_t i = 0; i < particles_.size(); ++i) {
    auto &particle = particles_[i];
    particle.velocity = nextVelocities_[i];
    particle.position += particle.velocity * deltaTime;

    for (int axis = 0; axis < 3; ++axis) {
      if (particle.position[axis] < -kBoundary) {
        particle.position[axis] = -kBoundary;
        if (particle.velocity[axis] < 0.0f)
          particle.velocity[axis] *= -kCollisionDamping;
      } else if (particle.position[axis] > kBoundary) {
        particle.position[axis] = kBoundary;
        if (particle.velocity[axis] > 0.0f)
          particle.velocity[axis] *= -kCollisionDamping;
      }
    }

    positions_[i] = particle.position;
  }
}

void PortableSPHSimulator::update(float deltaTime) {
  constexpr float maxSubstep = 1.0f / 120.0f;
  const int substeps =
      std::max(1, static_cast<int>(std::ceil(deltaTime / maxSubstep)));
  const float substep = deltaTime / static_cast<float>(substeps);

  for (int step = 0; step < substeps; ++step) {
    rebuildGrid();
    computeDensities();
    computeVelocities(substep);
    integrate(substep);
  }
}

const std::vector<glm::vec3> &PortableSPHSimulator::positions() const {
  return positions_;
}

} // namespace slime

#endif
