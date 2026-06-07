#ifndef PORTABLE_SIMULATION_CONSTANTS_H
#define PORTABLE_SIMULATION_CONSTANTS_H

#include <cstdint>

namespace slime::PortableSimulationConstants {

inline constexpr std::uint32_t particlesPerAxis = 16;
inline constexpr std::uint32_t particleCount =
    particlesPerAxis * particlesPerAxis * particlesPerAxis;
inline constexpr float initialWidth = 0.64f;
inline constexpr float spacing = initialWidth / particlesPerAxis;
inline constexpr float smoothingRadius = 0.10f;
inline constexpr float mass =
    initialWidth * initialWidth * initialWidth / particleCount;
inline constexpr float restDensity = 1.0f;
inline constexpr float pressureStiffness = 45.0f;
inline constexpr float viscosity = 0.12f;
inline constexpr float gravity = -3.0f;
inline constexpr float maxAcceleration = 35.0f;
inline constexpr float maxSpeed = 2.5f;
inline constexpr float boundary = 0.96f;
inline constexpr float collisionDamping = 0.25f;
inline constexpr float velocityDamping = 0.999f;
inline constexpr float maxTimeStep = 1.0f / 60.0f;

} // namespace slime::PortableSimulationConstants

#endif
