#ifndef SPH_SIMULATOR_CONSTANTS_H
#define SPH_SIMULATOR_CONSTANTS_H

namespace slime::SPHSimulatorConstants {

// Add __host__ __device__ and inline so that device code can reference these
// constants.
inline constexpr int NUM_PARTICLES = 50000;
inline constexpr int THREAD_SIZE_IN_UPDATE_PARTICLES = 512;
inline constexpr int THREAD_SIZE_IN_UPDATE_SCALAR_FIELD = 8;
inline constexpr int GRID_SIZE = 30;

inline __device__ constexpr int MAX_NEIGHBORS = 100;
inline __device__ constexpr float PARTICLE_MASS = 1.0f;
inline __device__ constexpr float REST_DENSITY = 1.0f;
inline __device__ constexpr float GAS_CONSTANT = 1e-20f;
inline __device__ constexpr float VISCOSITY_COEFFICIENT = 0.01f;
inline __device__ constexpr float GRAVITATIONAL_ACCELERATION = -1.0f;
inline __device__ constexpr float SMOOTHING_RADIUS = 1.0f;
inline __device__ constexpr float SURFACE_LEVEL = 0.5f;
}; // namespace slime::SPHSimulatorConstants

#endif