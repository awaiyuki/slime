#ifndef SPH_SIMULATOR_CONSTANTS_H
#define SPH_SIMULATOR_CONSTANTS_H

namespace slime::SPHSimulatorConstants {

inline constexpr int NUM_PARTICLES = 50000;
inline constexpr int THREAD_SIZE_IN_UPDATE_PARTICLES = 512;
inline constexpr int THREAD_SIZE_IN_UPDATE_SCALAR_FIELD = 8;
inline constexpr int GRID_SIZE = 20;

inline __device__ constexpr int MAX_NEIGHBORS = 100;
inline __device__ constexpr float PARTICLE_MASS = 1.0f;
inline __device__ constexpr float REST_DENSITY = 1.0f;
inline __device__ constexpr float GAS_CONSTANT = 1e-5f;
inline __device__ constexpr float VISCOSITY_COEFFICIENT = 0.01f;
inline __device__ constexpr float GRAVITATIONAL_ACCELERATION = -1.0f;
inline __device__ constexpr float SMOOTHING_RADIUS = 1.0f;
inline __device__ constexpr float SURFACE_LEVEL = 0.5f;
inline __device__ constexpr float SURFACE_TENSION_COEFFICIENT = 0.072f;
inline __device__ constexpr float SURFACE_NORMAL_THRESHOLD = 0.1f;
inline __device__ constexpr float EPS_CURVATURE = 1e-5f;
}; // namespace slime::SPHSimulatorConstants

#endif