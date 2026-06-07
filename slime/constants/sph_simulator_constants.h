#ifndef SPH_SIMULATOR_CONSTANTS_H
#define SPH_SIMULATOR_CONSTANTS_H

namespace slime::SPHSimulatorConstants {

inline constexpr int NUM_PARTICLES = 1000000;
inline constexpr int THREAD_SIZE_IN_UPDATE_PARTICLES = 512;
inline constexpr int THREAD_SIZE_IN_UPDATE_SCALAR_FIELD = 8;
inline constexpr int GRID_SIZE = 20;

inline constexpr float INITIAL_FLUID_WIDTH = 0.4f;
inline constexpr float PARTICLE_MASS =
    INITIAL_FLUID_WIDTH * INITIAL_FLUID_WIDTH * INITIAL_FLUID_WIDTH /
    static_cast<float>(NUM_PARTICLES);

inline __device__ constexpr int MAX_NEIGHBORS = 32;
inline __device__ constexpr float REST_DENSITY = 1.0f;
inline __device__ constexpr float GAS_CONSTANT = 20.0f;
inline __device__ constexpr float VISCOSITY_COEFFICIENT = 0.08f;
inline __device__ constexpr float GRAVITATIONAL_ACCELERATION = -1.0f;
inline __device__ constexpr float SMOOTHING_RADIUS = 0.006f;
inline __device__ constexpr float SURFACE_LEVEL = 0.5f;
inline __device__ constexpr float SURFACE_TENSION_COEFFICIENT = 0.002f;
inline __device__ constexpr float SURFACE_NORMAL_THRESHOLD = 0.1f;
inline __device__ constexpr float EPS_CURVATURE = 1e-5f;
inline __device__ constexpr float MAX_ACCELERATION = 30.0f;
inline __device__ constexpr float MAX_SPEED = 3.0f;
inline __device__ constexpr float VELOCITY_DAMPING = 0.999f;
inline __device__ constexpr float BOUNDARY_LIMIT = 0.98f;
inline __device__ constexpr float COLLISION_DAMPING = 0.35f;
}; // namespace slime::SPHSimulatorConstants

#endif
