#ifndef SPH_SIMULATOR_CONSTANTS_H
#define SPH_SIMULATOR_CONSTANTS_H
namespace slime::SPHSimulatorConstants {
constexpr int NUM_PARTICLES = 5000; // spatial hash 이후 10^6개 정도로
constexpr float PARTICLE_MASS = 1.0f;
constexpr float GAS_CONSTANT = 0.0001f;
constexpr float REST_DENSITY = 1.0f;
constexpr float VISCOSITY_COEFFICIENT = 0.1f;
constexpr float SMOOTHING_RADIUS = 1.0f; // correct?
constexpr float SURFACE_LEVEL = 0.5f;
constexpr int GRID_SIZE = 50; // for both spatial hashing and marching cubes
}; // namespace slime::SPHSimulatorConstants

#endif