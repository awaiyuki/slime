#ifndef SPH_SIMULATOR_CONSTANTS_H
#define SPH_SIMULATOR_CONSTANTS_H
namespace slime::SPHSimulatorConstants {
constexpr int NUM_PARTICLES = 20000; // spatial hash 이후 10^6개 정도로
constexpr float PARTICLE_MASS = 0.1f;
constexpr float GAS_CONSTANT = 0.0001f;
constexpr float REST_DENSITY = 1.0f;
constexpr float VISCOSITY_COEFFICIENT =
    0.1f; // 어째서인지 참조를 못해서 sph_simulator_device.cu 코드 내에서 magic
          // number로 넣음.
constexpr float SMOOTHING_RADIUS = 1.0f; // 현재 grid_size로 결정됨
constexpr float SURFACE_LEVEL = 0.5f;
constexpr int GRID_SIZE =
    50; // for both spatial hashing and marching cubes,
        // cuda-opengl interop 후 200으로(smoothing radius 달라지는 것 고려.)
}; // namespace slime::SPHSimulatorConstants

#endif