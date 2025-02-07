#ifndef SPH_SIMULATOR_CONSTANTS_H
#define SPH_SIMULATOR_CONSTANTS_H

namespace slime::SPHSimulatorConstants {

constexpr int NUM_PARTICLES = 50000; // 10만에서 100만 정도로
constexpr int THREAD_SIZE_IN_UPDATE_PARTICLES = 512;
constexpr int THREAD_SIZE_IN_UPDATE_SCALAR_FIELD = 8;
constexpr float PARTICLE_MASS = 1.0f;
constexpr float REST_DENSITY = 1.0f;
constexpr float GAS_CONSTANT = 1e-6f;
constexpr float VISCOSITY_COEFFICIENT =
    0.01f; // 어째서인지 참조를 못해서 sph_simulator_device.cu 내에서 magic
// number로 넣음.
constexpr float GRAVITATIONAL_ACCELERATION = -0.2f;
constexpr float SMOOTHING_RADIUS =
    1.0f; // SPH kernels에서의 조건. 현재 spatial hashing 사용 중이므로
          // grid_size로 결정됨
constexpr float SURFACE_LEVEL = 0.5f;
constexpr int MAX_NEIGHBORS =
    100; // for resolving load imbalance in spatial hashing
constexpr int GRID_SIZE =
    10; // for both spatial hashing and marching cubes, -> spatial hashing은
        // 50으로 하고 marching cubes는 500 정도로 하는 게 좋을듯. cuda-opengl
        // interop 후 변경

}; // namespace slime::SPHSimulatorConstants

#endif