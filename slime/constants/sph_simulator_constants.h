#ifndef SPH_SIMULATOR_CONSTANTS_H
#define SPH_SIMULATOR_CONSTANTS_H
namespace slime {
class SPHSimulatorConstants {
public:
  static constexpr int NUM_PARTICLES = 5000;
  static constexpr float PARTICLE_MASS = 1.0f;
  static constexpr float GAS_CONSTANT = 0.001f;
  static constexpr float REST_DENSITY = 1.0f;
  static constexpr float VISCOSITY_COEFFICIENT = 0.1f;
  static constexpr float SMOOTHING_RADIUS = 1.0f;
  static constexpr float SURFACE_LEVEL = 0.8f;
};
} // namespace slime

#endif