#ifndef SPH_SIMULATOR_CONSTANTS_H
#define SPH_SIMULATOR_CONSTANTS_H
namespace slime {
class SPHSimulatorConstants {
public:
  static constexpr int NUM_PARTICLES = 300;
  static constexpr float GAS_CONSTANT = 0.009f;
  static constexpr float REST_DENSITY = 0.1f;
  static constexpr float VISCOSITY_COEFFICIENT = 0.1f;
  static constexpr float SMOOTHING_RADIUS = 1.0f;
  static constexpr float SURFACE_LEVEL = 0.5f;
};
} // namespace slime

#endif