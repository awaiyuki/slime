#ifndef SPH_SIMULATOR_CONSTANTS_H
#define SPH_SIMULATOR_CONSTANTS_H
namespace slime {
class SPHSimulatorConstants {
public:
  static constexpr int NUM_PARTICLES = 40;
  static constexpr float GAS_CONSTANT = 0.1f;
  static constexpr float REST_DENSITY = 0.1f;
  static constexpr float VISCOSITY_COEFFICIENT = 0.1f;
  static constexpr float SMOOTHING_RADIUS = 0.1f;
  static constexpr float SURFACE_LEVEL = 10.0f;
};
} // namespace slime

#endif