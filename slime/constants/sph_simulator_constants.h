#ifndef SPH_SIMULATOR_CONSTANTS_H
#define SPH_SIMULATOR_CONSTANTS_H
namespace slime {
class SPHSimulatorConstants {
public:
  static constexpr int NUM_PARTICLES = 50;
  static constexpr float GAS_CONSTANT = 100.0;
  static constexpr float REST_DENSITY = 1.0; // unit scale
  static constexpr float VISCOSITY_COEFFICIENT = 0.1f;
  static constexpr float SMOOTHING_RADIUS = 0.1f;
  static constexpr float SURFACE_LEVEL = 0.5f;
};
} // namespace slime

#endif SPH_SIMULATOR_CONSTANTS_H