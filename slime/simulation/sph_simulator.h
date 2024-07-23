#ifndef SPH_SIMULATOR_H
#define SPH_SIMULATOR_H

#include "marching_cubes.h"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace slime {

class SPHSimulator {

public:
  struct Particle {
    glm::vec3 position, velocity, acceleration;
    float density, pressure, mass;
    glm::vec4 color;
    float life;
  };
  SPHSimulator();
  ~SPHSimulator();

  float poly6Kernel(glm::vec3 rSquare, float h);
  float spikyKernel(glm::vec3 r, float h);
  float gradientSpikyKernel(glm::vec3 r, float h);
  float viscosityKernel(glm::vec3 r, float h);
  float laplacianViscosityKernel(glm::vec3 r, float h);

  void updateParticles(float deltaTime);
  void computeDensity();
  void computePressureForce(double deltaTime);
  void computeViscosityForce(double deltaTime);
  void computeGravity();

  void initScalarField();
  void updateScalarField();

  std::vector<MarchingCubes::Triangle> extractSurface();

private:
  static const int NUM_PARTICLES = 5000;
  static const int GRID_SIZE = 200;
  static const float GAS_CONSTANT = 100.0;
  static const float REST_DENSITY = 1.0; // unit scale
  static const float VISCOSITY_COEFFICIENT = 0.1f;
  static const float SMOOTHING_RADIUS = 0.1f;
  static const float SURFACE_LEVEL = 0.5f;
  std::vector<std::unique_ptr<Particle>> particles;

  float densityField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
  float pressureField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
  float viscosityField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
  float colorField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
  float surfaceTensionField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
};
} // namespace slime
#endif SPH_SIMULATOR_H