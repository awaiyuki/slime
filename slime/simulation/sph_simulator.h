#ifndef SPH_SIMULATOR_H
#define SPH_SIMULATOR_H

#include "marching_cubes.h"
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <slime/constants/sph_simulator_constants.h>

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

  float poly6Kernel(glm::vec3 r, float h);
  float spikyKernel(glm::vec3 r, float h);
  float gradientSpikyKernel(glm::vec3 r, float h);
  float viscosityKernel(glm::vec3 r, float h);
  float laplacianViscosityKernel(glm::vec3 r, float h);

  void updateParticles(double deltaTime);
  void computeDensity();
  void computePressureForce(double deltaTime);
  void computeViscosityForce(double deltaTime);
  void computeGravity(double deltaTime);
  void computeWallConstraint(double deltaTime);

  void initScalarField();
  void updateScalarField();

  std::vector<MarchingCubes::Triangle> extractSurface();
  std::vector<glm::vec3> extractParticlePositions();

private:
  std::vector<std::unique_ptr<Particle>> particles;

  static constexpr int GRID_SIZE = 20;
  float densityField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
  float pressureField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
  float viscosityField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
  float colorField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
  float surfaceTensionField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
};
} // namespace slime
#endif