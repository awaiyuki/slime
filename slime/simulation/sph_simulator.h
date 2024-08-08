#ifndef SPH_SIMULATOR_CUH
#define SPH_SIMULATOR_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <memory>
#include <slime/constants/sph_simulator_constants.h>
#include "marching_cubes.h"

namespace slime {

struct Particle {
  int id;
  glm::vec3 position, velocity, acceleration;
  float density, pressure, mass;
  glm::vec4 color;
  float life;

  bool operator==(const Particle &p) { return this->id == p.id; }
};

class SPHSimulator {

public:
  SPHSimulator();
  ~SPHSimulator();

  void updateScalarField();
  void updateParticles(double deltaTime);

  float poly6Kernel(glm::vec3 r, float h);
  float spikyKernel(glm::vec3 r, float h);
  float gradientSpikyKernel(glm::vec3 r, float h);
  float viscosityKernel(glm::vec3 r, float h);
  float laplacianViscosityKernel(glm::vec3 r, float h);

  void computeDensity();
  void computePressureForce(double deltaTime);
  void computeViscosityForce(double deltaTime);
  void computeGravity(double deltaTime);
  void computeWallConstraint(double deltaTime);

  std::vector<MarchingCubes::Triangle> extractSurface();
  std::vector<glm::vec3> extractParticlePositions();

private:
  std::vector<Particle> particles;
  Particle *particlesDevice;

  static constexpr int GRID_SIZE = 50;
  float colorField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
  float *colorFieldDevice;
};

} // namespace slime
#endif