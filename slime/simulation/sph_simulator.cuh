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
  float3 position, velocity, acceleration;
  float density, pressure, mass;
  float4 color;
  float life;

  __host__ __device__ bool operator==(const Particle &p) { return this->id == p.id; }
};

class SPHSimulator {

public:
  SPHSimulator();
  ~SPHSimulator();

  void updateScalarField();
  void updateParticles(double deltaTime);

  std::vector<glm::vec3> extractSurface();
  std::vector<float> extractParticlePositions();

private:
  std::vector<Particle> particles;
  Particle *particlesDevice;

  static constexpr int GRID_SIZE = 50;
  float colorField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
  float *colorFieldDevice;
};

} // namespace slime
#endif