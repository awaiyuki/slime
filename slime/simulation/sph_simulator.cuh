#ifndef SPH_SIMULATOR_H
#define SPH_SIMULATOR_H

#include "marching_cubes.h"
#include <glm/glm.hpp>
#include <memory>
#include <slime/constants/sph_simulator_constants.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace slime {

__global__ void updateParticles(Particle *particles, int particleCount,
                                double deltaTime);
__global__void initScalarField(int gridSize);
__global__ void updateScalarField(Particle *particles, int particleCount,
                                  int gridSize);

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

  __device__ float poly6Kernel(glm::vec3 rSquare, float h);
  __device__ float spikyKernel(glm::vec3 r, float h);
  __device__ float gradientSpikyKernel(glm::vec3 r, float h);
  __device__ float viscosityKernel(glm::vec3 r, float h);
  __device__ float laplacianViscosityKernel(glm::vec3 r, float h);

  __device__ void computeDensity();
  __device__ void computePressureForce(double deltaTime);
  __device__ void computeViscosityForce(double deltaTime);
  __device__ void computeGravity(double deltaTime);

  std::vector<MarchingCubes::Triangle> extractSurface();

private:
  thrust::host_vector<Particle> particles;

  static constexpr int GRID_SIZE = 200;
  float densityField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
  float pressureField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
  float viscosityField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
  float colorField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
  float surfaceTensionField[GRID_SIZE][GRID_SIZE][GRID_SIZE];
};

} // namespace slime
#endif