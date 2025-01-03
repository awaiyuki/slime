#ifndef SPH_SIMULATOR_CUH
#define SPH_SIMULATOR_CUH

#include <functional>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <slime/constants/sph_simulator_constants.h>
#include <slime/renderer/marching_cubes.cuh>
#include <glm/glm.hpp>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace slime {

struct Particle {
  int id;
  float3 position, velocity, acceleration;
  float density, pressure, mass;
  float4 color;
  float life;
};

class SPHSimulator {

public:
  SPHSimulator(const unsigned int vbo);
  ~SPHSimulator();

  std::vector<Particle> *getParticlesPointer();
  void updateScalarField();
  void updateParticles(double deltaTime);

  VertexData extractSurface();

private:
  std::vector<Particle> particles;

  thrust::device_vector<unsigned int> hashKeys;
  thrust::device_vector<unsigned int> hashIndices;

  unsigned int *raw_hashKeys;
  unsigned int *raw_hashIndices;

  Particle *d_particles;
  cudaGraphicsResource_t cudaVBOResource;

  static constexpr int GRID_SIZE = SPHSimulatorConstants::GRID_SIZE;
  float scalarField[GRID_SIZE * GRID_SIZE * GRID_SIZE];
  float *d_scalarField;

  std::unique_ptr<MarchingCubes> marchingCubes;
};

} // namespace slime
#endif