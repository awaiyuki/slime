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
  float3 position, velocity;
  float density, pressure, mass;
  // float4 color;
  // float life;
};

class SPHSimulator {

public:
  SPHSimulator(const unsigned int vbo, const std::string _renderMode = "point");
  ~SPHSimulator();

  std::vector<Particle> *getParticlesPointer();
  void updateScalarField();
  void updateParticles(double deltaTime);

  VertexData extractSurface();

private:
  std::vector<Particle> particles;

  std::string renderMode;

  thrust::device_vector<unsigned int> hashKeys;
  thrust::device_vector<unsigned int> hashIndices;
  thrust::device_vector<unsigned int> bucketStart;
  thrust::device_vector<unsigned int> bucketEnd;

  unsigned int *raw_hashKeys;
  unsigned int *raw_hashIndices;
  unsigned int *raw_bucketStart;
  unsigned int *raw_bucketEnd;

  Particle *d_particles;
  cudaGraphicsResource_t cudaVBOResource;

  float *d_scalarField;

  std::unique_ptr<MarchingCubes> marchingCubes;
};

} // namespace slime
#endif