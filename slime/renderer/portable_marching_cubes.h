#ifndef PORTABLE_MARCHING_CUBES_H
#define PORTABLE_MARCHING_CUBES_H

#include <glm/glm.hpp>
#include <vector>

namespace slime {

struct SurfaceVertex {
  glm::vec3 position;
  glm::vec3 normal;
};

class PortableMarchingCubes {
public:
  void rebuild(const std::vector<glm::vec3> &particles);
  const std::vector<SurfaceVertex> &vertices() const;

private:
  static constexpr int gridSize_ = 64;
  std::vector<float> scalarField_;
  std::vector<SurfaceVertex> vertices_;
};

} // namespace slime

#endif
