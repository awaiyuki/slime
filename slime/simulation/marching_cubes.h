#ifndef MARCHING_CUBES
#define MARCHING_CUBES

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

namespace slime {
class MarchingCubes {
private:
  static const std::vector<std::vector<int8_t>> triangulation;
  static const std::vector<int16_t> edgeTable;

public:
  struct Triangle {
    glm::vec3 v1, v2, v3;
  };

  MarchingCubes();
  ~MarchingCubes();

  template <size_t X, size_t Y, size_t Z>
  std::vector<Triangle> march(float (&scalarField)[X][Y][Z],
                              float surfaceLevel);
};
} // namespace slime
#endif MARCHING_CUBES