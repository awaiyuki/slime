#ifndef MARCHING_CUBES
#define MARCHING_CUBES

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>
#include <marching_cubes_tables.h>

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

  glm::vec3 interpolateVertices(glm::vec3 vertexA, glm::vec3 vertexB);

  template <size_t X, size_t Y, size_t Z>
  std::vector<MarchingCubes::Triangle> march(float (&scalarField)[X][Y][Z],
                                             float surfaceLevel) {

    std::vector<MarchingCubes::Triangle> triangles;

    /* verify if the vertex order is correct */
    int8_t diff[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1},
                         {0, 1, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1}};

    for (int x = 0; x < X; x++) {
      for (int y = 0; y < Y; y++) {
        for (int z = 0; z < Z; z++) {
          uint8_t tableKey = 0;
          for (int i = 0; i < 8; i++) {
            if (scalarField[x + diff[i][0]][y + diff[i][1]][z + diff[i][2]] >=
                surfaceLevel) {
              tableKey |= 1 << i;
            }
          }
          const std::vector<int8_t> &edges =
              MarchingCubesTables::triangulation[tableKey];

          for (int8_t edge : edges) {
            if (edge == -1)
              continue;
          }
        }
      }
    }
    return triangles;
  }
};
} // namespace slime
#endif MARCHING_CUBES