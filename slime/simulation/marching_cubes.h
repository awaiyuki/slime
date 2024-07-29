#ifndef MARCHING_CUBES
#define MARCHING_CUBES

#include <cstdint>
#include <glm/glm.hpp>
#include <marching_cubes_tables.h>
#include <algorithm>
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

  std::pair<uint16_t, uint16_t> CornerIndexFromEdge(uint8_t edge);
  glm::vec3 CoordFromIndex(uint16_t x, uint16_t y, uint16_t z);
  glm::vec3 interpolateVertices(glm::vec3 vertexA, glm::vec3 vertexB);

  template <size_t X, size_t Y, size_t Z>
  std::vector<MarchingCubes::Triangle> march(float (&scalarField)[X][Y][Z],
                                             float surfaceLevel) {

    std::vector<MarchingCubes::Triangle> triangles;

    /* verify if the vertex order is correct */
    const int8_t diff[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1},
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

          for (int i = 0; i < edges.size(); i += 3) {
            if (i == -1)
              continue;

            MarchingCubes::Triangle triangle;
            triangle.v1 = interpolateVertices(
                CoordFromIndex(CornerIndexFromEdge(edges[i]).first),
                CoordFromIndex(CornerIndexFromEdge(edges[i]).second));
            triangle.v2 = interpolateVertices(
                CoordFromIndex(CornerIndexFromEdge(edges[i + 1]).first),
                CoordFromIndex(CornerIndexFromEdge(edges[i + 1]).second));
            triangle.v2 = interpolateVertices(
                CoordFromIndex(CornerIndexFromEdge(edges[i + 2]).first),
                CoordFromIndex(CornerIndexFromEdge(edges[i + 2]).second));

            triangles.push_back(triangle);
          }
        }
      }
    }
    return triangles;
  }
};
} // namespace slime
#endif MARCHING_CUBES