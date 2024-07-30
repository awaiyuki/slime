#ifndef MARCHING_CUBES_H
#define MARCHING_CUBES_H

#include <cstdint>
#include <glm/glm.hpp>
#include "marching_cubes_tables.h"
#include <algorithm>
#include <vector>
#include <iostream>

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

  glm::vec3 CoordFromIndex(uint16_t x, uint16_t y, uint16_t z);

  template <size_t X, size_t Y, size_t Z>
  glm::vec3 interpolateVertices(float (&scalarField)[X][Y][Z],
                                float surfaceLevel, int va[3], int vb[3]) {
    float scalarA = scalarField[va[0]][va[1]][va[2]];
    float scalarB = scalarField[vb[0]][vb[1]][vb[2]];
    float t = (surfaceLevel - scalarA) / (scalarB - scalarA);
    return glm::vec3(va[0], va[1], va[2]) +
           t * (glm::vec3(vb[0], vb[1], vb[2]) -
                glm::vec3(va[0], va[1], va[2]));
  }

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
          // std::cout << "march: " << x << y << z << std::endl;
          glm::vec3 currentPosition = glm::vec3(x, y, z);
          glm::vec3 cubeVertices[8];
          int cubeVertexCoordInt[8][3];

          for (int i = 0; i < 8; i++) {
            cubeVertices[i] =
                currentPosition + glm::vec3(diff[i][0], diff[i][1], diff[i][1]);
            cubeVertexCoordInt[i][0] = x + diff[i][0];
            cubeVertexCoordInt[i][1] = y + diff[i][1];
            cubeVertexCoordInt[i][2] = z + diff[i][2];
          }

          uint8_t tableKey = 0;
          for (int i = 0; i < 8; i++) {
            if (scalarField[x + diff[i][0]][y + diff[i][1]][z + diff[i][2]] >=
                surfaceLevel) { // correct?
              tableKey |= 1 << i;
            }
          }
          int8_t edges[16];
          std::copy(MarchingCubesTables::triangulation[tableKey],
                    MarchingCubesTables::triangulation[tableKey] + 16, edges);

          for (int i = 0; i < 16; i += 3) {
            if (edges[i] == -1)
              continue;
            MarchingCubes::Triangle triangle;
            triangle.v1 = interpolateVertices(
                scalarField, surfaceLevel,
                cubeVertexCoordInt
                    [MarchingCubesTables::cornerIndexFromEdge[edges[i]][0]],
                cubeVertexCoordInt
                    [MarchingCubesTables::cornerIndexFromEdge[edges[i]][1]]);
            triangle.v2 = interpolateVertices(
                scalarField, surfaceLevel,
                cubeVertexCoordInt
                    [MarchingCubesTables::cornerIndexFromEdge[edges[i + 1]][0]],
                cubeVertexCoordInt[MarchingCubesTables::cornerIndexFromEdge
                                       [edges[i + 1]][1]]);
            triangle.v2 = interpolateVertices(
                scalarField, surfaceLevel,
                cubeVertexCoordInt
                    [MarchingCubesTables::cornerIndexFromEdge[edges[i + 2]][0]],
                cubeVertexCoordInt[MarchingCubesTables::cornerIndexFromEdge
                                       [edges[i + 2]][1]]);

            triangles.push_back(triangle);
            // std::cout << "extract surface, triangle.v1[0]: " <<
            // triangle.v1[0]
            //           << std::endl;
          }
        }
      }
    }
    return triangles;
  }
};
} // namespace slime
#endif