#include "marching_cubes.h"

#include <bitset>
#include <iostream>

using namespace slime;
using namespace std;

MarchingCubes::MarchingCubes() {}

MarchingCubes::~MarchingCubes() {}

glm::vec3 CoordFromIndex(uint16_t x, uint16_t y, uint16_t z) {
  return glm::vec3(static_cast<float>(x), static_cast<float>(y),
                   static_cast<float>(z));
}
