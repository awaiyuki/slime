#include "marching_cubes.h"

#include <bitset>
#include <iostream>

using namespace slime;
using namespace std;

MarchingCubes::MarchingCubes() {}

MarchingCubes::~MarchingCubes() {}

std::pair<uint16_t, uint16_t> CornerIndexFromEdge(uint8_t edge);
glm::vec3 CoordFromIndex(uint16_t x, uint16_t y, uint16_t z);

glm::vec3 interpolateVertices(glm::vec3 vertexA, glm::vec3 vertexB) {}