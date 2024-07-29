#include "marching_cubes_tables.h"

using namespace slime;

constexpr int8_t MarchingCubesTables::cornerIndexFromEdge[12][2];
constexpr int8_t MarchingCubesTables::triangulation[256][16];