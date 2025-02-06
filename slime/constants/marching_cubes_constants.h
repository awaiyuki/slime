#ifndef MARCHING_CUBES_CONSTANTS_H
#define MARCHING_CUBES_CONSTANTS_H
namespace slime::MarchingCubesConstants {

constexpr int THREAD_SIZE_IN_MARCH = 8;
constexpr int THREAD_SIZE_IN_COPY_VERTEX_DATA = 512;
}; // namespace slime::MarchingCubesConstants

#endif