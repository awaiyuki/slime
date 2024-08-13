#include "marching_cubes_parallel.cuh"

using namespace slime;

__device__ __host__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ float3 operator*(const float &a, const float3 &b) {
  return make_float3(a * b.x, a * b.y, a * b.z);
}
__device__ __host__ float3 operator*(const float3 &b, const float &a) {
  return make_float3(a * b.x, a * b.y, a * b.z);
}
__device__ __host__ float3 operator/(const float3 &b, const float &a) {
  return make_float3(b.x / a, b.y / a, b.z / a);
}
__device__ __host__ inline float3 &operator*=(float3 &a, const float &s) {
  a.x *= s;
  a.y *= s;
  a.z *= s;
  return a;
}
__device__ __host__ inline float3 &operator+=(float3 &a, const float &s) {
  a.x += s;
  a.y += s;
  a.z += s;
  return a;
}
__device__ __host__ inline float3 &operator+=(float3 &a, float3 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}
__device__ __host__ inline float length(float3 &a) {
  return sqrt(a.x * a.x * +a.y * a.y + a.z * a.z);
}
__device__ __host__ inline float3 &normalize(float3 &a) {
  return a / length(a);
}

__device__ float3 slime::interpolateVertices(float *d_scalarField, int gridSize,
                                             float surfaceLevel, int va[3],
                                             int vb[3]) {
  float scalarA =
      d_scalarField[va[2] * gridSize * gridSize + va[1] * gridSize + va[0]];
  float scalarB =
      d_scalarField[vb[2] * gridSize * gridSize + vb[1] * gridSize + vb[0]];
  float t = (surfaceLevel - scalarA) / (scalarB - scalarA);
  return make_float3(va[0], va[1], va[2]) +
         t * (make_float3(vb[0], vb[1], vb[2]) -
              make_float3(va[0], va[1], va[2]));
}

__global__ void slime::marchParallel(float *d_scalarField, int gridSize,
                                     float surfaceLevel,
                                     MarchingCubes::Triangle *d_triangles) {

  /* verify if the vertex order is correct */
  __shared__ const int diff[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 0, 1},
                                     {0, 0, 1}, {0, 1, 0}, {1, 1, 0},
                                     {1, 1, 1}, {0, 1, 1}};

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.x * blockIdx.y;
  int z = threadIdx.z + blockDim.x * blockIdx.z;

  float3 currentPosition = make_float3(x, y, z);
  float3 cubeVertices[8];
  int cubeVertexCoordInt[8][3];

  for (int i = 0; i < 8; i++) {
    cubeVertices[i] =
        currentPosition + make_float3(diff[i][0], diff[i][1], diff[i][2]);
    cubeVertexCoordInt[i][0] = x + diff[i][0];
    cubeVertexCoordInt[i][1] = y + diff[i][1];
    cubeVertexCoordInt[i][2] = z + diff[i][2];
  }

  uint8_t tableKey = 0;
  for (int i = 0; i < 8; i++) {
    if (d_scalarField[(z + diff[i][2]) * gridSize * gridSize +
                      (y + diff[i][1]) * gridSize + (x + diff[i][0])] <
        surfaceLevel) { // correct?
      tableKey |= 1 << i;
    }
  }
  int edges[16];
  std::copy(d_triangulation[tableKey], d_triangulation[tableKey] + 16, edges);

  for (int i = 0; i < 16; i += 3) {
    if (edges[i] == -1)
      continue;
    MarchingCubes::Triangle triangle;
    triangle.v1 = interpolateVertices(
        d_scalarField, gridSize, surfaceLevel,
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i]][0]],
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i]][1]]);
    triangle.v2 = interpolateVertices(
        d_scalarField, gridSize, surfaceLevel,
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i + 1]][0]],
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i + 1]][1]]);
    triangle.v3 = interpolateVertices(
        d_scalarField, gridSize, surfaceLevel,
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i + 2]][0]],
        cubeVertexCoordInt[d_cornerIndexFromEdge[edges[i + 2]][1]]);

    d_triangles.push_back(
        triangle); // replace with fixed-size array + atomic counter
    // std::cout << "extract surface, triangle.v1[0]: " <<
    // triangle.v1[0]
    //           << std::endl;
  }
}