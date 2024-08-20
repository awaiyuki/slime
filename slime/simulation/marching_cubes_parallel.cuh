#ifndef MARCHING_CUBES_PARALLEL_CUH
#define MARCHING_CUBES_PARALLEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>


inline __device__ __host__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __device__ __host__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __device__ __host__ float3 operator*(const float& a, const float3& b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}
inline __device__ __host__ float3 operator*(const float3& b, const float& a)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}
inline __device__ __host__ float3 operator/(const float3& b, const float& a)
{
    return make_float3(b.x / a, b.y / a, b.z / a);
}
inline __device__ __host__ float3& operator*=(float3& a, const float& s)
{
    a.x *= s;
    a.y *= s;
    a.z *= s;
    return a;
}
inline __device__ __host__ float3& operator+=(float3& a, const float& s)
{
    a.x += s;
    a.y += s;
    a.z += s;
    return a;
}
inline __device__ __host__ float3& operator+=(float3& a, float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
inline __device__ __host__ float length(float3& a)
{
    return sqrt(a.x * a.x * +a.y * a.y + a.z * a.z);
}
inline __device__ __host__ float3& normalize(float3& a)
{
    return a / length(a);
}


namespace slime {
    struct VertexData {
        float3* vertices;
        int size;
    };
extern __constant__ int d_triangulation[256][16];
extern __constant__ int d_cornerIndexFromEdge[12][2];

extern __device__ float3 interpolateVertices(float *d_scalarField, int gridSize,
                                             float surfaceLevel, int va[3],
                                             int vb[3]);

extern __global__ void marchParallel(float *d_scalarField, int gridSize,
                                     float surfaceLevel,
                                     slime::VertexData *d_vertexDataPtr, int *d_counter);

} // namespace slime

#endif