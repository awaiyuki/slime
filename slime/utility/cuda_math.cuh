#ifndef CUDA_MATH_CUH
#define CUDA_MATH_CUH

#include <cuda.h>
#include <cuda_runtime.h>

inline __device__ __host__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __device__ __host__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __device__ __host__ float3 operator-(const float3 &a) {
  return make_float3(-a.x, -a.y, -a.z);
}
inline __device__ __host__ float3 operator*(const float &a, const float3 &b) {
  return make_float3(a * b.x, a * b.y, a * b.z);
}
inline __device__ __host__ float3 operator*(const float3 &b, const float &a) {
  return make_float3(a * b.x, a * b.y, a * b.z);
}
inline __device__ __host__ float3 operator/(const float3 &b, const float &a) {
  return make_float3(b.x / a, b.y / a, b.z / a);
}
inline __device__ __host__ float3 &operator*=(float3 &a, const float &s) {
  a.x *= s;
  a.y *= s;
  a.z *= s;
  return a;
}
inline __device__ __host__ float3 &operator+=(float3 &a, const float &s) {
  a.x += s;
  a.y += s;
  a.z += s;
  return a;
}
inline __device__ __host__ float3 &operator+=(float3 &a, float3 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}
inline __device__ __host__ float length(const float3 &a) {
  return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}
inline __device__ __host__ float3 normalize(const float3 &a) {
  return a / length(a);
}

inline __device__ __host__ float dot(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

#endif