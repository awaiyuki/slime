#include "sph_simulator_device.cuh"
#include <slime/constants/sph_simulator_constants.h>
#include <math.h>
#include <slime/utility/cuda_math.cuh>
#define PI 3.141592653589793238462643
#define EPSILON 0.000001
using namespace slime;
using namespace slime::SPHSimulatorConstants;

__device__ unsigned int hash(int3 cell, unsigned int seed = 73856093) {
  unsigned int hash = seed;
  hash ^= (unsigned int)cell.x * 73856093;
  hash ^= (unsigned int)cell.y * 19349663;
  hash ^= (unsigned int)cell.z * 83492791;
  return hash;
}

__device__ int3 toCellPosition(float3 &position) {
  return make_int3(floor((position.x + 1.0f) / 2.0f * GRID_SIZE),
                   floor((position.y + 1.0f) / 2.0f * GRID_SIZE),
                   floor((position.z + 1.0f) / 2.0f * GRID_SIZE));
}

__device__ unsigned int keyFromHash(unsigned int &hashKey) {
  return hashKey % NUM_PARTICLES;
}

__device__ float poly6KernelDevice(float3 r, float h) {
  float rMagnitude = length(r);
  if (rMagnitude > h)
    return 0.0f;

  float result = 315.0f / (64.0f * PI * pow(h, 9)) *
                 pow(h * h - rMagnitude * rMagnitude, 3);
  return result;
}

__device__ float3 gradientPoly6Device(float3 r, float h) {
  float rLen = length(r);
  if (rLen > h || rLen < EPSILON)
    return make_float3(0, 0, 0);
  float base = 945.0f / (32.0f * PI * powf(h, 9.0f));
  float scalar = base * powf(h * h - rLen * rLen, 2);
  float3 grad = scalar * r;
  return -grad;
}

__device__ float spikyKernelDevice(float3 r, float h) { return 0.0f; }

__device__ float gradientSpikyKernelDevice(float3 r, float h) {
  float rMagnitude = length(r);
  if (rMagnitude > h)
    return 0.0f;

  return -45.0f / (PI * pow(h, 6)) * pow(h - rMagnitude, 2);
}

__device__ float viscosityKernelDevice(float3 r, float h) { return 0.0f; }

__device__ float laplacianViscosityKernelDevice(float3 r, float h) {
  float rMagnitude = length(r);
  if (rMagnitude > h)
    return 0.0f;

  return 45 / (PI * pow(h, 6)) * (h - rMagnitude);
}

__global__ void slime::updateScalarFieldDevice(float *colorFieldDevice,
                                               Particle *particlesDevice,
                                               int gridSize,
                                               float maxColorQuantity) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  const int idx = z * gridSize * gridSize + y * gridSize + x;

  // printf("test\n");
  if (x >= gridSize || y >= gridSize || z >= gridSize)
    return;

  float colorQuantity = 0.0f;
  float3 positionOnGrid =
      make_float3((static_cast<float>(x) / static_cast<float>(gridSize)),
                  (static_cast<float>(y) / static_cast<float>(gridSize)),
                  (static_cast<float>(z) / static_cast<float>(gridSize))) -
      make_float3(0.5f, 0.5f, 0.5f);
  for (int j = 0; j < NUM_PARTICLES; j++) {
    float3 r = positionOnGrid - particlesDevice[j].position;
    if (particlesDevice[j].density < EPSILON)
      continue;
    colorQuantity += particlesDevice[j].mass *
                     (1 / particlesDevice[j].density) *
                     poly6KernelDevice(r, 2.0 / static_cast<float>(gridSize));

    // colorQuantity += particlesDevice[j].density *
    //                  poly6KernelDevice(r, 2.0 /
    //                  static_cast<float>(gridSize));
  }

  colorFieldDevice[idx] = colorQuantity;

  // Test: Sphere
  // float dx = (x / float(gridSize - 1)) - 0.5f;
  // float dy = (y / float(gridSize - 1)) - 0.5f;
  // float dz = (z / float(gridSize - 1)) - 0.5f;
  // colorFieldDevice[idx] = dx * dx + dy * dy + dz * dz - 0.5f * 0.5f;
}

__global__ void slime::updateSpatialHashDevice(Particle *particlesDevice,
                                               unsigned int *hashKeys,
                                               unsigned int *hashIndices) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;

  auto &i = particlesDevice[idx];
  const int r = SMOOTHING_RADIUS;
  if (fabsf(i.position.x) > 1 || fabsf(i.position.y) > 1 ||
      fabsf(i.position.z) > 1)
    return;
  int3 cellPosition = toCellPosition(i.position);
  unsigned int cellHash = hash(cellPosition);
  unsigned int key = keyFromHash(cellHash);

  hashKeys[idx] = key;
  hashIndices[idx] = idx;
}

__global__ void slime::updateHashBucketDevice(unsigned int *hashKeys,
                                              unsigned int *hashIndices,
                                              unsigned int *bucketStart,
                                              unsigned int *bucketEnd) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;

  bucketStart[idx] = -1;
  bucketEnd[idx] = -1;

  if (idx == 0 || hashKeys[idx] != hashKeys[idx - 1]) {
    bucketStart[hashKeys[idx]] = idx;
  }

  if (idx == NUM_PARTICLES - 1) {
    bucketEnd[hashKeys[idx]] = hashIndices[idx] + 1;
  } else if (hashKeys[idx] != hashKeys[idx + 1]) {
    bucketEnd[hashKeys[idx]] = idx + 1;
  }
}

__global__ void slime::computeDensityDevice(Particle *particlesDevice,
                                            unsigned int *hashIndices,
                                            unsigned int *bucketStart,
                                            unsigned int *bucketEnd) {

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];

  i.density = 0.0f;

  int3 cellPosition = toCellPosition(i.position);
  /* Neighbour search using spatial hash */
  for (int nx = -1; nx <= 1; nx++) {
    for (int ny = -1; ny <= 1; ny++) {
      for (int nz = -1; nz <= 1; nz++) {
        unsigned int cellHash =
            hash(toCellPosition(i.position + make_float3(nx, ny, nz)));
        unsigned int key = keyFromHash(cellHash);
        if (!(key >= 0 && key < NUM_PARTICLES)) {
          printf("key:%d\n", key);
          continue;
        }
        if (bucketEnd[key] - bucketStart[key] > MAX_NEIGHBORS) {
          bucketEnd[key] = bucketStart[key] + MAX_NEIGHBORS;
          if (bucketEnd[key] > NUM_PARTICLES)
            bucketEnd[key] = NUM_PARTICLES - 1;
        }
        for (int bucketIdx = bucketStart[key]; bucketIdx < bucketEnd[key];
             bucketIdx++) {
          if (bucketIdx == -1)
            break;

          if (!(bucketIdx >= 0 && bucketIdx < NUM_PARTICLES)) {
            printf("bucketIdx:%d\n", bucketIdx);
            continue;
          }
          if (idx == hashIndices[bucketIdx])
            continue;
          if (!(hashIndices[bucketIdx] >= 0 &&
                hashIndices[bucketIdx] < NUM_PARTICLES)) {
            printf("hashIndices[bucketIdx]:%d\n", hashIndices[bucketIdx]);
            continue;
          }
          auto &j = particlesDevice[hashIndices[bucketIdx]];
          auto r = i.position - j.position;
          i.density += j.mass * poly6KernelDevice(r, SMOOTHING_RADIUS);
        }
      }
    }
  }
}

__global__ void slime::computePressureDevice(Particle *particlesDevice) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];
  i.pressure = GAS_CONSTANT * (i.density - REST_DENSITY);
}

__global__ void slime::computePressureForceDevice(Particle *particlesDevice,
                                                  unsigned int *hashIndices,
                                                  unsigned int *bucketStart,
                                                  unsigned int *bucketEnd,
                                                  double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];

  float3 pressureForce = make_float3(0.0f, 0.0f, 0.0f);

  int3 cellPosition = toCellPosition(i.position);

  /* Neighbour search using spatial hash */
  for (int nx = -1; nx <= 1; nx++) {
    for (int ny = -1; ny <= 1; ny++) {
      for (int nz = -1; nz <= 1; nz++) {
        unsigned int cellHash =
            hash(toCellPosition(i.position + make_float3(nx, ny, nz)));
        unsigned int key = keyFromHash(cellHash);
        if (bucketEnd[key] - bucketStart[key] > MAX_NEIGHBORS) {
          bucketEnd[key] = bucketStart[key] + MAX_NEIGHBORS;
          if (bucketEnd[key] > NUM_PARTICLES)
            bucketEnd[key] = NUM_PARTICLES - 1;
        }
        for (int bucketIdx = bucketStart[key]; bucketIdx < bucketEnd[key];
             bucketIdx++) {
          if (bucketIdx == -1)
            break;
          // TODO: implement
          if (idx == hashIndices[bucketIdx])
            continue;
          auto &j = particlesDevice[hashIndices[bucketIdx]];
          if (j.density < EPSILON)
            continue;

          auto r = i.position - j.position;
          pressureForce += (-1) * normalize(r) * j.mass *
                           (i.pressure + j.pressure) / (2.0f * j.density) *
                           gradientSpikyKernelDevice(r, SMOOTHING_RADIUS);
        }
      }
    }
  }
  auto acceleration = pressureForce / i.mass;
  auto deltaVelocity = acceleration * float(deltaTime);
  i.velocity += deltaVelocity;
}

__global__ void slime::computeViscosityForceDevice(Particle *particlesDevice,
                                                   unsigned int *hashIndices,
                                                   unsigned int *bucketStart,
                                                   unsigned int *bucketEnd,
                                                   double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];
  float3 viscosityForce = make_float3(0.0f, 0.0f, 0.0f);

  /* Neighbour search using spatial hash */
  for (int nx = -1; nx <= 1; nx++) {
    for (int ny = -1; ny <= 1; ny++) {
      for (int nz = -1; nz <= 1; nz++) {
        unsigned int cellHash =
            hash(toCellPosition(i.position + make_float3(nx, ny, nz)));
        unsigned int key = keyFromHash(cellHash);
        if (bucketEnd[key] - bucketStart[key] > MAX_NEIGHBORS) {
          bucketEnd[key] = bucketStart[key] + MAX_NEIGHBORS;
          if (bucketEnd[key] > NUM_PARTICLES)
            bucketEnd[key] = NUM_PARTICLES - 1;
        }
        for (int bucketIdx = bucketStart[key]; bucketIdx < bucketEnd[key];
             bucketIdx++) {
          if (bucketIdx == -1)
            break;
          // TODO: implement
          if (idx == hashIndices[bucketIdx])
            continue;
          auto &j = particlesDevice[hashIndices[bucketIdx]];

          if (j.density < EPSILON)
            continue;

          auto r = i.position - j.position;
          viscosityForce += j.mass * (j.velocity - i.velocity) / j.density *
                            laplacianViscosityKernelDevice(r, SMOOTHING_RADIUS);
        }
      }
    }
  }
  viscosityForce *= 0.16f;
  // slime::SPHSimulatorConstants::VISCOSITY_COEFFICIENT

  auto acceleration = viscosityForce / i.mass;
  auto deltaVelocity = acceleration * float(deltaTime);
  i.velocity += deltaVelocity;
}
__device__ constexpr float SURFACE_TENSION_COEFFICIENT = 0.072f;
__device__ constexpr float SURFACE_NORMAL_THRESHOLD = 0.1f;
__device__ constexpr float EPS_CURVATURE = 1e-5f;

__global__ void slime::computeSurfaceTensionDevice(Particle *particlesDevice,
                                                   unsigned int *hashIndices,
                                                   unsigned int *bucketStart,
                                                   unsigned int *bucketEnd,
                                                   double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;

  Particle &pi = particlesDevice[idx];
  float ci = pi.density / REST_DENSITY;

  float3 normal_i = make_float3(0.0f, 0.0f, 0.0f);

  int3 cellPos = toCellPosition(pi.position);
  for (int nx = -1; nx <= 1; nx++) {
    for (int ny = -1; ny <= 1; ny++) {
      for (int nz = -1; nz <= 1; nz++) {
        unsigned int cellHash =
            hash(toCellPosition(pi.position + make_float3(nx, ny, nz)));
        unsigned int key = keyFromHash(cellHash);

        if (bucketEnd[key] - bucketStart[key] > MAX_NEIGHBORS) {
          bucketEnd[key] = bucketStart[key] + MAX_NEIGHBORS;
          if (bucketEnd[key] > NUM_PARTICLES)
            bucketEnd[key] = NUM_PARTICLES - 1;
        }
        for (int bucketIdx = bucketStart[key]; bucketIdx < bucketEnd[key];
             bucketIdx++) {
          if (bucketIdx == -1)
            break;
          if (idx == hashIndices[bucketIdx])
            continue;

          Particle &pj = particlesDevice[hashIndices[bucketIdx]];

          float cj = pj.density / REST_DENSITY;

          float3 r = pi.position - pj.position;
          float rLen = length(r);
          if (rLen > SMOOTHING_RADIUS || rLen < EPSILON)
            continue;

          float3 gradW = gradientPoly6Device(r, SMOOTHING_RADIUS);

          normal_i += (pj.mass / pj.density) * cj * gradW;
        }
      }
    }
  }

  float normLen = length(normal_i);

  if (normLen > SURFACE_NORMAL_THRESHOLD) {
    float curvature_i = 0.0f;

    for (int nx = -1; nx <= 1; nx++) {
      for (int ny = -1; ny <= 1; ny++) {
        for (int nz = -1; nz <= 1; nz++) {
          unsigned int cellHash =
              hash(toCellPosition(pi.position + make_float3(nx, ny, nz)));
          unsigned int key = keyFromHash(cellHash);
          if (bucketEnd[key] - bucketStart[key] > MAX_NEIGHBORS) {
            bucketEnd[key] = bucketStart[key] + MAX_NEIGHBORS;
            if (bucketEnd[key] > NUM_PARTICLES)
              bucketEnd[key] = NUM_PARTICLES - 1;
          }
          for (int bucketIdx = bucketStart[key]; bucketIdx < bucketEnd[key];
               bucketIdx++) {
            if (bucketIdx == -1)
              break;
            if (idx == hashIndices[bucketIdx])
              continue;

            Particle &pj = particlesDevice[hashIndices[bucketIdx]];

            float3 r = pi.position - pj.position;
            float rLen = length(r);
            if (rLen > SMOOTHING_RADIUS || rLen < EPSILON)
              continue;

            float cj = pj.density / REST_DENSITY;
            float3 gradW = gradientPoly6Device(r, SMOOTHING_RADIUS);

            float3 normal_j = (pj.mass / pj.density) * cj * gradW;
            float normal_j_len = length(normal_j);
            if (normal_j_len < EPSILON)
              continue;

            float3 n_j_norm = normal_j / normal_j_len;

            curvature_i -= (pj.mass / pj.density) * dot(n_j_norm, gradW);
          }
        }
      }
    }

    float3 n_i_dir =
        (normLen > EPS_CURVATURE) ? (normal_i / normLen) : make_float3(0, 0, 0);

    float3 F_st = -SURFACE_TENSION_COEFFICIENT * curvature_i * n_i_dir;
    float3 a_st = F_st / pi.mass;

    pi.velocity += a_st * static_cast<float>(deltaTime);
  }
}

__global__ void slime::computeGravityDevice(Particle *particlesDevice,
                                            double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];
  auto acceleration = make_float3(0, GRAVITATIONAL_ACCELERATION, 0);
  auto deltaVelocity = acceleration * float(deltaTime);
  i.velocity += deltaVelocity;
}

__global__ void slime::computeWallConstraintDevice(Particle *particlesDevice,
                                                   double deltaTime) {
  /* Handling Collision */
  /* Simulation Space: x, y, z in [-1, 1] */

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;

  auto &i = particlesDevice[idx];
  const float COLLISION_DAMPING = 0.9f;
  const float3 offsetFromCenter = i.position - make_float3(0.0f, 0.0f, 0.0f);

  if (fabsf(offsetFromCenter.x) > 1.01) {
    i.position.x = (i.position.x > 0) - (i.position.x < 0);
    i.velocity.x = -i.velocity.x * COLLISION_DAMPING;
  }
  if (fabsf(offsetFromCenter.y) > 1.01) {
    i.position.y = (i.position.y > 0) - (i.position.y < 0);
    i.velocity.y = -i.velocity.y * COLLISION_DAMPING;
  }
  if (fabsf(offsetFromCenter.z) > 1.01) {
    i.position.z = (i.position.z > 0) - (i.position.z < 0);
    i.velocity.z = -i.velocity.z * COLLISION_DAMPING;
  }
}

__global__ void slime::computePositionDevice(Particle *particlesDevice,
                                             double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];
  i.position += i.velocity * static_cast<float>(deltaTime);

  // if (idx % int(NUM_PARTICLES / 5) == 0) {
  //   printf("v %f %f %f\n", i.velocity.x, i.velocity.y, i.velocity.z);
  //   printf("p %f %f %f\n", i.position.x, i.position.y, i.position.z);
  // }
}

__global__ void slime::copyPositionToVBODevice(float *d_positions,
                                               Particle *particlesDevice) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];

  d_positions[3 * idx] = i.position.x;
  d_positions[3 * idx + 1] = i.position.y;
  d_positions[3 * idx + 2] = i.position.z;
}