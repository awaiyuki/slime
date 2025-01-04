#include "sph_simulator_device.cuh"
#include <slime/constants/sph_simulator_constants.h>
#include <math.h>
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
        // printf("%d %d %d\n", key, bucketStart[key], bucketStart[key]);
        for (int bucketIdx = bucketStart[key]; bucketIdx < bucketEnd[key];
             bucketIdx++) {
          if (bucketIdx == -1)
            break;
          // TODO: implement
          if (idx == hashIndices[bucketIdx])
            continue;
          auto &j = particlesDevice[hashIndices[bucketIdx]];
          auto r = j.position - i.position;
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
        unsigned int key =
            hash(toCellPosition(i.position + make_float3(nx, ny, nz)));
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

          auto r = j.position - i.position;
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
        unsigned int key =
            hash(toCellPosition(i.position + make_float3(nx, ny, nz)));
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

          auto r = j.position - i.position;
          viscosityForce += j.mass * (j.velocity - i.velocity) / j.density *
                            laplacianViscosityKernelDevice(r, SMOOTHING_RADIUS);
        }
      }
    }
  }
  viscosityForce *= 0.1f; // VISCOSITY_COEFFICIENT

  auto acceleration = viscosityForce / i.mass;
  auto deltaVelocity = acceleration * float(deltaTime);
  i.velocity += deltaVelocity;
}

__global__ void slime::computeSurfaceTensionDevice(Particle *particlesDevice,
                                                   double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];

  // TODO: implement surface tension

  return;
}

__global__ void slime::computeGravityDevice(Particle *particlesDevice,
                                            double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];
  auto acceleration = make_float3(0, -0.098f, 0);
  auto deltaVelocity = acceleration * float(deltaTime);
  i.velocity += deltaVelocity;
}

__global__ void slime::computeWallConstraintDevice(Particle *particlesDevice,
                                                   double deltaTime) {
  /* Handling Collision */
  /* Simulation Space: x, y, z in [-0.5, 0.5] */

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];
  const float FLOOR_CONSTRAINT = -0.5f;
  const float CEILING_CONSTRAINT = 0.5f;
  const float SPRING_CONSTANT = 10.0f;
  const float DAMPING = 0.7f;
  if (i.position.x < FLOOR_CONSTRAINT) {
    auto deltaVelocity = (SPRING_CONSTANT * (FLOOR_CONSTRAINT - i.position.x) +
                          DAMPING * i.velocity.x) *
                         float(deltaTime);
    i.velocity.x += deltaVelocity;
  }

  if (i.position.x > CEILING_CONSTRAINT) {
    auto deltaVelocity =
        (SPRING_CONSTANT * (i.position.x - CEILING_CONSTRAINT) +
         DAMPING * i.velocity.x) *
        float(deltaTime);
    i.velocity.x -= deltaVelocity;
  }
  if (i.position.y < FLOOR_CONSTRAINT) {
    auto deltaVelocity = (SPRING_CONSTANT * (FLOOR_CONSTRAINT - i.position.y) +
                          DAMPING * i.velocity.y) *
                         float(deltaTime);
    i.velocity.y += deltaVelocity;
  }

  if (i.position.y > CEILING_CONSTRAINT) {
    auto deltaVelocity =
        (SPRING_CONSTANT * (i.position.y - CEILING_CONSTRAINT) +
         DAMPING * i.velocity.y) *
        float(deltaTime);
    i.velocity.y -= deltaVelocity;
  }
  if (i.position.z < FLOOR_CONSTRAINT) {
    auto deltaVelocity = (SPRING_CONSTANT * (FLOOR_CONSTRAINT - i.position.z) +
                          DAMPING * i.velocity.z) *
                         float(deltaTime);
    i.velocity.z += deltaVelocity;
  }

  if (i.position.z > CEILING_CONSTRAINT) {
    auto deltaVelocity =
        (SPRING_CONSTANT * (i.position.z - CEILING_CONSTRAINT) +
         DAMPING * i.velocity.z) *
        float(deltaTime);
    i.velocity.z -= deltaVelocity;
  }
}

__global__ void slime::computePositionDevice(Particle *particlesDevice,
                                             double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = particlesDevice[idx];
  i.position += i.velocity * static_cast<float>(deltaTime);
}

__global__ void slime::updateSpatialHashDevice(Particle *particlesDevice,
                                               unsigned int *hashKeys,
                                               unsigned int *hashIndices) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;

  auto &i = particlesDevice[idx];
  const int r = SMOOTHING_RADIUS;

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

  if (idx == 0 || hashKeys[idx] != hashKeys[idx - 1]) {
    bucketStart[hashKeys[idx]] = idx;
  }

  if (idx == NUM_PARTICLES - 1) {
    bucketEnd[hashKeys[idx]] = hashIndices[idx] + 1;
  } else if (hashKeys[idx] != hashKeys[idx + 1]) {
    bucketEnd[hashKeys[idx]] = idx + 1;
  }
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