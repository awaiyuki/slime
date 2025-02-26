#include "sph_simulator_parallel.cuh"
#include <slime/constants/sph_simulator_constants.h>
#include <math.h>
#include <slime/utility/cuda_math.cuh>
#define PI 3.141592653589793238462643
#define EPSILON 1e-5
using namespace slime;
using namespace slime::SPHSimulatorConstants;

/* rename: global -> g_~, device -> d_~ */
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
  float rLen = length(r);
  if (rLen > h)
    return 0.0f;

  float result =
      315.0f / (64.0f * PI * powf(h, 9)) * pow(h * h - rLen * rLen, 3);
  return result;
}

__device__ float3 gradientPoly6Device(float3 r, float h) {
  float rLen = length(r);
  if (rLen > h || rLen < EPSILON)
    return make_float3(0, 0, 0);
  float h9 = powf(h, 9);
  float base = 945.0f / (32.0f * PI * h9);
  float scalar = base * powf(h * h - rLen * rLen, 2);
  float3 grad = scalar * r;
  return -grad;
}

__device__ float laplacianPoly6Device(const float3 &r, float h) {

  float rSquare = dot(r, r);
  float hSqaure = h * h;

  if (rSquare > hSqaure)
    return 0.0f;

  float h9 = powf(h, 9);
  float factor = 945.0f / (32.0f * PI * h9);

  float term = hSqaure - rSquare;

  return -factor * term * (3.0f * hSqaure - 7.0f * rSquare);
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

__global__ void slime::g_updateScalarField(float *colorFieldDevice,
                                           Particle *d_particles,
                                           int gridSize) {
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
    float3 r = positionOnGrid - d_particles[j].position;
    if (d_particles[j].density < EPSILON)
      continue;
    colorQuantity += d_particles[j].mass * (1 / d_particles[j].density) *
                     poly6KernelDevice(r, 2.0 / static_cast<float>(gridSize));

    // colorQuantity += d_particles[j].density *
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

__global__ void slime::g_updateSpatialHash(Particle *d_particles,
                                           unsigned int *hashKeys,
                                           unsigned int *hashIndices) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;

  auto &i = d_particles[idx];
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

__global__ void slime::g_updateHashBucket(unsigned int *hashKeys,
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

  if (idx == NUM_PARTICLES - 1 || hashKeys[idx] != hashKeys[idx + 1]) {
    bucketEnd[hashKeys[idx]] = idx + 1;
  }
}

__global__ void slime::g_computeDensity(Particle *d_particles,
                                        unsigned int *hashIndices,
                                        unsigned int *bucketStart,
                                        unsigned int *bucketEnd) {

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = d_particles[idx];

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
          auto &j = d_particles[hashIndices[bucketIdx]];
          auto r = i.position - j.position;
          i.density += j.mass * poly6KernelDevice(r, SMOOTHING_RADIUS);
        }
      }
    }
  }
}

__global__ void slime::g_computePressure(Particle *d_particles) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = d_particles[idx];
  i.pressure = GAS_CONSTANT * (i.density - REST_DENSITY);
}

__global__ void slime::g_computeForces(Particle *d_particles,
                                       unsigned int *hashIndices,
                                       unsigned int *bucketStart,
                                       unsigned int *bucketEnd,
                                       double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &pi = d_particles[idx];

  float3 pressureForce = make_float3(0.0f, 0.0f, 0.0f);
  float3 viscosityForce = make_float3(0.0f, 0.0f, 0.0f);
  float3 grad_cS = make_float3(0.0f, 0.0f, 0.0f);
  float laplacian_cS = 0.0f;

  int3 cellPosition = toCellPosition(pi.position);

  /* Neighbour search using spatial hash */
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
          // TODO: implement
          if (idx == hashIndices[bucketIdx])
            continue;
          auto &pj = d_particles[hashIndices[bucketIdx]];
          if (pj.density < EPSILON)
            continue;

          auto r = pi.position - pj.position;

          float rMagnitude = length(r);
          if (rMagnitude < EPSILON)
            continue;

          pressureForce += (-1) * normalize(r) * pj.mass *
                           (pi.pressure + pj.pressure) / (2.0f * pj.density) *
                           gradientSpikyKernelDevice(r, SMOOTHING_RADIUS);
          viscosityForce += pj.mass * (pj.velocity - pi.velocity) / pj.density *
                            laplacianViscosityKernelDevice(r, SMOOTHING_RADIUS);
          float3 gradW = gradientPoly6Device(r, SMOOTHING_RADIUS);
          grad_cS += (pj.mass / pj.density) * gradW;

          float laplacianW = laplacianPoly6Device(r, SMOOTHING_RADIUS);
          laplacian_cS += (pj.mass / pj.density) * laplacianW;
        }
      }
    }
  }
  viscosityForce *= VISCOSITY_COEFFICIENT;

  float normGrad = length(grad_cS);

  float3 acceleration = make_float3(0.0f, GRAVITATIONAL_ACCELERATION, 0.0f);
  acceleration += (pressureForce + viscosityForce) / pi.mass;

  if (normGrad > SURFACE_NORMAL_THRESHOLD) {
    float curvature = -laplacian_cS / normGrad;

    float3 surfaceTensionForce =
        SURFACE_TENSION_COEFFICIENT * curvature * grad_cS;

    acceleration += surfaceTensionForce / pi.mass;
  }

  auto deltaVelocity = acceleration * static_cast<float>(deltaTime);
  pi.velocity += deltaVelocity;
}

__global__ void slime::g_computeWallConstraint(Particle *d_particles,
                                               double deltaTime) {
  /* Handling Collision */
  /* Simulation Space: x, y, z in [-1, 1] */

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;

  auto &i = d_particles[idx];
  const float COLLISION_DAMPING = 0.9f;
  const float BOUNDARY_LIMIT = 1.00f;
  const float BOUNDARY_LIMIT_RELAXED = 1.3f;

  const float3 offsetFromCenter = i.position - make_float3(0.0f, 0.0f, 0.0f);

  if (fabsf(offsetFromCenter.x) > BOUNDARY_LIMIT_RELAXED) {
    i.position.x = (i.position.x > 0) ? BOUNDARY_LIMIT : -BOUNDARY_LIMIT;
    i.velocity.x = -i.velocity.x * COLLISION_DAMPING;
  }
  if (fabsf(offsetFromCenter.y) > BOUNDARY_LIMIT_RELAXED) {
    i.position.y = (i.position.y > 0) ? BOUNDARY_LIMIT : -BOUNDARY_LIMIT;
    i.velocity.y = -i.velocity.y * COLLISION_DAMPING;
  }
  if (fabsf(offsetFromCenter.z) > BOUNDARY_LIMIT_RELAXED) {
    i.position.z = (i.position.z > 0) ? BOUNDARY_LIMIT : -BOUNDARY_LIMIT;
    i.velocity.z = -i.velocity.z * COLLISION_DAMPING;
  }

  // Optional: Debug print to monitor particle position (remove after debugging)
  // printf("Particle %d pos: (%f, %f, %f)\n", idx, i.position.x, i.position.y,
  // i.position.z);
}

__global__ void slime::g_computePosition(Particle *d_particles,
                                         double deltaTime) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = d_particles[idx];
  i.position += i.velocity * static_cast<float>(deltaTime);

  // if (idx % int(NUM_PARTICLES / 5) == 0) {
  //   printf("v %f %f %f\n", i.velocity.x, i.velocity.y, i.velocity.z);
  //   printf("p %f %f %f\n", i.position.x, i.position.y, i.position.z);
  // }
}

__global__ void slime::g_copyPositionToVBO(float *d_positions,
                                           Particle *d_particles) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= NUM_PARTICLES)
    return;
  auto &i = d_particles[idx];
  // printf("%f %f %f\n", i.position.x, i.position.y, i.position.z);
  d_positions[3 * idx] = i.position.x;
  d_positions[3 * idx + 1] = i.position.y;
  d_positions[3 * idx + 2] = i.position.z;
}