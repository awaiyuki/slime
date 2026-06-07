#include "portable_marching_cubes.h"

#include "marching_cubes_tables.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace {

constexpr float kDomainMinimum = -1.12f;
constexpr float kDomainMaximum = 1.12f;
constexpr float kBlobRadius = 0.13f;
constexpr float kSurfaceLevel = 0.85f;
constexpr float kEpsilon = 1e-6f;

constexpr std::array<glm::ivec3, 8> kCornerOffsets = {
    glm::ivec3(0, 0, 0), glm::ivec3(1, 0, 0), glm::ivec3(1, 1, 0),
    glm::ivec3(0, 1, 0), glm::ivec3(0, 0, 1), glm::ivec3(1, 0, 1),
    glm::ivec3(1, 1, 1), glm::ivec3(0, 1, 1)};

float compactBlob(float distanceSquared) {
  const float radiusSquared = kBlobRadius * kBlobRadius;
  if (distanceSquared >= radiusSquared)
    return 0.0f;
  const float term = 1.0f - distanceSquared / radiusSquared;
  return term * term * term;
}

} // namespace

namespace slime {

void PortableMarchingCubes::rebuild(
    const std::vector<glm::vec3> &particles) {
  const float spacing =
      (kDomainMaximum - kDomainMinimum) / static_cast<float>(gridSize_ - 1);
  scalarField_.assign(gridSize_ * gridSize_ * gridSize_, 0.0f);
  vertices_.clear();
  vertices_.reserve(gridSize_ * gridSize_ * 18);

  const auto index = [](int x, int y, int z) {
    return z * gridSize_ * gridSize_ + y * gridSize_ + x;
  };
  const auto position = [spacing](const glm::ivec3 &gridPosition) {
    return glm::vec3(kDomainMinimum) +
           glm::vec3(gridPosition) * spacing;
  };
  const auto gradientAt = [&](const glm::vec3 &worldPosition) {
    const glm::ivec3 gridPosition =
        glm::clamp(glm::ivec3(glm::round(
                       (worldPosition - glm::vec3(kDomainMinimum)) / spacing)),
                   glm::ivec3(1), glm::ivec3(gridSize_ - 2));
    const float dx =
        scalarField_[index(gridPosition.x + 1, gridPosition.y, gridPosition.z)] -
        scalarField_[index(gridPosition.x - 1, gridPosition.y, gridPosition.z)];
    const float dy =
        scalarField_[index(gridPosition.x, gridPosition.y + 1, gridPosition.z)] -
        scalarField_[index(gridPosition.x, gridPosition.y - 1, gridPosition.z)];
    const float dz =
        scalarField_[index(gridPosition.x, gridPosition.y, gridPosition.z + 1)] -
        scalarField_[index(gridPosition.x, gridPosition.y, gridPosition.z - 1)];
    const glm::vec3 gradient(dx, dy, dz);
    return glm::dot(gradient, gradient) > kEpsilon
               ? -glm::normalize(gradient)
               : glm::vec3(0.0f, 1.0f, 0.0f);
  };

  const int influence = static_cast<int>(std::ceil(kBlobRadius / spacing));
  for (const auto &particle : particles) {
    const glm::ivec3 center =
        glm::ivec3(glm::round((particle - glm::vec3(kDomainMinimum)) / spacing));
    for (int z = std::max(0, center.z - influence);
         z <= std::min(gridSize_ - 1, center.z + influence); ++z) {
      for (int y = std::max(0, center.y - influence);
           y <= std::min(gridSize_ - 1, center.y + influence); ++y) {
        for (int x = std::max(0, center.x - influence);
             x <= std::min(gridSize_ - 1, center.x + influence); ++x) {
          const glm::vec3 offset = position(glm::ivec3(x, y, z)) - particle;
          scalarField_[index(x, y, z)] += compactBlob(glm::dot(offset, offset));
        }
      }
    }
  }

  std::vector<float> smoothedField(scalarField_.size());
  for (int pass = 0; pass < 2; ++pass) {
    smoothedField = scalarField_;
    for (int z = 1; z < gridSize_ - 1; ++z) {
      for (int y = 1; y < gridSize_ - 1; ++y) {
        for (int x = 1; x < gridSize_ - 1; ++x) {
          const float neighborAverage =
              (scalarField_[index(x - 1, y, z)] +
               scalarField_[index(x + 1, y, z)] +
               scalarField_[index(x, y - 1, z)] +
               scalarField_[index(x, y + 1, z)] +
               scalarField_[index(x, y, z - 1)] +
               scalarField_[index(x, y, z + 1)]) /
              6.0f;
          smoothedField[index(x, y, z)] =
              0.55f * scalarField_[index(x, y, z)] + 0.45f * neighborAverage;
        }
      }
    }
    scalarField_.swap(smoothedField);
  }

  for (int z = 0; z < gridSize_ - 1; ++z) {
    for (int y = 0; y < gridSize_ - 1; ++y) {
      for (int x = 0; x < gridSize_ - 1; ++x) {
        const glm::ivec3 cell(x, y, z);
        std::array<glm::ivec3, 8> corners;
        std::array<float, 8> values;
        int tableKey = 0;
        for (int corner = 0; corner < 8; ++corner) {
          corners[corner] = cell + kCornerOffsets[corner];
          values[corner] = scalarField_[index(
              corners[corner].x, corners[corner].y, corners[corner].z)];
          if (values[corner] > kSurfaceLevel)
            tableKey |= 1 << corner;
        }

        const int *edges = MarchingCubesTables::triangulation[tableKey];
        for (int triangle = 0; triangle < 16 && edges[triangle] != -1;
             triangle += 3) {
          std::array<glm::vec3, 3> trianglePositions;
          for (int vertex = 0; vertex < 3; ++vertex) {
            const int edge = edges[triangle + vertex];
            const int cornerA = MarchingCubesTables::cornerIndexFromEdge[edge][0];
            const int cornerB = MarchingCubesTables::cornerIndexFromEdge[edge][1];
            const float valueA = values[cornerA];
            const float valueB = values[cornerB];
            const float difference = valueB - valueA;
            const float amount =
                std::abs(difference) > kEpsilon
                    ? std::clamp((kSurfaceLevel - valueA) / difference, 0.0f,
                                 1.0f)
                    : 0.5f;
            trianglePositions[vertex] =
                glm::mix(position(corners[cornerA]),
                         position(corners[cornerB]), amount);
          }

          const glm::vec3 faceNormal =
              glm::cross(trianglePositions[1] - trianglePositions[0],
                         trianglePositions[2] - trianglePositions[0]);
          if (glm::dot(faceNormal, faceNormal) < kEpsilon)
            continue;
          const glm::vec3 expectedNormal =
              gradientAt((trianglePositions[0] + trianglePositions[1] +
                          trianglePositions[2]) /
                         3.0f);
          if (glm::dot(faceNormal, expectedNormal) < 0.0f)
            std::swap(trianglePositions[1], trianglePositions[2]);
          for (const auto &trianglePosition : trianglePositions)
            vertices_.push_back({trianglePosition, gradientAt(trianglePosition)});
        }
      }
    }
  }
}

const std::vector<SurfaceVertex> &PortableMarchingCubes::vertices() const {
  return vertices_;
}

} // namespace slime
