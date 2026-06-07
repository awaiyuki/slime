#ifndef PORTABLE_SPH_SIMULATOR_H
#define PORTABLE_SPH_SIMULATOR_H

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

namespace slime {

struct PortableParticle {
  glm::vec3 position{0.0f};
  glm::vec3 velocity{0.0f};
  float density = 1.0f;
  float pressure = 0.0f;
};

class PortableSPHSimulator {
public:
  PortableSPHSimulator();
  ~PortableSPHSimulator();

  void update(float deltaTime);
  const std::vector<glm::vec3> &positions() const;

private:
  using CellKey = std::int64_t;

  CellKey cellKey(const glm::ivec3 &cell) const;
  glm::ivec3 cellFor(const glm::vec3 &position) const;
  void rebuildGrid();
  void computeDensities();
  void computeVelocities(float deltaTime);
  void integrate(float deltaTime);

  std::vector<PortableParticle> particles_;
  std::vector<glm::vec3> nextVelocities_;
  std::vector<glm::vec3> positions_;
  std::unordered_map<CellKey, std::vector<std::uint32_t>> grid_;

#ifdef __APPLE__
  struct MetalState;
  std::unique_ptr<MetalState> metal_;
#endif
};

} // namespace slime

#endif
