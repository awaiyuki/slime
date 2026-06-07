#ifndef PORTABLE_SPH_SIMULATOR_H
#define PORTABLE_SPH_SIMULATOR_H

#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace slime {

class PortableSPHSimulator {
public:
  PortableSPHSimulator();
  ~PortableSPHSimulator();

  void update(float deltaTime);
  const std::vector<glm::vec3> &positions() const;

private:
  std::vector<glm::vec3> positions_;

#ifdef __APPLE__
  struct MetalState;
  std::unique_ptr<MetalState> metal_;
#elif defined(SLIME_USE_CUDA)
  struct CudaState;
  std::unique_ptr<CudaState> cuda_;
#endif
};

} // namespace slime

#endif
