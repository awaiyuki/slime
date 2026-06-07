#ifndef SLIME_PORTABLE_H
#define SLIME_PORTABLE_H

#include "world_object.h"
#include <slime/renderer/portable_marching_cubes.h>
#include <slime/simulation/portable_sph_simulator.h>

namespace slime {

class SlimePortable : public WorldObject {
public:
  SlimePortable();
  ~SlimePortable();

  void setup() override;
  void updateSimulation(double deltaTime) override;
  void render(double deltaTime) override;
  void clear() override;
  void updateView(glm::mat4 view) override;
  void updateProjection(glm::mat4 projection) override;
  void updateCameraPosition(glm::vec3 cameraPosition) override;

private:
  PortableSPHSimulator simulator_;
  PortableMarchingCubes surface_;
  std::size_t vertexCount_ = 0;
  int surfaceUpdateCounter_ = 0;
};

} // namespace slime

#endif
