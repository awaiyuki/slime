#ifndef SLIME_H
#define SLIME_H

#include <slime/renderer/shader.h>
#include "object.h"
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <slime/simulation/sph_simulator.h>
#include <slime/simulation/marching_cubes.h>

namespace slime {
struct Edge {
  glm::vec3 p1, p2;
};

class Slime : public Object {

public:
  Slime();
  Slime(float initX, float initY, float initZ);
  ~Slime();
  void setup();
  void render();
  void clear();
  void updateView(glm::mat4 _view);
  void updateProjection(glm::mat4 _projection);
  void updateCameraPosition(glm::vec3 _cameraPosition);

  /* Needed to implement Marching Cubes  */
  void initMarchingCubes();
  void march();

private:
  std::unique_ptr<SPHSimulator> sphSimulator;
};
} // namespace slime

#endif SLIME_H