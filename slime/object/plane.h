#ifndef PLANE_H
#define PLANE_H

#include "object.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <slime/renderer/shader.h>
#include <vector>

namespace slime {
class Plane : public Object {

public:
  Plane();
  Plane(float initX, float initY, float initZ, int planeSize);
  ~Plane();
  void setup();
  void render();
  void clear();
  void updateView(glm::mat4 _view);
  void updateProjection(glm::mat4 _projection);
  void updateCameraPosition(glm::vec3 _cameraPosition);

private:
  int planeSize;
};

} // namespace slime

#endif