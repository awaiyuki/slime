#ifndef OBJECT_H
#define OBJECT_H

#include <glad/gl.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <slime/renderer/shader.h>

namespace slime {
class Object {
public:
  Object();
  ~Object();
  virtual void setup() = 0;
  virtual void render() = 0;
  virtual void clear() = 0;
  virtual void updateView(glm::mat4 _view) = 0;
  virtual void updateProjection(glm::mat4 _projection) = 0;
  virtual void updateCameraPosition(glm::vec3 _cameraPosition) = 0;

protected:
  unsigned int VAO;
  unsigned int VBO;
  unsigned int EBO;
  unsigned int texture0;
  unsigned int texture1;
  glm::mat4 view;
  glm::mat4 projection;
  glm::vec3 cameraPosition;

  float initX, initY, initZ;
  double mass;
  Shader *shader;
};

} // namespace slime

#endif