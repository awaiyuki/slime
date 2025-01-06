#ifndef SLIME_H
#define SLIME_H

#include <slime/renderer/shader.h>
#include "world_object.h"
#include <vector>
#include <memory>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <slime/simulation/sph_simulator.cuh>
#include <slime/renderer/marching_cubes.cuh>
#include <glad/gl.h>
#include <GLFW/glfw3.h>

namespace slime {
struct Edge {
  glm::vec3 p1, p2;
};

class Slime : public WorldObject {

public:
  Slime(const std::string &renderMode = "cube");
  Slime(float initX, float initY, float initZ,
        const std::string &renderMode = "cube");
  ~Slime();
  void setup();
  void render(double deltaTime);
  void clear();
  void updateView(glm::mat4 _view);
  void updateProjection(glm::mat4 _projection);
  void updateCameraPosition(glm::vec3 _cameraPosition);

  /* Needed to implement Marching Cubes  */
  void initMarchingCubes();
  void march();

private:
  std::unique_ptr<SPHSimulator> sphSimulator;
  std::string renderMode;
  const double fixedTimeStep = 0.016; // 고정된 시간 간격 (60 FPS 기준)
  double accumulator = 0.0;
  double previousTime = glfwGetTime();
};
} // namespace slime

#endif SLIME_H