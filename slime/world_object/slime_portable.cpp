#include "slime_portable.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cstddef>
#include <iostream>

namespace slime {

SlimePortable::SlimePortable() {
  initX = 0.0f;
  initY = 0.0f;
  initZ = 0.0f;
}

SlimePortable::~SlimePortable() = default;

void SlimePortable::setup() {
  std::cout << "setup portable slime renderer" << std::endl;
  shader = new Shader("./shaders/slime.vert", "./shaders/slime.frag");
  shader->use();
  shader->setVec3("material.ambient", 0.1f, 0.25f, 0.45f);
  shader->setVec3("material.diffuse", 0.2f, 0.55f, 1.0f);
  shader->setVec3("material.specular", 0.8f, 0.9f, 1.0f);
  shader->setFloat("material.shininess", 64.0f);

  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SurfaceVertex),
                        reinterpret_cast<void *>(offsetof(SurfaceVertex, position)));
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SurfaceVertex),
                        reinterpret_cast<void *>(offsetof(SurfaceVertex, normal)));
  glEnableVertexAttribArray(1);
}

void SlimePortable::updateSimulation(double deltaTime) {
  simulator_.update(static_cast<float>(deltaTime));
  if (++surfaceUpdateCounter_ >= 3 || vertexCount_ == 0) {
    surfaceUpdateCounter_ = 0;
    surface_.rebuild(simulator_.positions());
    const auto &vertices = surface_.vertices();
    vertexCount_ = vertices.size();
    static bool loggedSurfaceSize = false;
    if (!loggedSurfaceSize && vertexCount_ > 0) {
      std::cout << "Marching Cubes surface: " << vertexCount_ << " vertices"
                << std::endl;
      loggedSurfaceSize = true;
    }
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(SurfaceVertex),
                 vertices.data(), GL_DYNAMIC_DRAW);
  }
}

void SlimePortable::render(double deltaTime) {
  shader->use();
  const glm::mat4 model =
      glm::translate(glm::mat4(1.0f), glm::vec3(initX, initY, initZ));
  const glm::mat3 normalMatrix =
      glm::mat3(glm::transpose(glm::inverse(model)));
  shader->setVec3("lightPos", glm::vec3(1.1f, 3.0f, 2.0f));
  shader->setVec3("viewPos", cameraPosition);
  shader->setVec3("lightColor", glm::vec3(1.0f));
  shader->setVec3("objectColor", glm::vec3(0.2f, 0.55f, 1.0f));
  shader->setFloat("time", static_cast<float>(glfwGetTime()));

  glUniformMatrix4fv(glGetUniformLocation(shader->getID(), "model"), 1,
                     GL_FALSE, glm::value_ptr(model));
  glUniformMatrix3fv(glGetUniformLocation(shader->getID(), "normalMatrix"), 1,
                     GL_FALSE, glm::value_ptr(normalMatrix));
  glUniformMatrix4fv(glGetUniformLocation(shader->getID(), "view"), 1,
                     GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(glGetUniformLocation(shader->getID(), "projection"), 1,
                     GL_FALSE, glm::value_ptr(projection));

  glBindVertexArray(VAO);
  glDisable(GL_BLEND);
  glDisable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
  glDepthFunc(GL_LESS);
  glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertexCount_));
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glDepthFunc(GL_LEQUAL);
  glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertexCount_));
  glDepthFunc(GL_LESS);
}

void SlimePortable::clear() {
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  delete shader;
  shader = nullptr;
}

void SlimePortable::updateView(glm::mat4 newView) { view = newView; }
void SlimePortable::updateProjection(glm::mat4 newProjection) {
  projection = newProjection;
}
void SlimePortable::updateCameraPosition(glm::vec3 newCameraPosition) {
  cameraPosition = newCameraPosition;
}

} // namespace slime
