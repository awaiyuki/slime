#include "slime.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <slime/renderer/shader.h>
#include <stb_image.h>

using namespace slime;
using namespace std;

Slime::Slime() {
  initX = 0.0f;
  initY = 0.0f;
  initZ = 0.0f;
}

Slime::Slime(float _initX, float _initY, float _initZ) {
  initX = _initX;
  initY = _initY;
  initZ = _initZ;
}

Slime::~Slime() {}

void Slime::setup() {
  cout << "setup Slime" << endl;
  this->shader = new Shader("./shaders/slime.vert", "./shaders/slime.frag");
  this->sphSimulator = make_unique<SPHSimulator>();
  shader->use();

  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);

  glBufferData(GL_ARRAY_BUFFER, 1000000 * sizeof(float), nullptr,
               GL_DYNAMIC_DRAW);

  /* position attribute */
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
}

void Slime::render(double deltaTime) {
  //   cout << "render Slime" << endl;

  /* SPH Simulation */

  updateParticles(deltaTime);
  updateScalarField();

  const vector<MarchingCubes::Triangle> &triangles =
      sphSimulator->extractSurface();

  const int32_t vertexCount = 9 * triangles.size();
  unique_ptr<float[]> triangleData(new float[vertexCount]);

  for (uint32_t i = 0; i < triangles.size(); i++) {
    triangleData[i] = triangles[i].v1[0];
    triangleData[i + 1] = triangles[i].v1[1];
    triangleData[i + 2] = triangles[i].v1[2];
    triangleData[i + 3] = triangles[i].v2[0];
    triangleData[i + 4] = triangles[i].v2[1];
    triangleData[i + 5] = triangles[i].v2[2];
    triangleData[i + 6] = triangles[i].v3[0];
    triangleData[i + 7] = triangles[i].v3[1];
    triangleData[i + 8] = triangles[i].v3[2];
  }
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * vertexCount,
                  triangleData.get());

  /* Transform */
  shader->use();

  glm::mat4 model = glm::mat4(1.0f);

  model = glm::scale(model, glm::vec3(0.5f, 0.5f, 0.5f));
  model = glm::translate(model, glm::vec3(initX, initY, initZ));

  glm::mat3 normalMatrix = glm::mat3(glm::transpose(glm::inverse(model)));

  glm::vec3 lightPos(1.1f, 3.0f, 2.0f);
  glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
  glm::vec3 objectColor(0.1f, 1.0f, 0.3f);
  shader->setVec3("lightPos", lightPos);
  shader->setVec3("viewPos", cameraPosition);
  shader->setVec3("lightColor", lightColor);
  shader->setVec3("objectColor", objectColor);
  shader->setFloat("time", glfwGetTime());

  int modelLoc = glGetUniformLocation(shader->getID(), "model");
  glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
  int normalMatrixLoc = glGetUniformLocation(shader->getID(), "normalMatrix");
  glUniformMatrix3fv(normalMatrixLoc, 1, GL_FALSE,
                     glm::value_ptr(normalMatrix));
  int viewLoc = glGetUniformLocation(shader->getID(), "view");
  glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
  int projectionLoc = glGetUniformLocation(shader->getID(), "projection");
  glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

  /* Draw */
  glBindVertexArray(VAO);

  // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  glDrawArrays(GL_TRIANGLES, 0, vertexCount);

  // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void Slime::clear() {
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  // glDeleteBuffers(1, &EBO);
}

void Slime::updateView(glm::mat4 _view) { this->view = _view; }

void Slime::updateProjection(glm::mat4 _projection) {
  this->projection = _projection;
}

void Slime::updateCameraPosition(glm::vec3 _cameraPosition) {
  this->cameraPosition = _cameraPosition;
}
