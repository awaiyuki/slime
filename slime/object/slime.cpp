#include "slime.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <slime/renderer/shader.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
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

  shader->use();
}

void Slime::render() {
  // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  // cout << "render Slime" << endl;

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
  // glDrawElements(GL_TRIANGLES, (slimeSize - 1) * (slimeSize - 1) * 2 * 3,
  // GL_UNSIGNED_INT, 0); glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
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
