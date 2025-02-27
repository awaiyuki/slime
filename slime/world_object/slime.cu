#include "slime.cuh"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <slime/renderer/shader.h>
#include <stb_image.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <slime/constants/marching_cubes_constants.h>

using namespace slime;
using namespace std;

Slime::Slime(const std::string &renderMode) {
  initX = 0.0f;
  initY = 0.0f;
  initZ = 0.0f;
  this->renderMode = renderMode;
  cout << "renderMode: " << renderMode << endl;
}

Slime::Slime(float _initX, float _initY, float _initZ,
             const std::string &renderMode) {
  initX = _initX;
  initY = _initY;
  initZ = _initZ;
  this->renderMode = renderMode;
  cout << "renderMode: " << renderMode << endl;
}

Slime::~Slime() {}

void Slime::setup() {
  cout << "setup Slime" << endl;
  this->shader = new Shader("./shaders/slime.vert", "./shaders/slime.frag");
  shader->use();

  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);

  if (renderMode == "point") {
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float) * SPHSimulatorConstants::NUM_PARTICLES * 3,
                 nullptr, GL_DYNAMIC_DRAW);

  } else {
    const int gridSize = SPHSimulatorConstants::GRID_SIZE;
    const int vertexCount = gridSize * gridSize * gridSize * 15;
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertexCount * 3, nullptr,
                 GL_DYNAMIC_DRAW);
  }

  /* position attribute */
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  this->sphSimulator = make_unique<SPHSimulator>(VBO, renderMode);
  std::vector<Particle> *particles = sphSimulator->getParticlesPointer();
}

void Slime::render(double deltaTime) {
  //   cout << "render Slime" << endl;

  /* SPH Simulation */

  sphSimulator->updateParticles(deltaTime);

  if (renderMode == "point") {
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // Render with points
    const int32_t pointCount = SPHSimulatorConstants::NUM_PARTICLES;

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
    glPointSize(5.0f);
    glDrawArrays(GL_POINTS, 0, pointCount);
  } else {
    // Render with triangles
    sphSimulator->updateScalarField();

    sphSimulator->extractSurface();
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // float3 *vertices = vertexData.vertices;
    // const int vertexCount = vertexData.size;
    // cout << "vertexCount: " << vertexCount << endl;
    // const int triangleDataSize = 3 * vertexCount;
    // unique_ptr<float[]> triangleData(new float[triangleDataSize]);

    // for (int i = 0; i < vertexCount; i++) {
    //   triangleData[3 * i] = vertices[i].x;
    //   triangleData[3 * i + 1] = vertices[i].y;
    //   triangleData[3 * i + 2] = vertices[i].z;
    // }
    // glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * triangleDataSize,
    //                 triangleData.get());

    /* Transform */
    shader->use();

    glm::mat4 model = glm::mat4(1.0f);

    model = glm::scale(model, glm::vec3(1.0f, 1.0f, 1.0f));
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
    const int gridSize = SPHSimulatorConstants::GRID_SIZE;
    const int vertexCount = gridSize * gridSize * gridSize * 15;

    glDrawArrays(GL_TRIANGLES, 0, vertexCount * 3);

    // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }
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
