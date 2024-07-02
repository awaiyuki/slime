#include "camera.h"

#include <glm/gtc/matrix_transform.hpp>

using namespace slime;
using namespace std;

Camera::Camera(GLFWwindow *_window, int _windowWidth, int _windowHeight)
    : window(_window), windowWidth(_windowWidth), windowHeight(_windowHeight) {
  position = glm::vec3(0.0f, 0.0f, 3.0f);
  target = glm::vec3(0.0f, 0.0f, 0.0f);
  direction = glm::normalize(position - target);

  up = glm::vec3(0.0f, 1.0f, 0.0f);
  cameraRight = glm::normalize(glm::cross(up, direction));
  cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
  cameraUp = glm::cross(direction, cameraRight);

  fov = 60.0f;
  yaw = -90.0f;
  pitch = 0.0f;
  lastX = 400, lastY = 300;

  near = 0.1f;
  far = 100.0f;
  rotateSensitivity = 0.1f;
  fovMinimum = 10.0f;
  fovMaximum = 60.0f;
  pitchMinimum = -90.0f;
  pitchMaximum = 90.0f;
  speedCoefficient = 2.0f;

  cursorMoved = false;
}

Camera::~Camera() {}

void Camera::updateView() {
  view = glm::lookAt(position, position + cameraFront, cameraUp);
}

void Camera::updateProjection() {
  projection = glm::perspective(
      glm::radians(fov), float(windowWidth) / float(windowHeight), near, far);
}

void Camera::cursorPosEventHandler(double xpos, double ypos) {
  if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL)
    return;

  float positionX = static_cast<float>(xpos);
  float positionY = static_cast<float>(ypos);

  if (!cursorMoved) {
    lastX = positionX;
    lastY = positionY;
    cursorMoved = true;
  }

  float deltaX = positionX - lastX;
  float deltaY = lastY - positionY;
  lastX = positionX;
  lastY = positionY;

  deltaX *= rotateSensitivity;
  deltaY *= rotateSensitivity;

  yaw += deltaX;
  pitch += deltaY;

  if (pitch < pitchMinimum)
    pitch = pitchMinimum;
  if (pitch > pitchMaximum)
    pitch = pitchMaximum;

  glm::vec3 newCameraFront;
  newCameraFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
  newCameraFront.y = sin(glm::radians(pitch));
  newCameraFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
  cameraFront = glm::normalize(newCameraFront);
};

void Camera::scrollEventHandler(double xoffset, double yoffset) {
  fov -= static_cast<float>(yoffset);
  if (fov < fovMinimum)
    fov = fovMinimum;
  if (fov > fovMaximum)
    fov = fovMaximum;
}

void Camera::handleMovement(double deltaTime) {
  if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL)
    return;

  float speed = speedCoefficient * static_cast<float>(deltaTime);

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    position += speed * cameraFront;
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    position -= speed * cameraFront;
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    position -= glm::normalize(glm::cross(cameraFront, cameraUp)) * speed;
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    position += glm::normalize(glm::cross(cameraFront, cameraUp)) * speed;
}
