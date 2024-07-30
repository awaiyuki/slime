#ifndef CAMERA_H
#define CAMERA_H

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace slime {
class Camera {
public:
  glm::mat4 view;
  glm::mat4 projection;
  glm::vec3 position;

  Camera(GLFWwindow *_window, int _windowWidth, int _windowHeight);
  ~Camera();

  void updateView();
  void updateProjection();

  void cursorPosEventHandler(double xpos, double ypos);
  void scrollEventHandler(double xoffset, double yoffset);
  void handleMovement(double deltaTime);

private:
  GLFWwindow *window;
  int windowWidth, windowHeight;

  glm::vec3 target;
  glm::vec3 direction;
  glm::vec3 up;
  glm::vec3 cameraRight;
  glm::vec3 cameraFront;
  glm::vec3 cameraUp;
  float fov;
  float yaw;
  float pitch;
  float lastX, lastY;

  float near;
  float far;
  float rotateSensitivity;
  float fovMinimum;
  float fovMaximum;
  float pitchMinimum;
  float pitchMaximum;
  float speedCoefficient;

  bool cursorMoved;
};
} // namespace slime

#endif