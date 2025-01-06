#ifndef RENDERER_H
#define RENDERER_H

#include <glad/gl.h>
#include "shader.h"
#include "camera.h"
#include <slime/world_object/world_object.h>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <slime/world_object/world_object.h>
#include <vector>

namespace slime {
class Renderer {
public:
  Renderer();
  ~Renderer();
  void setup();
  void render();
  void clear();
  void registerWorldObject(WorldObject *object);

private:
  GLFWwindow *window;

  std::unique_ptr<Camera> camera;
  double deltaTime;

  enum Color { COLOR_BLACK, COLOR_WHITE };

  Color colorMode;

  std::vector<WorldObject *> objectPool;

  static void framebufferSizeCallback(GLFWwindow *window, int width,
                                      int height);
  static void cursorPosCallback(GLFWwindow *window, double xpos, double ypos);
  void cursorPosEventHandler(double xpos, double ypos);
  static void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                          int mods);
  void keyEventHandler(int key, int scancode, int action, int mods);
  static void scrollCallback(GLFWwindow *window, double xoffset,
                             double yoffset);
  void scrollEventHandler(double xoffset, double yoffset);
};

} // namespace slime

#endif RENDERER_H