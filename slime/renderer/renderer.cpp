#include "renderer.h"
#include <iostream>
#include <cstring>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <slime/constants/renderer_constants.h>

using namespace slime;
using namespace std;

Renderer::Renderer() {
  /* Setup Window */
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  window =
      glfwCreateWindow(RendererConstants::window_width,
                       RendererConstants::window_height, "slime", NULL, NULL);
  if (window == NULL) {
    std::cout << "slime: Error: Failed to create GLFW window" << std::endl;
    glfwTerminate();
    // return -1;
  }

  glfwMakeContextCurrent(window);
  if (!gladLoadGL(glfwGetProcAddress)) {
    cout << "slime: Error: Failed to initialize OpenGL context." << endl;
    return;
  };
  // glfwSwapInterval(1);
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glEnable(GL_DEPTH_TEST);

  colorMode = COLOR_BLACK;

  /* Setup Camera */
  camera = make_unique<Camera>(window, RendererConstants::window_width,
                               RendererConstants::window_height);

  /* Setup Callbacks */
  glfwSetWindowUserPointer(window, this);
  glfwSetFramebufferSizeCallback(window, &Renderer::framebufferSizeCallback);
  glfwSetCursorPosCallback(window, &Renderer::cursorPosCallback);
  glfwSetKeyCallback(window, &Renderer::keyCallback);
  glfwSetScrollCallback(window, &Renderer::scrollCallback);
}

Renderer::~Renderer() {
  glfwDestroyWindow(window);
  cout << "slime: destroy Renderer." << endl;
}

void Renderer::setup() {

  /* Setup all objects in Object Pool */
  for (auto itr = objectPool.begin(); itr != objectPool.end(); itr++) {
    (*itr)->setup();
  }
}

void Renderer::render() {
  deltaTime = 0.0f;
  float lastTime = 0.0f;

  while (!glfwWindowShouldClose(window)) {
    if (colorMode == COLOR_WHITE)
      glClearColor(0.99f, 0.99f, 0.99f, 1.0f);
    else
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float currentTime = glfwGetTime();
    deltaTime = currentTime - lastTime;
    lastTime = currentTime;

    camera->handleMovement(deltaTime);
    camera->updateView();
    camera->updateProjection();

    /* Render all objects in Object Pool */
    for (auto itr = objectPool.begin(); itr != objectPool.end(); itr++) {
      (*itr)->updateView(
          camera->view); // Update view transform matrix of objects
      (*itr)->updateProjection(
          camera->projection); // Update projection transform matrix of objects
      (*itr)->updateCameraPosition(camera->position);
      (*itr)->render();
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
}

void Renderer::clear() {
  for (auto itr = objectPool.begin(); itr != objectPool.end(); itr++) {
    (*itr)->clear();
  }
}

void Renderer::registerObject(Object *object) { objectPool.push_back(object); }

void Renderer::framebufferSizeCallback(GLFWwindow *window, int width,
                                       int height) {
  glViewport(0, 0, width, height);
}

void Renderer::cursorPosCallback(GLFWwindow *window, double xpos, double ypos) {
  static_cast<Renderer *>(glfwGetWindowUserPointer(window))
      ->cursorPosEventHandler(xpos, ypos);
}

void Renderer::cursorPosEventHandler(double xpos, double ypos) {
  camera->cursorPosEventHandler(xpos, ypos);
}
void Renderer::keyCallback(GLFWwindow *window, int key, int scancode,
                           int action, int mods) {
  static_cast<Renderer *>(glfwGetWindowUserPointer(window))
      ->keyEvent(key, scancode, action, mods);
}

void Renderer::keyEvent(int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_C && action == GLFW_PRESS) {
    if (colorMode == COLOR_BLACK)
      colorMode = COLOR_WHITE;
    else
      colorMode = COLOR_BLACK;
  }

  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
  if (key == GLFW_KEY_F && action == GLFW_PRESS)
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void Renderer::scrollCallback(GLFWwindow *window, double xoffset,
                              double yoffset) {
  static_cast<Renderer *>(glfwGetWindowUserPointer(window))
      ->scrollEventHandler(xoffset, yoffset);
}

void Renderer::scrollEventHandler(double xoffset, double yoffset) {
  camera->scrollEventHandler(xoffset, yoffset);
}