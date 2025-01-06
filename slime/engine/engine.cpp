#include "engine.h"
#include <iostream>

using namespace slime;
using namespace std;

Engine::Engine() { this->renderer = make_unique<Renderer>(); }

Engine::~Engine() {
  glfwTerminate();
  cout << "slime: Destoying the Engine." << endl;
}

void Engine::init() {}

void Engine::run() {
  this->renderer->setup();
  this->renderer->render();
  this->renderer->clear();
}

void Engine::registerWorldObject(WorldObject *object) {
  this->renderer->registerWorldObject(object);
}