#ifndef ENGINE_H
#define ENGINE_H

#include <slime/renderer/renderer.h>
#include <slime/world_object/world_object.h>

namespace slime {

class Engine {
private:
  std::unique_ptr<Renderer> renderer;

public:
  Engine();
  ~Engine();
  void init();
  void run();
  void registerWorldObject(WorldObject *object);
};

} // namespace slime

#endif ENGINE_H