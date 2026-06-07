#include <slime/engine/engine.h>
#include <slime/world_object/plane.h>
#include <slime/world_object/slime_portable.h>

int main() {
  slime::Engine engine;
  slime::Plane plane(0.0f, -1.0f, 0.0f, 32);
  slime::SlimePortable slime;

  engine.registerWorldObject(&plane);
  engine.registerWorldObject(&slime);
  engine.run();

  return 0;
}
