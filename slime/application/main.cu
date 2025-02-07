

#include <slime/engine/engine.h>
#include <slime/world_object/world_object.h>
#include <slime/world_object/plane.h>
#include <slime/world_object/slime.cuh>

using namespace std;

int main() {
  slime::Engine engine;

  /* Need to make objects detect lights */
  slime::Plane plane(1.0f, -1.0f, 1.0f, 32);

  slime::Slime slime(0.0f, 0.0f, 0.0f, "point");

  engine.registerWorldObject(&plane);
  engine.registerWorldObject(&slime);

  engine.run();

  return 0;
}