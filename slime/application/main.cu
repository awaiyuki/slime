

#include <slime/engine/engine.h>
#include <slime/object/object.h>
#include <slime/object/plane.h>
#include <slime/object/slime.cuh>

using namespace std;

int main() {
  slime::Engine engine;

  /* Need to make objects detect lights */
  slime::Plane plane(1.0f, -1.0f, 1.0f, 32);

  slime::Slime slime(0.0f, 0.0f, 0.0f, "cube");

  engine.registerObject(&plane);
  engine.registerObject(&slime);

  engine.run();

  return 0;
}