#ifndef GRASS_H
#define GRASS_H

#include <slime/vector/vector.h>
#include <slime/renderer/shader.h>
#include "object.h"
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace slime
{
    class Grass : public Object
    {

    public:
        Grass();
        Grass(float _initX, float _initY, float _initZ);
        ~Grass();
        void setup();
        void render();
        void clear();
        void updateView(glm::mat4 _view);
        void updateProjection(glm::mat4 _projection);
        void updateCameraPos(glm::vec3 _cameraPos);
    };

}

#endif