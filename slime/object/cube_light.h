#ifndef CUBE_LIGHT_H
#define CUBE_LIGHT_H

#include <hikari/vector/vector.h>
#include <hikari/renderer/shader.h>
#include "object.h"
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace hikari
{
    class CubeLight : public Object
    {
    public:
        CubeLight();
        CubeLight(float initX, float initY, float initZ);
        ~CubeLight();
        void setup();
        void render();
        void clear();
        void updateView(glm::mat4 _view);
        void updateProjection(glm::mat4 _projection);
        void updateCameraPos(glm::vec3 _cameraPos);
    };

}

#endif