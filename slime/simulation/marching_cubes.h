#ifndef MARCHING_CUBES
#define MARCHING_CUBES

#include <vector>
#include <glm/glm.hpp>

namespace slime {
    class MarchingCubes {
    private:
        std::vector<int> edges;
        std::vector<std::vector<int>> triangulation;
    public:
        struct Triangle {
            glm::vec3 v1, v2, v3;
        };
        
        MarchingCubes();
        ~MarchingCubes();
        std::vector<Triangle> march();
    };
}
#endif MARCHING_CUBES