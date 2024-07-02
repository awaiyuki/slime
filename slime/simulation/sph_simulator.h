#ifndef SPH_SIMULATOR_H
#define SPH_SIMULATOR_H

#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include "marching_cubes.h"

namespace slime {
    
    class SPHSimulator {

    public:
        struct Particle {
            glm::vec3 position, velocity, acceleration;
            float density, pressure, mass;
            glm::vec4 color;
            float life;
        };
        SPHSimulator();
        ~SPHSimulator();

        float poly6Kernel(glm::vec3 rSqure, float h);
        float pressureKernel(glm::vec3 r, float h);
        float viscosityKernel(glm::vec3 r, float h);

        void updateParticles();
        void computeDensity();
        void computePressure();
        void computeViscosity();
        
        void initScalarField();
        void updateScalarField();

        std::vector<MarchingCubes::Triangle> extractSurface() {
            MarchingCubes marchingCubes;
            return marchingCubes.march();
        }

    private:
        static const int numParticles = 5000;
        static const int gridSize = 200;
        std::vector<std::unique_ptr<Particle>> particles;

        float densityField[gridSize][gridSize][gridSize];
        float pressureField[gridSize][gridSize][gridSize];
        float viscosityField[gridSize][gridSize][gridSize];
        float colorField[gridSize][gridSize][gridSize];
        float surfaceTensionField[gridSize][gridSize][gridSize];
    };
}
#endif SPH_SIMULATOR_H