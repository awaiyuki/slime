
#include "sph_simulator.h"
#include <cstring>

using namespace slime;
using namespace std;

SPHSimulator::SPHSimulator() {
    for(int i = 0; i<NUM_PARTICLES; i++) {
        particles.push_back(make_unique<Particle>());
    }
}

SPHSimulator::~SPHSimulator() {

}

float SPHSimulator::poly6Kernel(glm::vec3 r, float h) {
    
}

float SPHSimulator::spikyKernel(glm::vec3 r, float h) {

}

float SPHSimulator::gradientSpikyKernel(glm::vec3 r, float h) {

}


float SPHSimulator::viscosityKernel(glm::vec3 r, float h) {
    
}

float SPHSimulator::laplacianViscosityKernel(glm::vec3 r, float h) {
    
}

void SPHSimulator::updateParticles() {
    computeDensity();
    computePressureForce();
    computeViscosityForce();
}

void SPHSimulator::computeDensity() {
    for(auto& i : particles) {
        i->density = 0;
        for(auto& j : particles) {
            if(i == j) continue;

            auto r = j->position - i->position;
            i->density += j->mass * poly6Kernel(r, SMOOTHING_RADIUS);
        }
    }
}

void SPHSimulator::computePressureForce() {
    for(auto& i : particles) {
        i->pressure = GAS_CONSTANT * (i->density - REST_DENSITY);
    }

    for(auto& i : particles) {
        glm::vec3 pressureForce = glm::vec3(0.0f, 0.0f, 0.0f);
        for(auto& j : particles) {
            if(i == j) continue;

            auto r = j->position - i->position;
            pressureForce += -glm::normalize(r) *j->mass *(i->pressure + j->pressure)/(2.0f * j->density)* gradientSpikyKernel(r, SMOOTHING_RADIUS);
        }
        auto acceleration = pressureForce / i->mass;
        auto deltaVelocity = acceleration * deltaTime;
        i->velocity += deltaVelocity;
    }
}

void SPHSimulator::computeViscosityForce() {
    for(auto& i : particles) {
        glm::vec3 viscosityForce = glm::vec3(0.0f, 0.0f, 0.0f);
        for(auto& j : particles) {
            if(i == j) continue;

            auto r = j->position - i->position;
            viscosityForce += j->mass * (j->velocity - i->velocity) / j->density * laplacianViscosityKernel(r, SMOOTHING_RADIUS);
        }
        viscosityForce *= VISCOSITY_COEFFICIENT;
        
        auto acceleration = viscosityForce / i->mass;
        auto deltaVelocity = acceleration * deltaTime;
        i->velocity += deltaVelocity;
    }
}


void SPHSimulator::initScalarField() {
    memset(densityField, 0, sizeof(float)*GRID_SIZE*GRID_SIZE*GRID_SIZE);
    memset(pressureField, 0, sizeof(float)*GRID_SIZE*GRID_SIZE*GRID_SIZE);
    memset(viscosityField, 0, sizeof(float)*GRID_SIZE*GRID_SIZE*GRID_SIZE);
    memset(colorField, 0, sizeof(float)*GRID_SIZE*GRID_SIZE*GRID_SIZE);
    memset(surfaceTensionField, 0, sizeof(float)*GRID_SIZE*GRID_SIZE*GRID_SIZE);
}

void SPHSimulator::updateScalarField() {
    for(int x = 0; x < GRID_SIZE; x++) {
        for(int y = 0; y < GRID_SIZE; y++) {
            for(int z = 0; z < GRID_SIZE; z++) {
                float colorQuantity = 0.0f;
                for(auto& j : particles) {
                    glm::vec3 r = glm::vec3(x, y, z) - j->position;
                    colorQuantity += j->mass * (1.0 / j->density) * poly6Kernel(r, SMOOTHING_RADIUS);
                }
                colorField[x][y][z] = colorQuantity;
            }
        }
    }
}

std::vector<MarchingCubes::Triangle> SPHSimulator::extractSurface() {
    MarchingCubes marchingCubes;
    return marchingCubes.march(colorField, SURFACE_LEVEL);
}