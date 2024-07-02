
#include "sph_simulator.h"
#include <cstring>

using namespace slime;
using namespace std;

SPHSimulator::SPHSimulator() {
    for(int i = 0; i<numParticles; i++) {
        particles.push_back(make_unique<Particle>());
    }
}

SPHSimulator::~SPHSimulator() {

}

float SPHSimulator::poly6Kernel(glm::vec3 rSquare, float h) {
    
}

float SPHSimulator::pressureKernel(glm::vec3 r, float h) {

}

float SPHSimulator::viscosityKernel(glm::vec3 r, float h) {
    
}

void SPHSimulator::updateParticles() {
    for(auto& itr : particles) {
        computeDensity();
        computePressure();
        computeViscosity();
    }
}

void SPHSimulator::computeDensity() {

}

void SPHSimulator::computePressure() {

}

void SPHSimulator::computeViscosity() {

}



void SPHSimulator::initScalarField() {
    memset(densityField, 0, sizeof(float)*gridSize*gridSize*gridSize);
    memset(pressureField, 0, sizeof(float)*gridSize*gridSize*gridSize);
    memset(viscosityField, 0, sizeof(float)*gridSize*gridSize*gridSize);
    memset(colorField, 0, sizeof(float)*gridSize*gridSize*gridSize);
    memset(surfaceTensionField, 0, sizeof(float)*gridSize*gridSize*gridSize);
}

void SPHSimulator::updateScalarField() {
    for(int i = 0; i < gridSize; i++) {
        for(int j = 0; j < gridSize; j++) {
            for(int k = 0; k < gridSize; k++) {

            }
        }
    }
}