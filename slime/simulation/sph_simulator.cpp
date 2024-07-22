
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

float SPHSimulator::spikyKernel(glm::vec3 r, float h) {

}

float SPHSimulator::derivativeSpikyKernel(glm::vec3 r, float h) {

}


float SPHSimulator::viscosityKernel(glm::vec3 r, float h) {
    
}

void SPHSimulator::updateParticles() {
    computeDensity();
    computePressure();
    computeViscosity();
}

void SPHSimulator::computeDensity() {
    for(auto& i : particles) {
        for(auto& j : particles) {
            i->density = j->mass * poly6Kernel(distance, radius);
        }
    }
}

void SPHSimulator::computePressure() {
    for(auto& i : particles) {
        for(auto& j : particles) {
            i->pressure = derivativeSpikyKernel(distance, radius);
        }
    }
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
                // use kernels to interpolate attributes of particles to provide continuity
            }
        }
    }
}