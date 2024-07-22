
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
    computePressureForce();
    computeViscosityForce();
}

void SPHSimulator::computeDensity() {
    for(auto& i : particles) {
        i->density = 0;
        for(auto& j : particles) {
            i->density += j->mass * poly6Kernel(distance, radius);
        }
    }
}

void SPHSimulator::computePressureForce() {
    for(auto& i : particles) {
        i->pressure = GAS_CONSTANT * (i->density - REST_DENSITY);
        glm::vec3 pressureforce = glm::vec3(0.0f, 0.0f, 0.0f);
        for(auto& j : particles) {

            // handle 3D Force
            pressureforce += i->derivativeSpikyKernel(distance, radius);
        }
    }

    // update velocity
}

void SPHSimulator::computeViscosityForce() {
    for(auto& i : particles) {
        glm::vec3 viscosityForce = glm::vec3(0.0f, 0.0f, 0.0f);
        for(auto& j : particles) {
            // handle 3D Force
            viscosityForce += (j->velocity - i->velocity)/j->density * poly6Kernel(distance, radius);
        }
    }

    // update velocity
}


void SPHSimulator::initScalarField() {
    memset(densityField, 0, sizeof(float)*GRID_SIZE*GRID_SIZE*GRID_SIZE);
    memset(pressureField, 0, sizeof(float)*GRID_SIZE*GRID_SIZE*GRID_SIZE);
    memset(viscosityField, 0, sizeof(float)*GRID_SIZE*GRID_SIZE*GRID_SIZE);
    memset(colorField, 0, sizeof(float)*GRID_SIZE*GRID_SIZE*GRID_SIZE);
    memset(surfaceTensionField, 0, sizeof(float)*GRID_SIZE*GRID_SIZE*GRID_SIZE);
}

void SPHSimulator::updateScalarField() {
    for(int i = 0; i < GRID_SIZE; i++) {
        for(int j = 0; j < GRID_SIZE; j++) {
            for(int k = 0; k < GRID_SIZE; k++) {
                // use kernels to interpolate attributes of particles to provide continuity
            }
        }
    }
}