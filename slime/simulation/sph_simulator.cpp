
#include "sph_simulator.h"
#include <cstring>
#include <random>
#include <iostream>
#include <glm/gtc/constants.hpp>
#define PI 3.141592653589793238462643

using namespace slime;
using namespace std;

SPHSimulator::SPHSimulator() {
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(-200.0, 200.0);

  for (int i = 0; i < SPHSimulatorConstants::NUM_PARTICLES; i++) {
    auto particle = make_unique<Particle>();
    particle->position =
        glm::vec3(static_cast<float>(dis(gen)), dis(gen), dis(gen));
    particles.push_back(move(particle));
  }
}

SPHSimulator::~SPHSimulator() {}

float SPHSimulator::poly6Kernel(glm::vec3 r, float h) {
  float rMagnitude = glm::length(r);
  if (rMagnitude > h)
    return 0.0f;

  return 315.0f / (64.0f * PI * glm::pow(h, 9)) *
         glm::pow(h * h - rMagnitude * rMagnitude, 3);
}

float SPHSimulator::spikyKernel(glm::vec3 r, float h) {}

float SPHSimulator::gradientSpikyKernel(glm::vec3 r, float h) {
  float rMagnitude = glm::length(r);
  if (rMagnitude > h)
    return 0.0f;

  return -45.0f / (PI * glm::pow(h, 6)) * glm::pow(h - rMagnitude, 2);
}

float SPHSimulator::viscosityKernel(glm::vec3 r, float h) {}

float SPHSimulator::laplacianViscosityKernel(glm::vec3 r, float h) {
  float rMagnitude = glm::length(r);
  if (rMagnitude > h)
    return 0.0f;

  return 45 / (PI * glm::pow(h, 6)) * (h - rMagnitude);
}

void SPHSimulator::updateParticles(double deltaTime) {
  computeDensity();
  computePressureForce(deltaTime);
  computeViscosityForce(deltaTime);
  computeGravity(deltaTime);

  /* Update the positions of particles */
  for (auto &i : particles) {
    cout << "update particle:" << i->position.x << i->position.y
         << i->position.z << endl;
    i->position += i->velocity * static_cast<float>(deltaTime);
  }
}

void SPHSimulator::computeDensity() {
  for (auto &i : particles) {
    i->density = 0;
    for (auto &j : particles) {
      if (i == j)
        continue;

      auto r = j->position - i->position;
      i->density +=
          j->mass * poly6Kernel(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
    }
  }
}

void SPHSimulator::computePressureForce(double deltaTime) {
  for (auto &i : particles) {
    i->pressure = SPHSimulatorConstants::GAS_CONSTANT *
                  (i->density - SPHSimulatorConstants::REST_DENSITY);
  }

  for (auto &i : particles) {
    glm::vec3 pressureForce = glm::vec3(0.0f, 0.0f, 0.0f);
    for (auto &j : particles) {
      if (i == j)
        continue;

      auto r = j->position - i->position;
      pressureForce +=
          -glm::normalize(r) * j->mass * (i->pressure + j->pressure) /
          (2.0f * j->density) *
          gradientSpikyKernel(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
    }
    auto acceleration = pressureForce / i->mass;
    auto deltaVelocity = acceleration * float(deltaTime);
    i->velocity += deltaVelocity;
  }
}

void SPHSimulator::computeViscosityForce(double deltaTime) {
  for (auto &i : particles) {
    glm::vec3 viscosityForce = glm::vec3(0.0f, 0.0f, 0.0f);
    for (auto &j : particles) {
      if (i == j)
        continue;

      auto r = j->position - i->position;
      viscosityForce +=
          j->mass * (j->velocity - i->velocity) / j->density *
          laplacianViscosityKernel(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
    }
    viscosityForce *= SPHSimulatorConstants::VISCOSITY_COEFFICIENT;

    auto acceleration = viscosityForce / i->mass;
    auto deltaVelocity = acceleration * float(deltaTime);
    i->velocity += deltaVelocity;
  }
}

void SPHSimulator::computeGravity(double deltaTime) {
  /* TODO: implement this. */
}

void SPHSimulator::initScalarField() {
  memset(densityField, 0, sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);
  memset(pressureField, 0, sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);
  memset(viscosityField, 0, sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);
  memset(colorField, 0, sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);
  memset(surfaceTensionField, 0,
         sizeof(float) * GRID_SIZE * GRID_SIZE * GRID_SIZE);
}

void SPHSimulator::updateScalarField() {
  for (int x = 0; x < GRID_SIZE; x++) {
    for (int y = 0; y < GRID_SIZE; y++) {
      for (int z = 0; z < GRID_SIZE; z++) {
        float colorQuantity = 0.0f;
        for (auto &j : particles) {
          glm::vec3 r = glm::vec3(x, y, z) - j->position;
          colorQuantity +=
              j->mass * (1.0 / j->density) *
              poly6Kernel(r, SPHSimulatorConstants::SMOOTHING_RADIUS);
        }
        colorField[x][y][z] = colorQuantity;
      }
    }
  }
}

std::vector<MarchingCubes::Triangle> SPHSimulator::extractSurface() {
  MarchingCubes marchingCubes;
  return marchingCubes.march(colorField, SPHSimulatorConstants::SURFACE_LEVEL);
}