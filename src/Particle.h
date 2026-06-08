// Particle: CPU-side particle data used only for scene initialisation.
// At runtime all particle state lives in the GPU SSBO (see GPUSimulator).
#pragma once
#include <glm/glm.hpp>

struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 force;
    float     density;
    float     pressure;
    float     mass;

    Particle()
        : position(0.0f), velocity(0.0f), force(0.0f)
        , density(0.0f), pressure(0.0f), mass(0.02f) {}

    explicit Particle(const glm::vec3& pos, float m = 0.02f)
        : position(pos), velocity(0.0f), force(0.0f)
        , density(0.0f), pressure(0.0f), mass(m) {}
};
