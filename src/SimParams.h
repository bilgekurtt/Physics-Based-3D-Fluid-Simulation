// SimParams: simulation parameter block shared between the CPU solver and the GPU UBO.
// Default values are tuned for the massless-kernel GPU pipeline.
#pragma once
#include <glm/glm.hpp>

struct SimParams {
    float smoothingRadius  = 0.2f;
    float smoothingRadius2 = 0.04f;

    float targetDensity          = 630.0f;
    float pressureMultiplier     = 288.0f;
    float nearPressureMultiplier = 2.15f;
    float viscosityStrength      = 0.0f;
    float collisionDamping       = 0.95f;
    float maxVelocity            = 40.0f;

    glm::vec3 gravity = glm::vec3(0.0f, -10.0f, 0.0f);

    float timeStep         = 0.003f;
    float predictionFactor = 1.0f / 120.0f;

    glm::vec3 boxMin = glm::vec3(-0.6f, 0.0f, -0.6f);
    glm::vec3 boxMax = glm::vec3( 0.6f, 1.8f,  0.6f);

    void updateDerivedValues() {
        smoothingRadius2 = smoothingRadius * smoothingRadius;
    }
};
