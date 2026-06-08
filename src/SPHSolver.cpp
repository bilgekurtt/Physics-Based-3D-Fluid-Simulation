// SPHSolver: places particles in their starting positions for each scene.
// Particle mass is calibrated so a fully-packed region reaches targetDensity.
#include "SPHSolver.h"
#include <cmath>
#include <algorithm>
#include <iostream>

SPHSolver::SPHSolver(const SimParams& params) : m_params(params) {}

void SPHSolver::initDamBreak(int targetCount) {
    m_particles.clear();

    glm::vec3 bMin = m_params.boxMin;
    glm::vec3 bMax = m_params.boxMax;

    float fillX = (bMax.x - bMin.x) * 0.22f;
    float fillY = (bMax.y - bMin.y) * 0.88f;
    float fillZ = (bMax.z - bMin.z) * 0.22f;
    float fillVol = fillX * fillY * fillZ;

    float spacing = std::cbrt(fillVol / static_cast<float>(targetCount));
    spacing = std::max(spacing, 0.01f);

    float xMid = (bMin.x + bMax.x) * 0.5f;
    float zMid = (bMin.z + bMax.z) * 0.5f;
    float yMax = bMin.y + fillY;
    float particleMass = m_params.targetDensity * spacing * spacing * spacing;

    for (float x = xMid - fillX*0.5f + spacing*0.5f; x < xMid + fillX*0.5f; x += spacing)
        for (float y = bMin.y + spacing*0.5f; y < yMax; y += spacing)
            for (float z = zMid - fillZ*0.5f + spacing*0.5f; z < zMid + fillZ*0.5f; z += spacing)
                m_particles.emplace_back(glm::vec3(x, y, z), particleMass);

    std::cout << "[SPH] initDamBreak: " << m_particles.size()
              << " particles (target: " << targetCount << ")\n";
}

void SPHSolver::initStaircase(int targetCount) {
    m_particles.clear();

    glm::vec3 bMin = m_params.boxMin;
    glm::vec3 bMax = m_params.boxMax;
    float bW = bMax.x - bMin.x;
    float bH = bMax.y - bMin.y;
    float bD = bMax.z - bMin.z;

    float xStart = bMin.x + bW * 0.62f;
    float xEnd   = bMin.x + bW * 0.78f;
    float yStart = bMin.y + bH * 0.82f;
    float yEnd   = bMax.y - bH * 0.04f;
    float zStart = bMin.z + bD * 0.10f;
    float zEnd   = bMax.z - bD * 0.10f;

    float fillVol = (xEnd - xStart) * (yEnd - yStart) * (zEnd - zStart);
    float spacing = std::cbrt(fillVol / float(targetCount));
    spacing = std::max(spacing, 0.01f);

    float particleMass = m_params.targetDensity * spacing * spacing * spacing;

    for (float x = xStart + spacing*0.5f; x < xEnd; x += spacing)
        for (float y = yStart + spacing*0.5f; y < yEnd; y += spacing)
            for (float z = zStart + spacing*0.5f; z < zEnd; z += spacing)
                m_particles.emplace_back(glm::vec3(x, y, z), particleMass);

    std::cout << "[SPH] initStaircase: " << m_particles.size()
              << " particles (target: " << targetCount << ")\n";
}

void SPHSolver::initUBore(int targetCount, float junctionY) {
    m_particles.clear();

    glm::vec3 bMin = m_params.boxMin;
    glm::vec3 bMax = m_params.boxMax;
    float bW   = bMax.x - bMin.x;
    float bD   = bMax.z - bMin.z;
    float armW = bW * 0.22f;

    const float margin = 0.02f;
    float xStart =  bMin.x          + margin;
    float xEnd   = (bMin.x + armW)  - margin;
    float yStart =  junctionY      + margin;
    float yEnd   =  bMax.y         - margin;
    float zStart =  bMin.z         + bD * 0.05f;
    float zEnd   =  bMax.z         - bD * 0.05f;

    if (xEnd <= xStart || yEnd <= yStart || zEnd <= zStart) {
        std::cout << "[SPH] initUBore: spawn zone too small\n";
        return;
    }

    float fillVol = (xEnd - xStart) * (yEnd - yStart) * (zEnd - zStart);
    float spacing = std::cbrt(fillVol / float(targetCount));
    spacing = std::max(spacing, 0.01f);

    float particleMass = m_params.targetDensity * spacing * spacing * spacing;

    for (float x = xStart + spacing*0.5f; x < xEnd; x += spacing)
        for (float y = yStart + spacing*0.5f; y < yEnd; y += spacing)
            for (float z = zStart + spacing*0.5f; z < zEnd; z += spacing)
                m_particles.emplace_back(glm::vec3(x, y, z), particleMass);

    std::cout << "[SPH] initUBore: " << m_particles.size()
              << " particles (target: " << targetCount << ")\n";
}

void SPHSolver::initHourglass(int targetCount) {
    m_particles.clear();

    glm::vec3 bMin = m_params.boxMin;
    glm::vec3 bMax = m_params.boxMax;
    float bD = bMax.z - bMin.z;

    const float margin = 0.05f;
    float xStart = bMin.x + margin;
    float xEnd   = bMax.x - margin;
    float yStart = 0.10f  + margin;
    float yEnd   = bMax.y - margin;
    float zStart = bMin.z + bD * 0.05f;
    float zEnd   = bMax.z - bD * 0.05f;

    if (xEnd <= xStart || yEnd <= yStart || zEnd <= zStart) {
        std::cout << "[SPH] initHourglass: spawn zone too small\n";
        return;
    }

    float fillVol = (xEnd - xStart) * (yEnd - yStart) * (zEnd - zStart);
    float spacing = std::cbrt(fillVol / float(targetCount));
    spacing = std::max(spacing, 0.01f);

    float particleMass = m_params.targetDensity * spacing * spacing * spacing;

    for (float x = xStart + spacing*0.5f; x < xEnd; x += spacing)
        for (float y = yStart + spacing*0.5f; y < yEnd; y += spacing)
            for (float z = zStart + spacing*0.5f; z < zEnd; z += spacing)
                m_particles.emplace_back(glm::vec3(x, y, z), particleMass);

    std::cout << "[SPH] initHourglass: " << m_particles.size()
              << " particles (target: " << targetCount << ")\n";
}
