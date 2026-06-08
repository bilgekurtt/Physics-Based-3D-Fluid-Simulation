// SPHSolver: CPU-side particle spawner. Computes initial particle positions and
// masses for each scene. Physics runs entirely on the GPU via GPUSimulator.
#pragma once
#include <vector>
#include "Particle.h"
#include "SimParams.h"

class SPHSolver {
public:
    explicit SPHSolver(const SimParams& params);

    void initDamBreak (int targetCount = 1000);
    void initStaircase(int targetCount = 1000);
    void initUBore    (int targetCount, float junctionY);
    void initHourglass(int targetCount = 1000);

    const std::vector<Particle>& particles() const { return m_particles; }
    SimParams& params() { return m_params; }

private:
    SimParams             m_params;
    std::vector<Particle> m_particles;
};
