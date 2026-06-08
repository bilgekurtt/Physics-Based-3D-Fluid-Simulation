// GPUSimulator: owns the particle SSBO, SimUniforms UBO, and the 5-pass compute
// shader pipeline (external forces → density → pressure → viscosity → integrate).
// The CPU never reads or writes particle data between frames.
#pragma once
#include <glad/glad.h>
#include <vector>
#include <glm/glm.hpp>
#include "Particle.h"
#include "SimParams.h"
#include "Shader.h"

struct alignas(16) SimUniforms {
    glm::vec4 gravity;
    glm::vec4 boxMin;
    glm::vec4 boxMax;
    float smoothingRadius;
    float smoothingRadius2;
    float targetDensity;
    float pressureMultiplier;
    float nearPressureMultiplier;
    float viscosityStrength;
    float timeStep;
    float collisionDamping;
    int   numParticles;
    float predictionFactor;
    float maxVelocity;
    float _pad0, _pad1, _pad2, _pad3, _pad4;
};

struct GPUParticle {
    glm::vec4 position;
    glm::vec4 predictedPosition;
    glm::vec4 velocity;
    glm::vec4 densities;
};

class GPUSimulator {
public:
    explicit GPUSimulator(const std::string& shaderBase)
        : m_shaderBase(shaderBase) {}

    bool init(const std::vector<Particle>& particles, const SimParams& params) {
        m_csExternal  = Shader::compute((m_shaderBase + "sph_external.comp").c_str());
        m_csDensity   = Shader::compute((m_shaderBase + "sph_density.comp").c_str());
        m_csForces    = Shader::compute((m_shaderBase + "sph_forces.comp").c_str());
        m_csViscosity = Shader::compute((m_shaderBase + "sph_viscosity.comp").c_str());
        m_csIntegrate = Shader::compute((m_shaderBase + "sph_integrate.comp").c_str());

        if (!m_csExternal.valid() || !m_csDensity.valid()  ||
            !m_csForces.valid()   || !m_csViscosity.valid() || !m_csIntegrate.valid())
            return false;

        createUBO(params, (int)particles.size());
        uploadParticles(particles);
        return true;
    }

    void step(const SimParams& params, int numParticles, int substeps = 3) {
        if (numParticles == 0) return;
        GLuint groups = ((GLuint)numParticles + 255u) / 256u;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboParticles);

        for (int s = 0; s < substeps; s++) {
            updateUBO(params, numParticles);

            m_csExternal.dispatch(groups);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            m_csDensity.dispatch(groups);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            m_csForces.dispatch(groups);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            m_csViscosity.dispatch(groups);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            m_csIntegrate.dispatch(groups);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }

    void uploadParticles(const std::vector<Particle>& particles) {
        std::vector<GPUParticle> buf(particles.size());
        for (size_t i = 0; i < particles.size(); i++) {
            buf[i].position          = glm::vec4(particles[i].position, 0.0f);
            buf[i].predictedPosition = glm::vec4(particles[i].position, 0.0f);
            buf[i].velocity          = glm::vec4(particles[i].velocity, 0.0f);
            buf[i].densities         = glm::vec4(0.0f);
        }
        if (!m_ssboParticles) glGenBuffers(1, &m_ssboParticles);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboParticles);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
            buf.size() * sizeof(GPUParticle), buf.data(), GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboParticles);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    GLuint particleSSBO() const { return m_ssboParticles; }

    void syncUBO(const SimParams& p, int n) { updateUBO(p, n); }

    ~GPUSimulator() {
        if (m_ssboParticles) glDeleteBuffers(1, &m_ssboParticles);
        if (m_ubo)           glDeleteBuffers(1, &m_ubo);
    }

    GPUSimulator(const GPUSimulator&)            = delete;
    GPUSimulator& operator=(const GPUSimulator&) = delete;

private:
    std::string m_shaderBase;
    Shader      m_csExternal, m_csDensity, m_csForces, m_csViscosity, m_csIntegrate;
    GLuint      m_ssboParticles = 0;
    GLuint      m_ubo           = 0;

    void createUBO(const SimParams& p, int n) {
        if (!m_ubo) glGenBuffers(1, &m_ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, m_ubo);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(SimUniforms), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
        updateUBO(p, n);
    }

    void updateUBO(const SimParams& p, int n) {
        SimUniforms u{};
        u.gravity                = glm::vec4(p.gravity, 0.0f);
        u.boxMin                 = glm::vec4(p.boxMin,  0.0f);
        u.boxMax                 = glm::vec4(p.boxMax,  0.0f);
        u.smoothingRadius        = p.smoothingRadius;
        u.smoothingRadius2       = p.smoothingRadius * p.smoothingRadius;
        u.targetDensity          = p.targetDensity;
        u.pressureMultiplier     = p.pressureMultiplier;
        u.nearPressureMultiplier = p.nearPressureMultiplier;
        u.viscosityStrength      = p.viscosityStrength;
        u.timeStep               = p.timeStep;
        u.collisionDamping       = p.collisionDamping;
        u.numParticles           = n;
        u.predictionFactor       = p.predictionFactor;
        u.maxVelocity            = p.maxVelocity;
        glBindBuffer(GL_UNIFORM_BUFFER, m_ubo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(SimUniforms), &u);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }
};
