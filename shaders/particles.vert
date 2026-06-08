// Particle vertex shader: reads position and velocity from the particle SSBO,
// projects to clip space, and sizes each point by distance from the camera.
#version 430 core

struct Particle {
    vec4 position;
    vec4 predictedPosition;
    vec4 velocity;
    vec4 densities;
};

layout(std430, binding = 0) readonly buffer ParticleBuffer { Particle particles[]; };

uniform mat4  uView;
uniform mat4  uProjection;
uniform float uPointSize = 10.0;
uniform int   uColorMode = 0;

out float vSpeed;
out float vDensity;

void main() {
    vec3  pos     = particles[gl_VertexID].position.xyz;
    vec3  vel     = particles[gl_VertexID].velocity.xyz;
    float density = particles[gl_VertexID].densities.x;

    gl_Position  = uProjection * uView * vec4(pos, 1.0);
    float dist   = length((uView * vec4(pos, 1.0)).xyz);
    gl_PointSize = clamp(uPointSize / dist, 2.0, 80.0);

    vSpeed   = length(vel);
    vDensity = density;
}
