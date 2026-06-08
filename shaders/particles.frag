// Particle fragment shader: draws each particle as a soft circular impostor.
// Color is mapped through a heatmap keyed to either speed or density.
#version 430 core

in float vSpeed;
in float vDensity;

uniform int   uColorMode   = 0;
uniform float uMaxSpeed    = 5.0;
uniform float uMaxPressure = 2000.0;

out vec4 FragColor;

vec3 heatmap(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.0, 0.0, 1.0);
    vec3 c1 = vec3(0.0, 1.0, 1.0);
    vec3 c2 = vec3(0.0, 1.0, 0.0);
    vec3 c3 = vec3(1.0, 1.0, 0.0);
    vec3 c4 = vec3(1.0, 0.0, 0.0);
    if (t < 0.25) return mix(c0, c1, t / 0.25);
    if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
    if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
    return             mix(c3, c4, (t - 0.75) / 0.25);
}

void main() {
    vec2  coord = gl_PointCoord - vec2(0.5);
    float r2    = dot(coord, coord);
    if (r2 > 0.25) discard;

    float alpha = 1.0 - smoothstep(0.20, 0.25, r2);
    float t     = (uColorMode == 0)
                ? clamp(vSpeed   / uMaxSpeed,    0.0, 1.0)
                : clamp(vDensity / uMaxPressure, 0.0, 1.0);

    vec3 color = heatmap(t);
    color += vec3((1.0 - sqrt(r2) * 2.0) * 0.15);

    FragColor = vec4(color, alpha);
}
