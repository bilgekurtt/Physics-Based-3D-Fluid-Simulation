// Sphere fragment shader: Phong shading with ambient, diffuse, and specular terms.
#version 430 core

in vec3 vNormal;
in vec3 vFragPos;

uniform vec3 uLightDir;
uniform vec3 uCameraPos;

out vec4 fragColor;

void main() {
    vec3  N         = normalize(vNormal);
    vec3  L         = normalize(uLightDir);
    vec3  V         = normalize(uCameraPos - vFragPos);
    vec3  H         = normalize(L + V);
    vec3  baseColor = vec3(0.85, 0.08, 0.08);
    float diff      = max(dot(N, L), 0.0);
    float spec      = pow(max(dot(N, H), 0.0), 64.0);
    vec3  color     = 0.22 * baseColor
                    + diff * baseColor
                    + spec * vec3(1.0, 0.9, 0.9) * 0.55;
    fragColor = vec4(color, 1.0);
}
