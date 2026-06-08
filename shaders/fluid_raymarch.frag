// Ray-march fragment shader: reconstructs a world-space ray per pixel, marches
// through the 3-D density texture to find the fluid surface, then shades it with
// Fresnel reflection, specular highlights, and depth-based colour blending.
#version 430 core

in vec2 vUV;

uniform sampler3D uDensityGrid;
uniform mat4  uInvProjection;
uniform mat4  uInvView;
uniform mat4  uProjection;
uniform mat4  uView;
uniform vec3  uCameraPos;
uniform vec3  uBoxMin;
uniform vec3  uBoxMax;
uniform ivec3 uGridRes;
uniform float uIsoLevel;
uniform float uStepScale;
uniform vec3  uLightDir;
uniform vec3  uLightColor;
uniform vec3  uWaterColor;
uniform vec3  uDeepColor;
uniform float uAmbient;
uniform float uSpecular;
uniform float uShininess;
uniform float uFresnelBias;
uniform float uThicknessScale;

out vec4 fragColor;

bool rayAABB(vec3 ro, vec3 rd, out float tMin, out float tMax) {
    vec3 invRd = 1.0 / rd;
    vec3 t1    = (uBoxMin - ro) * invRd;
    vec3 t2    = (uBoxMax - ro) * invRd;
    tMin = max(max(min(t1.x,t2.x), min(t1.y,t2.y)), min(t1.z,t2.z));
    tMax = min(min(max(t1.x,t2.x), max(t1.y,t2.y)), max(t1.z,t2.z));
    return tMax > max(tMin, 0.0);
}

float sampleDensity(vec3 world) {
    vec3 uvw = (world - uBoxMin) / (uBoxMax - uBoxMin);
    return texture(uDensityGrid, uvw).r;
}

vec3 surfaceNormal(vec3 world) {
    vec3  eps = (uBoxMax - uBoxMin) / vec3(uGridRes);
    float dx  = sampleDensity(world + vec3(eps.x,0,0)) - sampleDensity(world - vec3(eps.x,0,0));
    float dy  = sampleDensity(world + vec3(0,eps.y,0)) - sampleDensity(world - vec3(0,eps.y,0));
    float dz  = sampleDensity(world + vec3(0,0,eps.z)) - sampleDensity(world - vec3(0,0,eps.z));
    vec3  g   = vec3(dx, dy, dz);
    return (dot(g,g) > 1e-12) ? normalize(-g) : vec3(0,1,0);
}

void main() {
    vec2 ndc     = vUV * 2.0 - 1.0;
    vec4 viewRay = uInvProjection * vec4(ndc, -1.0, 1.0);
    viewRay     /= viewRay.w;
    viewRay.w    = 0.0;
    vec3 rd = normalize((uInvView * viewRay).xyz);
    vec3 ro = uCameraPos;

    float tMin, tMax;
    if (!rayAABB(ro, rd, tMin, tMax)) discard;

    vec3  boxSize  = uBoxMax - uBoxMin;
    float maxDim   = max(max(boxSize.x, boxSize.y), boxSize.z);
    float voxSize  = maxDim / float(max(max(uGridRes.x, uGridRes.y), uGridRes.z));
    float stepSize = voxSize * uStepScale;
    int   maxSteps = int(length(boxSize) / stepSize) + 2;

    float t  = max(tMin, 0.001);
    float ta = 0.0, tb = 0.0;
    bool  hit = false;

    for (int i = 0; i < maxSteps && t < tMax; i++) {
        if (sampleDensity(ro + rd * t) > uIsoLevel) {
            ta = t - stepSize; tb = t; hit = true; break;
        }
        t += stepSize;
    }
    if (!hit) discard;

    for (int j = 0; j < 8; j++) {
        float tm = (ta + tb) * 0.5;
        if (sampleDensity(ro + rd * tm) > uIsoLevel) tb = tm;
        else                                          ta = tm;
    }
    vec3 hitPos = ro + rd * ((ta + tb) * 0.5);

    vec3 N = surfaceNormal(hitPos);
    if (dot(N, -rd) < 0.0) N = -N;

    vec3  V         = normalize(-rd);
    vec3  L         = normalize(uLightDir);
    vec3  H         = normalize(L + V);
    float diff      = max(dot(N, L), 0.0);
    float spec      = pow(max(dot(N, H), 0.0), uShininess);
    float fresnel   = uFresnelBias + (1.0 - uFresnelBias) * pow(1.0 - max(dot(N,V), 0.0), 5.0);
    float thickness = clamp((sampleDensity(hitPos) / max(uIsoLevel, 1.0) - 1.0) * uThicknessScale, 0.0, 1.0);

    vec3 baseColor = mix(uWaterColor, uDeepColor, thickness);
    vec3 color     = uAmbient * baseColor
                   + (1.0 - fresnel) * diff * baseColor * uLightColor
                   + fresnel         * spec * uSpecular * uLightColor;

    vec4 clip    = uProjection * uView * vec4(hitPos, 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;

    fragColor = vec4(color, clamp(0.82 + fresnel * 0.18, 0.82, 0.99));
}
