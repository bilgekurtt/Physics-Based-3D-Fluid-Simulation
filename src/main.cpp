// main.cpp: application entry point. Creates the OpenGL window, owns the GPU
// simulation pipeline, handles camera/mouse input, draws scene wireframes and
// the gizmo, and renders the ImGui control panel each frame.
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <algorithm>
#include <cstdint>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "SimParams.h"
#include "SPHSolver.h"
#include "GPUSimulator.h"
#include "Shader.h"
#include "Camera.h"

static const int SCR_WIDTH  = 1280;
static const int SCR_HEIGHT = 720;

static SimParams        g_params;
static const SimParams  g_defaultParams;
static int              g_scene       = 0;
static int              g_targetCount = 1000;
static int              g_substeps    = 4;

static bool g_paused    = true;
static bool g_started   = false;
static bool g_doStepFwd  = false;
static bool g_doStepBack = false;
static bool g_needReset  = false;

static const int MAX_HISTORY = 60;
static std::deque<std::vector<uint8_t>> g_history;

static Camera    g_camera;
static bool      g_mouseCapture  = false;
static bool      g_middleCapture = false;
static double    g_lastMouseX = 0.0, g_lastMouseY = 0.0;
static glm::mat4 g_viewLast, g_projLast;
static int       g_fbW = SCR_WIDTH, g_fbH = SCR_HEIGHT;

static int  g_gizmoAxis     = -1;
static bool g_gizmoDragging = false;

static int   g_colorMode  = 0;
static int   g_renderMode = 0;
static float g_pointSize  = 22.0f;

static int   g_rmGridRes   = 64;
static float g_rmIsoLevel  = 550.0f;
static float g_rmStepScale = 0.10f;

static glm::vec3 g_camPos = glm::vec3(0.0f);

static glm::vec3 g_lightDir       = glm::normalize(glm::vec3(-0.4f, 0.9f,-0.3f));
static glm::vec3 g_lightColor     = glm::vec3(1.0f, 0.98f, 0.92f);
static glm::vec3 g_waterColor     = glm::vec3(0.10f, 0.40f, 0.80f);
static glm::vec3 g_deepColor      = glm::vec3(0.02f, 0.12f, 0.40f);
static float     g_ambient        = 0.500f;
static float     g_specular       = 4.5f;
static float     g_shininess      = 128.0f;
static float     g_fresnelBias    = 0.23f;
static float     g_thicknessScale = 3.36f;

static float g_uboreSpawnH   = 0.60f;
static float g_uboreTubeTopY = 2.10f;

static constexpr float HOURGLASS_GAP_HW = 0.12f;
static constexpr float HOURGLASS_THICK  = 0.10f;

static float g_hourglassAngle   = 0.0f;   // radians; rotates gravity around Z
static bool  g_hourglassRotDrag = false;
static int   g_prevScene        = -1;      // tracks last reset scene to detect changes

struct alignas(16) ColliderUniforms {
    glm::vec4 cMin[8];
    glm::vec4 cMax[8];
    int       numColliders;
    float     _p[3];
    glm::vec4 spherePosRad;
    int       sphereActive;
    int       staircaseMode;
    float     _p2[2];
};
static ColliderUniforms g_colliderData{};
static GLuint           g_colliderUBO = 0;

static bool      g_sphereActive   = false;
static bool      g_sphereDropped  = false;
static float     g_sphereRadius   = 0.15f;
static glm::vec3 g_spherePos      = glm::vec3(0.0f);
static glm::vec3 g_sphereVel      = glm::vec3(0.0f);
static bool      g_sphereDragging = false;
static float     g_sphereDragZ    = 0.0f;
static glm::vec3 g_sphereDragDelta = glm::vec3(0.0f);

static glm::vec3 sphereRestPos() {
    glm::vec3 c = (g_params.boxMin + g_params.boxMax) * 0.5f;
    c.y = g_params.boxMax.y + g_sphereRadius + 0.05f;
    return c;
}

static GPUSimulator* g_gpu          = nullptr;
static int           g_currentCount = 0;

static const char* VERT_PATH         = "../../../shaders/particles.vert";
static const char* FRAG_PATH         = "../../../shaders/particles.frag";
static const char* BOX_VERT          = "../../../shaders/box.vert";
static const char* BOX_FRAG          = "../../../shaders/box.frag";
static const char* GIZMO_VERT        = "../../../shaders/gizmo.vert";
static const char* GIZMO_FRAG        = "../../../shaders/gizmo.frag";
static const char* FULL_VERT         = "../../../shaders/fullscreen.vert";
static const char* DENSITY_GRID_COMP = "../../../shaders/density_grid.comp";
static const char* RM_FRAG           = "../../../shaders/fluid_raymarch.frag";
static const char* SPHERE_VERT       = "../../../shaders/sphere.vert";
static const char* SPHERE_FRAG       = "../../../shaders/sphere.frag";
static const std::string SHADER_BASE = "../../../shaders/";

static GLuint       g_wireVAO = 0, g_wireVBO = 0, g_wireEBO = 0;
static const GLuint s_boxIdx[24] = {0,1,1,2,2,3,3,0,4,5,5,6,6,7,7,4,0,4,1,5,2,6,3,7};

static void initWireBox() {
    glGenVertexArrays(1, &g_wireVAO);
    glGenBuffers(1, &g_wireVBO);
    glGenBuffers(1, &g_wireEBO);
    glBindVertexArray(g_wireVAO);
    glBindBuffer(GL_ARRAY_BUFFER, g_wireVBO);
    glBufferData(GL_ARRAY_BUFFER, 24*sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_wireEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(s_boxIdx), s_boxIdx, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), nullptr);
    glBindVertexArray(0);
}

static void drawWireBox(glm::vec3 mn, glm::vec3 mx) {
    float v[24] = {
        mn.x,mn.y,mn.z, mx.x,mn.y,mn.z, mx.x,mx.y,mn.z, mn.x,mx.y,mn.z,
        mn.x,mn.y,mx.z, mx.x,mn.y,mx.z, mx.x,mx.y,mx.z, mn.x,mx.y,mx.z
    };
    glBindBuffer(GL_ARRAY_BUFFER, g_wireVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(v), v);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(g_wireVAO);
    glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

static GLuint g_gizmoVAO = 0, g_gizmoVBO = 0;

static void initGizmo() {
    glGenVertexArrays(1, &g_gizmoVAO);
    glGenBuffers(1, &g_gizmoVBO);
    glBindVertexArray(g_gizmoVAO);
    glBindBuffer(GL_ARRAY_BUFFER, g_gizmoVBO);
    glBufferData(GL_ARRAY_BUFFER, 6*sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), nullptr);
    glBindVertexArray(0);
}

static glm::vec3 gizmoHandle(int axis) {
    glm::vec3 c = (g_params.boxMin + g_params.boxMax) * 0.5f;
    if (axis == 0) return { g_params.boxMax.x, c.y, c.z };
    if (axis == 1) return { c.x, g_params.boxMax.y, c.z };
    return              { c.x, c.y, g_params.boxMax.z };
}

static bool worldToScreen(glm::vec3 wp, float& sx, float& sy) {
    glm::vec4 clip = g_projLast * g_viewLast * glm::vec4(wp, 1.0f);
    if (clip.w <= 0.0f) return false;
    glm::vec2 ndc = glm::vec2(clip.x, clip.y) / clip.w;
    sx = ( ndc.x * 0.5f + 0.5f) * (float)g_fbW;
    sy = (-ndc.y * 0.5f + 0.5f) * (float)g_fbH;
    return true;
}

// Forward declaration — drawLineList is defined below, used by drawGizmo
static void drawLineList(const std::vector<glm::vec3>& pts);

// ── Hourglass gizmo helpers (used by both drawGizmo and the hit/drag logic) ──
static float hourglassGizmoRadius() {
    float rx = (g_params.boxMax.x - g_params.boxMin.x) * 0.5f;
    float ry = (g_params.boxMax.y - g_params.boxMin.y) * 0.5f;
    return glm::max(rx, ry) + 0.40f;
}
static glm::vec3 hourglassGizmoHandle() {
    glm::vec3 c = (g_params.boxMin + g_params.boxMax) * 0.5f;
    float r = hourglassGizmoRadius();
    return c + glm::vec3(r * sinf(g_hourglassAngle), -r * cosf(g_hourglassAngle), 0.0f);
}
// ─────────────────────────────────────────────────────────────────────────────

static void drawGizmo(const Shader& sh) {
    static const glm::vec4 axisColor[3] = {
        {1.0f, 0.25f, 0.25f, 1.0f},
        {0.25f, 1.0f, 0.25f, 1.0f},
        {0.25f, 0.25f, 1.0f, 1.0f},
    };
    glm::vec3 center = (g_params.boxMin + g_params.boxMax) * 0.5f;

    glLineWidth(3.0f);
    glDisable(GL_DEPTH_TEST);
    sh.use();
    sh.setMat4("uView",       glm::value_ptr(g_viewLast));
    sh.setMat4("uProjection", glm::value_ptr(g_projLast));

    // Hourglass scene: draw a rotation circle + direction handle instead of axis handles
    if (g_scene == 3) {
        glm::vec3 ctr = (g_params.boxMin + g_params.boxMax) * 0.5f;
        float r = hourglassGizmoRadius();

        // Build circle line pairs
        const int NSEG = 48;
        std::vector<glm::vec3> pts;
        pts.reserve(NSEG * 2 + 10);
        for (int i = 0; i < NSEG; i++) {
            float a0 = (float)i       / NSEG * 6.28318530f;
            float a1 = (float)(i + 1) / NSEG * 6.28318530f;
            pts.push_back(ctr + glm::vec3(r * sinf(a0),  r * cosf(a0), 0.0f));
            pts.push_back(ctr + glm::vec3(r * sinf(a1),  r * cosf(a1), 0.0f));
        }
        // Arrow from centre to handle (shows gravity direction)
        glm::vec3 handle = hourglassGizmoHandle();
        pts.push_back(ctr);
        pts.push_back(handle);
        // Arrowhead
        glm::vec3 dir  = glm::normalize(handle - ctr);
        glm::vec3 perp = glm::vec3(-dir.y, dir.x, 0.0f) * 0.14f;
        pts.push_back(handle - dir * 0.20f + perp); pts.push_back(handle);
        pts.push_back(handle - dir * 0.20f - perp); pts.push_back(handle);

        glm::vec4 col = g_hourglassRotDrag
            ? glm::vec4(1.0f, 1.0f, 0.0f, 1.0f)
            : glm::vec4(0.3f, 0.95f, 1.0f, 0.85f);
        sh.setVec4("uColor", col.r, col.g, col.b, col.a);
        drawLineList(pts);

        glLineWidth(1.0f);
        glEnable(GL_DEPTH_TEST);
        return;
    }

    for (int a = 0; a < 3; a++) {
        if (g_scene == 2 && a != 1) continue;

        glm::vec3 h = gizmoHandle(a);
        float v[6] = { center.x, center.y, center.z, h.x, h.y, h.z };
        glBindBuffer(GL_ARRAY_BUFFER, g_gizmoVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(v), v);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glm::vec4 col = (g_gizmoAxis == a)
            ? glm::vec4(1.0f, 1.0f, 0.0f, 1.0f)
            : axisColor[a];
        sh.setVec4("uColor", col.r, col.g, col.b, col.a);

        glBindVertexArray(g_gizmoVAO);
        glDrawArrays(GL_LINES,  0, 2);
        glDrawArrays(GL_POINTS, 1, 1);
        glBindVertexArray(0);
    }
    glLineWidth(1.0f);
    glEnable(GL_DEPTH_TEST);
}

// ── Hourglass rotation gizmo ─────────────────────────────────────────────────

static bool tryHourglassRotHit(double mx, double my) {
    float sx, sy;
    if (!worldToScreen(hourglassGizmoHandle(), sx, sy)) return false;
    if (glm::length(glm::vec2((float)mx - sx, (float)my - sy)) < 22.0f) {
        g_hourglassRotDrag = true;
        return true;
    }
    return false;
}

static void applyHourglassRotDrag(double mx, double my) {
    if (!g_hourglassRotDrag) return;
    glm::vec3 c = (g_params.boxMin + g_params.boxMax) * 0.5f;

    // Unproject mouse to XY plane at z = box centre
    float ndcX =  (float)mx / (float)g_fbW * 2.0f - 1.0f;
    float ndcY = -(float)my / (float)g_fbH * 2.0f + 1.0f;
    glm::mat4 invVP = glm::inverse(g_projLast * g_viewLast);
    glm::vec4 n4 = invVP * glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 f4 = invVP * glm::vec4(ndcX, ndcY,  1.0f, 1.0f);
    glm::vec3 rO = glm::vec3(n4) / n4.w;
    glm::vec3 rD = glm::normalize(glm::vec3(f4) / f4.w - rO);

    if (fabsf(rD.z) > 1e-5f) {
        float t = (c.z - rO.z) / rD.z;
        if (t > 0.0f) {
            glm::vec3 hit = rO + rD * t;
            float dx = hit.x - c.x, dy = hit.y - c.y;
            g_hourglassAngle = atan2f(dx, -dy);

            float gMag = glm::length(glm::vec2(g_params.gravity.x, g_params.gravity.y));
            if (gMag < 0.1f) gMag = 10.0f;
            g_params.gravity.x = gMag * sinf(g_hourglassAngle);
            g_params.gravity.y = -gMag * cosf(g_hourglassAngle);
            if (g_gpu) g_gpu->syncUBO(g_params, g_currentCount);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

static void tryGizmoHit(double mx, double my) {
    g_gizmoAxis = -1; g_gizmoDragging = false;
    if (g_scene == 3) { tryHourglassRotHit(mx, my); return; }
    for (int a = 0; a < 3; a++) {
        if (g_scene == 2 && a != 1) continue;
        float sx, sy;
        if (!worldToScreen(gizmoHandle(a), sx, sy)) continue;
        if (glm::length(glm::vec2((float)mx-sx, (float)my-sy)) < 20.0f) {
            g_gizmoAxis = a; g_gizmoDragging = true; return;
        }
    }
}

static void applyGizmoDrag(double dx, double dy) {
    if (g_gizmoAxis < 0) return;
    glm::vec3 handle  = gizmoHandle(g_gizmoAxis);
    glm::vec3 axisDir = glm::vec3(g_gizmoAxis==0, g_gizmoAxis==1, g_gizmoAxis==2);
    constexpr float eps = 0.01f;

    auto ndcOf = [&](glm::vec3 wp) -> glm::vec2 {
        glm::vec4 c = g_projLast * g_viewLast * glm::vec4(wp, 1.0f);
        return (c.w > 0.0f) ? glm::vec2(c.x,c.y)/c.w : glm::vec2(0.0f);
    };
    glm::vec2 screenAxis = ndcOf(handle + axisDir*eps) - ndcOf(handle);
    float denom = glm::dot(screenAxis, screenAxis);
    if (denom < 1e-8f) return;

    glm::vec2 mouseDeltaNDC = { (float)dx*2.f/(float)g_fbW,
                                -(float)dy*2.f/(float)g_fbH };
    float worldDelta = glm::dot(mouseDeltaNDC, screenAxis) / denom * eps;

    if (g_gizmoAxis == 0) {
        glm::vec3 c = (g_params.boxMin + g_params.boxMax) * 0.5f;
        float hw = glm::max(0.15f, (g_params.boxMax.x-g_params.boxMin.x)*0.5f + worldDelta);
        g_params.boxMin.x = c.x - hw;
        g_params.boxMax.x = c.x + hw;
    } else if (g_gizmoAxis == 1) {
        if (g_scene == 2) {
            float minY = g_uboreTubeTopY + 0.10f;
            g_params.boxMax.y = glm::max(minY, g_params.boxMax.y + worldDelta);
            g_uboreSpawnH     = g_params.boxMax.y - g_uboreTubeTopY;
        } else {
            g_params.boxMax.y = glm::max(0.3f, g_params.boxMax.y + worldDelta);
        }
    } else {
        glm::vec3 c = (g_params.boxMin + g_params.boxMax) * 0.5f;
        float hd = glm::max(0.15f, (g_params.boxMax.z-g_params.boxMin.z)*0.5f + worldDelta);
        g_params.boxMin.z = c.z - hd;
        g_params.boxMax.z = c.z + hd;
    }

    if (g_gpu) g_gpu->syncUBO(g_params, g_currentCount);
}

static GLuint g_sphereVAO = 0, g_sphereVBO = 0, g_sphereEBO = 0;
struct SphVert { glm::vec3 pos; glm::vec3 nrm; };
static std::vector<SphVert>  g_sphereUnitVerts;
static std::vector<uint32_t> g_sphereIndices;

static void initSphereMesh() {
    const int LAT = 20, LON = 20;
    for (int i = 0; i <= LAT; i++) {
        float theta = (float)i / LAT * 3.14159265f;
        for (int j = 0; j <= LON; j++) {
            float phi = (float)j / LON * 6.28318530f;
            glm::vec3 p = { sinf(theta)*cosf(phi), cosf(theta), sinf(theta)*sinf(phi) };
            g_sphereUnitVerts.push_back({p, p});
        }
    }
    for (int i = 0; i < LAT; i++) {
        for (int j = 0; j < LON; j++) {
            uint32_t a = (uint32_t)(i*(LON+1)+j), b = a+(uint32_t)(LON+1);
            g_sphereIndices.insert(g_sphereIndices.end(), {a,b,a+1, b,b+1,a+1});
        }
    }
    glGenVertexArrays(1, &g_sphereVAO);
    glGenBuffers(1, &g_sphereVBO);
    glGenBuffers(1, &g_sphereEBO);
    glBindVertexArray(g_sphereVAO);
    glBindBuffer(GL_ARRAY_BUFFER, g_sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(g_sphereUnitVerts.size()*sizeof(SphVert)),
                 g_sphereUnitVerts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(g_sphereIndices.size()*sizeof(uint32_t)),
                 g_sphereIndices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SphVert), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SphVert), (void*)sizeof(glm::vec3));
    glBindVertexArray(0);
}

static void drawSphereSolid(const Shader& sh) {
    glm::mat4 model = glm::translate(glm::mat4(1.0f), g_spherePos);
    model           = glm::scale(model, glm::vec3(g_sphereRadius));

    sh.use();
    sh.setMat4("uModel",      glm::value_ptr(model));
    sh.setMat4("uView",       glm::value_ptr(g_viewLast));
    sh.setMat4("uProjection", glm::value_ptr(g_projLast));
    sh.setVec3("uLightDir",   glm::value_ptr(g_lightDir));
    sh.setVec3("uCameraPos",  glm::value_ptr(g_camPos));

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glBindVertexArray(g_sphereVAO);
    glDrawElements(GL_TRIANGLES, (GLsizei)g_sphereIndices.size(), GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

static glm::vec3 screenToWorld(double px, double py, float ndcZ) {
    glm::vec4 ndc = {
        (float)px / (float)g_fbW * 2.0f - 1.0f,
        1.0f - (float)py / (float)g_fbH * 2.0f,
        ndcZ, 1.0f };
    glm::vec4 w = glm::inverse(g_projLast * g_viewLast) * ndc;
    return glm::vec3(w) / w.w;
}

static bool trySphereDrag(double mx, double my) {
    if (!g_sphereActive) return false;
    float sx, sy;
    if (!worldToScreen(g_spherePos, sx, sy)) return false;
    float ex, ey;
    if (!worldToScreen(g_spherePos + glm::vec3(g_sphereRadius, 0, 0), ex, ey)) return false;
    float projR = glm::length(glm::vec2(ex-sx, ey-sy));
    if (glm::length(glm::vec2((float)mx-sx, (float)my-sy)) > projR + 6.0f) return false;

    glm::vec4 clip    = g_projLast * g_viewLast * glm::vec4(g_spherePos, 1.0f);
    g_sphereDragZ     = clip.z / clip.w;
    glm::vec3 pickWP  = screenToWorld(mx, my, g_sphereDragZ);
    g_sphereDragDelta = g_spherePos - pickWP;
    g_sphereDragging  = true;
    g_sphereVel       = glm::vec3(0.0f);
    return true;
}

static void applySphereDrag(double mx, double my) {
    if (!g_sphereDragging) return;
    g_spherePos = screenToWorld(mx, my, g_sphereDragZ) + g_sphereDragDelta;
}

static void updateSphere(float dt) {
    if (!g_sphereActive || !g_sphereDropped || g_sphereDragging) return;
    g_sphereVel.y += g_params.gravity.y * dt;
    g_spherePos   += g_sphereVel * dt;

    float r = g_sphereRadius;
    float d = g_params.collisionDamping;
    if (g_spherePos.x - r < g_params.boxMin.x) { g_spherePos.x = g_params.boxMin.x + r; g_sphereVel.x *= -d; }
    if (g_spherePos.x + r > g_params.boxMax.x) { g_spherePos.x = g_params.boxMax.x - r; g_sphereVel.x *= -d; }
    if (g_spherePos.y - r < g_params.boxMin.y) { g_spherePos.y = g_params.boxMin.y + r; g_sphereVel.y *= -d; }
    if (g_spherePos.z - r < g_params.boxMin.z) { g_spherePos.z = g_params.boxMin.z + r; g_sphereVel.z *= -d; }
    if (g_spherePos.z + r > g_params.boxMax.z) { g_spherePos.z = g_params.boxMax.z - r; g_sphereVel.z *= -d; }
}

static GLuint g_linesVAO = 0, g_linesVBO = 0;
static int    g_linesVBOCap = 0;

static void initLineList() {
    glGenVertexArrays(1, &g_linesVAO);
    glGenBuffers(1, &g_linesVBO);
    glBindVertexArray(g_linesVAO);
    glBindBuffer(GL_ARRAY_BUFFER, g_linesVBO);
    glBufferData(GL_ARRAY_BUFFER, 512*sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
    g_linesVBOCap = 512;
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
    glBindVertexArray(0);
}

static void drawLineList(const std::vector<glm::vec3>& pts) {
    if (pts.empty()) return;
    int needed = (int)pts.size();
    glBindBuffer(GL_ARRAY_BUFFER, g_linesVBO);
    if (needed > g_linesVBOCap) {
        glBufferData(GL_ARRAY_BUFFER, needed*sizeof(glm::vec3), pts.data(), GL_DYNAMIC_DRAW);
        g_linesVBOCap = needed;
    } else {
        glBufferSubData(GL_ARRAY_BUFFER, 0, needed*sizeof(glm::vec3), pts.data());
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(g_linesVAO);
    glDrawArrays(GL_LINES, 0, needed);
    glBindVertexArray(0);
}

static void drawStaircaseWireframe() {
    float x0  = g_params.boxMin.x, x1   = g_params.boxMax.x;
    float y0  = g_params.boxMin.y, yMax = g_params.boxMax.y;
    float z0  = g_params.boxMin.z, z1   = g_params.boxMax.z;
    float sw  = (x1 - x0) / 5.f;
    float sh  = (yMax - y0) / 5.f;

    const glm::vec2 prof[12] = {
        { x1,       yMax        },
        { x0+4*sw,  yMax        },
        { x0+4*sw,  y0+4.f*sh  },
        { x0+3*sw,  y0+4.f*sh  },
        { x0+3*sw,  y0+3.f*sh  },
        { x0+2*sw,  y0+3.f*sh  },
        { x0+2*sw,  y0+2.f*sh  },
        { x0+sw,    y0+2.f*sh  },
        { x0+sw,    y0+sh      },
        { x0,       y0+sh      },
        { x0,       y0         },
        { x1,       y0         },
    };
    const int N = 12;

    std::vector<glm::vec3> lines;
    lines.reserve(N*2 * 3);

    for (int f = 0; f < 2; f++) {
        float z = (f == 0) ? z0 : z1;
        for (int i = 0; i < N; i++) {
            int j = (i + 1) % N;
            lines.push_back({ prof[i].x, prof[i].y, z });
            lines.push_back({ prof[j].x, prof[j].y, z });
        }
    }
    for (int i = 0; i < N; i++) {
        lines.push_back({ prof[i].x, prof[i].y, z0 });
        lines.push_back({ prof[i].x, prof[i].y, z1 });
    }

    drawLineList(lines);
}

static void drawUBoreWireframe() {
    float x0   = g_params.boxMin.x, x1   = g_params.boxMax.x;
    float y0   = g_params.boxMin.y;
    float z0   = g_params.boxMin.z, z1   = g_params.boxMax.z;
    float armW = (x1 - x0) * 0.22f;
    float tubeH = g_uboreTubeTopY - y0;
    float botH  = tubeH * 0.10f;
    float jY    = g_uboreTubeTopY;

    const glm::vec2 prof[8] = {
        { x0,        jY      },
        { x0+armW,   jY      },
        { x0+armW,   y0+botH },
        { x1-armW,   y0+botH },
        { x1-armW,   jY      },
        { x1,        jY      },
        { x1,        y0      },
        { x0,        y0      },
    };
    const int N = 8;

    std::vector<glm::vec3> lines;
    lines.reserve(N * 2 * 3);

    for (int f = 0; f < 2; f++) {
        float z = (f == 0) ? z0 : z1;
        for (int i = 0; i < N; i++) {
            int j = (i + 1) % N;
            lines.push_back({ prof[i].x, prof[i].y, z });
            lines.push_back({ prof[j].x, prof[j].y, z });
        }
    }
    for (int i = 0; i < N; i++) {
        lines.push_back({ prof[i].x, prof[i].y, z0 });
        lines.push_back({ prof[i].x, prof[i].y, z1 });
    }

    drawLineList(lines);
}

static void drawHourglassWireframe() {
    glm::vec3 ctr = (g_params.boxMin + g_params.boxMax) * 0.5f;
    float ca = cosf(g_hourglassAngle), sa = sinf(g_hourglassAngle);

    // Rotate a point in the XY plane around ctr (Z unchanged)
    auto rot = [&](glm::vec3 p) -> glm::vec3 {
        float dx = p.x - ctr.x, dy = p.y - ctr.y;
        return glm::vec3(ctr.x + dx*ca - dy*sa,
                         ctr.y + dx*sa + dy*ca,
                         p.z);
    };

    // Build line pairs for an AABB, with each corner rotated
    auto addBox = [&](glm::vec3 mn, glm::vec3 mx, std::vector<glm::vec3>& lines) {
        const int idx[24] = {0,1,1,2,2,3,3,0,4,5,5,6,6,7,7,4,0,4,1,5,2,6,3,7};
        glm::vec3 c[8] = {
            rot({mn.x,mn.y,mn.z}), rot({mx.x,mn.y,mn.z}),
            rot({mx.x,mx.y,mn.z}), rot({mn.x,mx.y,mn.z}),
            rot({mn.x,mn.y,mx.z}), rot({mx.x,mn.y,mx.z}),
            rot({mx.x,mx.y,mx.z}), rot({mn.x,mx.y,mx.z})
        };
        for (int i = 0; i < 24; i += 2) { lines.push_back(c[idx[i]]); lines.push_back(c[idx[i+1]]); }
    };

    float z0 = g_params.boxMin.z, z1 = g_params.boxMax.z;
    std::vector<glm::vec3> lines;
    addBox(g_params.boxMin, g_params.boxMax, lines);
    addBox(glm::vec3(g_params.boxMin.x, -HOURGLASS_THICK, z0),
           glm::vec3(-HOURGLASS_GAP_HW,  HOURGLASS_THICK, z1), lines);
    addBox(glm::vec3(HOURGLASS_GAP_HW, -HOURGLASS_THICK, z0),
           glm::vec3(g_params.boxMax.x,  HOURGLASS_THICK, z1), lines);
    drawLineList(lines);
}

static void rebuildColliders() {
    g_colliderData = {};
    if (g_scene == 1) {
        float x0 = g_params.boxMin.x, x1   = g_params.boxMax.x;
        float y0 = g_params.boxMin.y, yMax = g_params.boxMax.y;
        float z0 = g_params.boxMin.z, z1   = g_params.boxMax.z;
        float bW = x1 - x0, bH = yMax - y0, bD = z1 - z0;
        float sw = bW / 5.f, sh = bH / 5.f;
        float zm = 0.05f;
        // Block right column above top step
        g_colliderData.cMin[0] = glm::vec4(x0+4.f*sw, y0+4.f*sh, z0-zm, 0.f);
        g_colliderData.cMax[0] = glm::vec4(x1,         yMax,      z1+zm, 0.f);
        // Inner back wall (z1 side) behind spawn area — reinforces outer AABB
        g_colliderData.cMin[1] = glm::vec4(x0+3.f*sw, y0+3.f*sh, z1-bD*0.15f, 0.f);
        g_colliderData.cMax[1] = glm::vec4(x1,         yMax,      z1+zm,        0.f);
        // Inner front wall (z0 side) behind spawn area — reinforces outer AABB
        g_colliderData.cMin[2] = glm::vec4(x0+3.f*sw, y0+3.f*sh, z0-zm,        0.f);
        g_colliderData.cMax[2] = glm::vec4(x1,         yMax,      z0+bD*0.15f,  0.f);
        g_colliderData.numColliders  = 3;
        g_colliderData.staircaseMode = 1;
    } else if (g_scene == 2) {
        float x0   = g_params.boxMin.x, x1   = g_params.boxMax.x;
        float y0   = g_params.boxMin.y;
        float z0   = g_params.boxMin.z, z1   = g_params.boxMax.z;
        float armW = (x1 - x0) * 0.22f;
        float tubeH = g_uboreTubeTopY - y0;
        float botH  = tubeH * 0.10f;
        float jY    = g_uboreTubeTopY;
        float yTop  = g_params.boxMax.y;
        float zm    = 0.05f;

        g_colliderData.cMin[0] = glm::vec4(x0+armW, y0+botH, z0-zm, 0.0f);
        g_colliderData.cMax[0] = glm::vec4(x1-armW, yTop,    z1+zm, 0.0f);
        g_colliderData.cMin[1] = glm::vec4(x1-armW, jY,   z0-zm, 0.0f);
        g_colliderData.cMax[1] = glm::vec4(x1,      yTop, z1+zm, 0.0f);

        g_colliderData.numColliders  = 2;
        g_colliderData.staircaseMode = 0;
    } else if (g_scene == 3) {
        float x0 = g_params.boxMin.x, x1 = g_params.boxMax.x;
        float z0 = g_params.boxMin.z, z1 = g_params.boxMax.z;
        float xm = 0.05f, zm = 0.05f;   // extend beyond box walls to seal edges
        g_colliderData.cMin[0] = glm::vec4(x0-xm,            -HOURGLASS_THICK, z0-zm, 0.f);
        g_colliderData.cMax[0] = glm::vec4(-HOURGLASS_GAP_HW,  HOURGLASS_THICK, z1+zm, 0.f);
        g_colliderData.cMin[1] = glm::vec4( HOURGLASS_GAP_HW, -HOURGLASS_THICK, z0-zm, 0.f);
        g_colliderData.cMax[1] = glm::vec4(x1+xm,              HOURGLASS_THICK, z1+zm, 0.f);
        g_colliderData.numColliders  = 2;
        g_colliderData.staircaseMode = 0;
    } else {
        g_colliderData.staircaseMode = 0;
    }

    if (g_sphereActive) {
        g_colliderData.spherePosRad = glm::vec4(g_spherePos, g_sphereRadius);
        g_colliderData.sphereActive = 1;
    } else {
        g_colliderData.sphereActive = 0;
    }

    glBindBuffer(GL_UNIFORM_BUFFER, g_colliderUBO);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(ColliderUniforms), &g_colliderData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

static void historySave() {
    if (!g_gpu) return;
    int bytes = g_currentCount * 64;
    std::vector<uint8_t> snap(bytes);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_gpu->particleSSBO());
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, snap.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    if ((int)g_history.size() >= MAX_HISTORY) g_history.pop_front();
    g_history.push_back(std::move(snap));
}

static void historyRestore() {
    if (!g_gpu || g_history.empty()) return;
    auto& snap = g_history.back();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_gpu->particleSSBO());
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)snap.size(), snap.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    g_history.pop_back();
}

static GLuint g_densityTex3D  = 0;
static int    g_densityTexRes = 0;

static void ensureDensityTex(int res) {
    if (g_densityTex3D && g_densityTexRes == res) return;
    if (g_densityTex3D) glDeleteTextures(1, &g_densityTex3D);
    glGenTextures(1, &g_densityTex3D);
    glBindTexture(GL_TEXTURE_3D, g_densityTex3D);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, res, res, res, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_3D, 0);
    g_densityTexRes = res;
}

void framebuffer_size_callback(GLFWwindow*, int, int);
void mouse_button_callback(GLFWwindow*, int, int, int);
void cursor_pos_callback(GLFWwindow*, double, double);
void scroll_callback(GLFWwindow*, double, double);
void key_callback(GLFWwindow*, int, int, int, int);
void InitImGui(GLFWwindow*);
void RenderUI(float fps, float ms, int count);
void Cleanup(GLFWwindow*);

int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT,
        "SPH Fluid Simulator [GPU]", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback   (window, mouse_button_callback);
    glfwSetCursorPosCallback     (window, cursor_pos_callback);
    glfwSetScrollCallback        (window, scroll_callback);
    glfwSetKeyCallback           (window, key_callback);

    InitImGui(window);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);

    g_params.boxMin = glm::vec3(-1.8f, -1.8f, -1.8f);
    g_params.boxMax = glm::vec3( 1.8f,  3.6f,  1.8f);

    SPHSolver    solver(g_params);
    solver.initDamBreak(g_targetCount);
    GPUSimulator gpu(SHADER_BASE);
    if (!gpu.init(solver.particles(), g_params)) {
        std::cerr << "[Main] GPU shaders failed\n"; return -1;
    }
    g_gpu          = &gpu;
    g_currentCount = (int)solver.particles().size();

    Shader particleShader(VERT_PATH, FRAG_PATH);
    Shader boxShader(BOX_VERT, BOX_FRAG);
    Shader gizmoShader(GIZMO_VERT, GIZMO_FRAG);
    Shader densityGridShader = Shader::compute(DENSITY_GRID_COMP);
    Shader rmShader(FULL_VERT, RM_FRAG);
    Shader sphereShader(SPHERE_VERT, SPHERE_FRAG);

    glGenBuffers(1, &g_colliderUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, g_colliderUBO);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(ColliderUniforms), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, g_colliderUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    rebuildColliders();

    initWireBox();
    initGizmo();
    initSphereMesh();
    initLineList();
    GLuint emptyVAO = 0, fullscreenVAO = 0;
    glGenVertexArrays(1, &emptyVAO);
    glGenVertexArrays(1, &fullscreenVAO);
    ensureDensityTex(g_rmGridRes);

    double lastTime = glfwGetTime(), accumTime = 0.0;
    int    nbFrames = 0;
    float  fps = 0.0f, frameMs = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (g_needReset) {
            bool sceneChanged = (g_scene != g_prevScene);
            g_prevScene = g_scene;

            // On scene change: apply the scene-specific box (but never touch physics params).
            // On same-scene Sifirla: keep the user's current box and all physics params as-is.
            if (sceneChanged) {
                if (g_scene == 0) {
                    g_params.boxMin = glm::vec3(-1.8f, -1.8f, -1.8f);
                    g_params.boxMax = glm::vec3( 1.8f,  3.6f,  1.8f);
                } else if (g_scene == 1) {
                    g_params.boxMin = glm::vec3(-1.8f, -1.8f, -1.5f);
                    g_params.boxMax = glm::vec3( 1.8f,  3.6f,  1.5f);
                } else if (g_scene == 2) {
                    constexpr float HW = 2.10f, HT = 2.10f, HD = 0.90f;
                    g_uboreTubeTopY = HT;
                    g_params.boxMin = glm::vec3(-HW, -HT, -HD);
                    g_params.boxMax = glm::vec3( HW, g_uboreTubeTopY + g_uboreSpawnH, HD);
                } else if (g_scene == 3) {
                    g_params.boxMin = glm::vec3(-0.8f, -2.0f, -0.75f);
                    g_params.boxMax = glm::vec3( 0.8f,  2.0f,  0.75f);
                }
            }
            // Always reset hourglass orientation on any hourglass reset
            if (g_scene == 3) {
                g_hourglassAngle   = 0.0f;
                float gMag = glm::length(glm::vec2(g_params.gravity.x, g_params.gravity.y));
                if (gMag < 0.1f) gMag = 10.0f;
                g_params.gravity.x = 0.0f;
                g_params.gravity.y = -gMag;
            }

            solver.params() = g_params;
            if      (g_scene == 1) solver.initStaircase(g_targetCount);
            else if (g_scene == 2) solver.initUBore(g_targetCount, g_uboreTubeTopY);
            else if (g_scene == 3) solver.initHourglass(g_targetCount);
            else                   solver.initDamBreak(g_targetCount);
            gpu.uploadParticles(solver.particles());
            g_currentCount = (int)solver.particles().size();

            rebuildColliders();
            g_history.clear();
            g_started   = false;
            g_paused    = true;
            g_needReset = false;
            g_sphereDropped = false;
            g_sphereVel     = glm::vec3(0.0f);
            if (g_sphereActive) g_spherePos = sphereRestPos();
        }

        if (g_doStepBack && g_paused && g_started) {
            historyRestore();
            g_doStepBack = false;
        }
        if (g_doStepFwd && g_paused && g_started) {
            historySave();
            glBindBufferBase(GL_UNIFORM_BUFFER, 1, g_colliderUBO);
            gpu.step(g_params, g_currentCount, g_substeps);
            g_doStepFwd = false;
        }

        if (!g_paused && g_started)
            updateSphere(g_params.timeStep * (float)g_substeps);
        rebuildColliders();

        if (!g_paused && g_started) {
            glBindBufferBase(GL_UNIFORM_BUFFER, 1, g_colliderUBO);
            gpu.step(g_params, g_currentCount, g_substeps);
        } else {
            gpu.syncUBO(g_params, g_currentCount);
        }

        double now = glfwGetTime();
        accumTime += now - lastTime; lastTime = now;
        if (++nbFrames >= 60 && accumTime > 0.0) {
            fps = float(nbFrames)/float(accumTime);
            frameMs = 1000.f/fps;
            nbFrames = 0; accumTime = 0.0;
        }

        glfwGetFramebufferSize(window, &g_fbW, &g_fbH);
        ensureDensityTex(g_rmGridRes);

        float     aspect = (g_fbH > 0) ? float(g_fbW)/float(g_fbH) : 1.0f;
        g_viewLast       = g_camera.view();
        g_projLast       = g_camera.projection(aspect);
        g_camPos         = g_camera.eye();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        RenderUI(fps, frameMs, g_currentCount);

        glViewport(0, 0, g_fbW, g_fbH);
        glClearColor(0.06f, 0.06f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (g_renderMode == 1 && g_currentCount > 0) {
            if (densityGridShader.valid()) {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gpu.particleSSBO());
                glBindImageTexture(1, g_densityTex3D, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
                densityGridShader.use();
                densityGridShader.setIVec3("uGridRes", g_rmGridRes, g_rmGridRes, g_rmGridRes);
                int gx=(g_rmGridRes+7)/8, gy=(g_rmGridRes+7)/8, gz=(g_rmGridRes+3)/4;
                glDispatchCompute(gx, gy, gz);
                glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT|GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            }

            glm::mat4 invProj = glm::inverse(g_projLast);
            glm::mat4 invView = glm::inverse(g_viewLast);
            rmShader.use();
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, g_densityTex3D);
            rmShader.setInt  ("uDensityGrid",    0);
            rmShader.setMat4 ("uInvProjection",  glm::value_ptr(invProj));
            rmShader.setMat4 ("uInvView",        glm::value_ptr(invView));
            rmShader.setMat4 ("uProjection",     glm::value_ptr(g_projLast));
            rmShader.setMat4 ("uView",           glm::value_ptr(g_viewLast));
            rmShader.setVec3 ("uCameraPos",      glm::value_ptr(g_camPos));
            rmShader.setVec3 ("uBoxMin",         glm::value_ptr(g_params.boxMin));
            rmShader.setVec3 ("uBoxMax",         glm::value_ptr(g_params.boxMax));
            rmShader.setIVec3("uGridRes",        g_rmGridRes, g_rmGridRes, g_rmGridRes);
            rmShader.setFloat("uIsoLevel",       g_rmIsoLevel);
            rmShader.setFloat("uStepScale",      g_rmStepScale);
            rmShader.setVec3 ("uLightDir",       glm::value_ptr(g_lightDir));
            rmShader.setVec3 ("uLightColor",     glm::value_ptr(g_lightColor));
            rmShader.setVec3 ("uWaterColor",     glm::value_ptr(g_waterColor));
            rmShader.setVec3 ("uDeepColor",      glm::value_ptr(g_deepColor));
            rmShader.setFloat("uAmbient",        g_ambient);
            rmShader.setFloat("uSpecular",       g_specular);
            rmShader.setFloat("uShininess",      g_shininess);
            rmShader.setFloat("uFresnelBias",    g_fresnelBias);
            rmShader.setFloat("uThicknessScale", g_thicknessScale);
            glBindVertexArray(fullscreenVAO);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glBindVertexArray(0);

        } else {
            if (particleShader.valid() && g_currentCount > 0) {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gpu.particleSSBO());
                particleShader.use();
                particleShader.setMat4 ("uView",        glm::value_ptr(g_viewLast));
                particleShader.setMat4 ("uProjection",  glm::value_ptr(g_projLast));
                particleShader.setInt  ("uColorMode",   g_colorMode);
                particleShader.setFloat("uPointSize",   g_pointSize);
                particleShader.setFloat("uMaxPressure", 1000.0f);
                glBindVertexArray(emptyVAO);
                glDrawArrays(GL_POINTS, 0, g_currentCount);
                glBindVertexArray(0);
            }
        }

        if (boxShader.valid()) {
            glLineWidth(1.5f);
            boxShader.use();
            boxShader.setMat4("uView",       glm::value_ptr(g_viewLast));
            boxShader.setMat4("uProjection", glm::value_ptr(g_projLast));

            if (g_scene == 1) {
                boxShader.setVec4("uColor", 0.5f, 0.8f, 1.0f, 0.45f);
                drawStaircaseWireframe();
            } else if (g_scene == 2) {
                boxShader.setVec4("uColor", 0.5f, 0.8f, 1.0f, 0.45f);
                drawUBoreWireframe();
                float x0   = g_params.boxMin.x, x1 = g_params.boxMax.x;
                float armW = (x1 - x0) * 0.22f;
                boxShader.setVec4("uColor", 0.3f, 1.0f, 0.4f, 0.50f);
                drawWireBox(glm::vec3(x0,       g_uboreTubeTopY, g_params.boxMin.z),
                            glm::vec3(x0+armW,  g_params.boxMax.y, g_params.boxMax.z));
            } else if (g_scene == 3) {
                boxShader.setVec4("uColor", 0.5f, 0.8f, 1.0f, 0.45f);
                drawHourglassWireframe();
            } else {
                boxShader.setVec4("uColor", 0.5f, 0.8f, 1.0f, 0.35f);
                drawWireBox(g_params.boxMin, g_params.boxMax);
            }
        }

        if (g_sphereActive && sphereShader.valid())
            drawSphereSolid(sphereShader);

        if (gizmoShader.valid()) drawGizmo(gizmoShader);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    if (g_densityTex3D) glDeleteTextures(1, &g_densityTex3D);
    if (g_colliderUBO)  glDeleteBuffers(1, &g_colliderUBO);
    glDeleteBuffers(1, &g_wireVBO); glDeleteBuffers(1, &g_wireEBO);
    glDeleteVertexArrays(1, &g_wireVAO);
    glDeleteBuffers(1, &g_gizmoVBO);
    glDeleteVertexArrays(1, &g_gizmoVAO);
    glDeleteBuffers(1, &g_sphereVBO);
    glDeleteBuffers(1, &g_sphereEBO);
    glDeleteVertexArrays(1, &g_sphereVAO);
    glDeleteBuffers(1, &g_linesVBO);
    glDeleteVertexArrays(1, &g_linesVAO);
    glDeleteVertexArrays(1, &emptyVAO);
    glDeleteVertexArrays(1, &fullscreenVAO);
    Cleanup(window);
    return 0;
}

void framebuffer_size_callback(GLFWwindow*, int w, int h) {
    glViewport(0,0,w,h); g_fbW=w; g_fbH=h;
}

void mouse_button_callback(GLFWwindow*, int btn, int action, int) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (btn == GLFW_MOUSE_BUTTON_RIGHT)  g_mouseCapture  = (action==GLFW_PRESS);
    if (btn == GLFW_MOUSE_BUTTON_MIDDLE) g_middleCapture = (action==GLFW_PRESS);
    if (btn == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            if (!trySphereDrag(g_lastMouseX, g_lastMouseY))
                tryGizmoHit(g_lastMouseX, g_lastMouseY);
        } else {
            g_sphereDragging    = false;
            g_gizmoDragging     = false;
            g_gizmoAxis         = -1;
            g_hourglassRotDrag  = false;
        }
    }
}

void cursor_pos_callback(GLFWwindow*, double x, double y) {
    double dx = x-g_lastMouseX, dy = y-g_lastMouseY;
    g_lastMouseX=x; g_lastMouseY=y;
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (g_sphereDragging)    { applySphereDrag(x, y); return; }
    if (g_gizmoDragging)     { applyGizmoDrag(dx, dy); return; }
    if (g_hourglassRotDrag)  { applyHourglassRotDrag(x, y); return; }
    if (g_mouseCapture)      g_camera.orbit( (float)dx,-(float)dy);
    if (g_middleCapture)  g_camera.pan  ( (float)dx, (float)dy);
}

void scroll_callback(GLFWwindow*, double, double y) {
    if (!ImGui::GetIO().WantCaptureMouse) g_camera.zoom((float)y);
}

void key_callback(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, true);
    if (key == GLFW_KEY_SPACE && g_started) g_paused = !g_paused;
    if (key == GLFW_KEY_TAB)   g_renderMode = 1 - g_renderMode;
}

void InitImGui(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 430");
}

void RenderUI(float fps, float ms, int count)
{
    ImGui::Begin("SPH Kontrol Paneli");
    ImGui::Text("FPS: %.1f   Frame: %.2f ms", fps, ms);
    ImGui::Separator();

    ImGui::Text("Sahne:");
    int prevScene = g_scene;
    ImGui::RadioButton("Dam Break",  &g_scene, 0); ImGui::SameLine();
    ImGui::RadioButton("Merdiven",   &g_scene, 1); ImGui::SameLine();
    ImGui::RadioButton("U Boru",     &g_scene, 2); ImGui::SameLine();
    ImGui::RadioButton("Kum Saati",  &g_scene, 3);
    if (g_scene != prevScene) g_needReset = true;

    if (g_scene == 3) {
        if (ImGui::Button("Cevir 180")) {
            g_hourglassAngle += 3.14159265f;
            float gMag = glm::length(glm::vec2(g_params.gravity.x, g_params.gravity.y));
            if (gMag < 0.1f) gMag = 10.0f;
            g_params.gravity.x = gMag * sinf(g_hourglassAngle);
            g_params.gravity.y = -gMag * cosf(g_hourglassAngle);
            if (g_gpu) g_gpu->syncUBO(g_params, g_currentCount);
        }
    }
    ImGui::Separator();

    ImGui::SliderInt("Parcacik Sayisi", &g_targetCount, 100, 50000);
    if (ImGui::Button("Uygula##cnt")) g_needReset = true;
    ImGui::Separator();

    auto PushGreen = []{ ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.13f,0.60f,0.13f,1.f));
                         ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.18f,0.78f,0.18f,1.f));
                         ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.09f,0.45f,0.09f,1.f)); };
    auto PushRed   = []{ ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.72f,0.13f,0.13f,1.f));
                         ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.88f,0.18f,0.18f,1.f));
                         ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.55f,0.09f,0.09f,1.f)); };
    auto PushBlue  = []{ ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.13f,0.33f,0.72f,1.f));
                         ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.18f,0.44f,0.88f,1.f));
                         ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.09f,0.24f,0.55f,1.f)); };
    auto PushGrey  = []{ ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.33f,0.33f,0.33f,1.f));
                         ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f,0.45f,0.45f,1.f));
                         ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.22f,0.22f,0.22f,1.f)); };

    if (!g_started) {
        PushGreen();
        if (ImGui::Button("  Baslat  ", ImVec2(0, 34))) { g_started=true; g_paused=false; }
        ImGui::PopStyleColor(3);
    } else {
        PushGrey();
        if (ImGui::Button("Sifirla", ImVec2(80, 34))) g_needReset = true;
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        PushRed();
        const char* pauseLabel = g_paused ? "Devam  [Space]" : "Durdur [Space]";
        if (ImGui::Button(pauseLabel, ImVec2(0, 34))) g_paused = !g_paused;
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        ImGui::BeginDisabled(!g_paused);
        PushBlue();
        if (ImGui::Button("<<", ImVec2(44, 34))) g_doStepBack = true;
        ImGui::SameLine();
        if (ImGui::Button(">>", ImVec2(44, 34))) g_doStepFwd  = true;
        ImGui::PopStyleColor(3);
        ImGui::EndDisabled();
    }
    ImGui::Separator();

    ImGui::Text("Fizik");
    ImGui::SliderFloat("Yercekimi",      &g_params.gravity.y,         -20.f,  -1.f,    "%.2f");
    ImGui::SliderFloat("Hedef Yogunluk", &g_params.targetDensity,      100.f, 2000.f,  "%.0f");
    ImGui::SliderFloat("Viskozite",      &g_params.viscosityStrength,   0.f,    0.09f, "%.4f");
    ImGui::SliderFloat("Zaman Adimi",    &g_params.timeStep,            0.0005f,0.005f,"%.4f");
    ImGui::Separator();

    ImGui::Text("Kure");
    if (!g_sphereActive) {
        if (ImGui::Button("Kure Ekle")) {
            g_sphereActive  = true;
            g_sphereDropped = false;
            g_sphereVel     = glm::vec3(0.0f);
            g_spherePos     = sphereRestPos();
        }
    } else {
        if (ImGui::Button("Kure Kaldir")) {
            g_sphereActive  = false;
            g_sphereDropped = false;
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(g_sphereDropped);
        if (ImGui::Button("Birak")) {
            g_sphereDropped = true;
            g_sphereVel     = glm::vec3(0.0f);
        }
        ImGui::EndDisabled();
        if (ImGui::SliderFloat("Kure Boyutu", &g_sphereRadius, 0.05f, 1.0f, "%.3f")) {
            if (!g_sphereDropped) g_spherePos = sphereRestPos();
        }
    }
    ImGui::Separator();

    ImGui::Text("Render [Tab]:");
    ImGui::RadioButton("Parcaciklar", &g_renderMode, 0); ImGui::SameLine();
    ImGui::RadioButton("Ray March",   &g_renderMode, 1);

    if (g_renderMode == 0) {
        ImGui::SliderFloat("Parcacik Boyutu", &g_pointSize, 2.0f, 80.0f, "%.1f");
        ImGui::RadioButton("Hiz",      &g_colorMode, 0); ImGui::SameLine();
        ImGui::RadioButton("Yogunluk", &g_colorMode, 1);
    }

    if (g_renderMode == 1) {
        ImGui::Separator();
        ImGui::SliderFloat("Hassasiyet", &g_rmIsoLevel, 50.0f, 700.0f, "%.0f");
        float wc[3] = {g_waterColor.x, g_waterColor.y, g_waterColor.z};
        if (ImGui::ColorEdit3("Sivi Rengi", wc)) g_waterColor = {wc[0], wc[1], wc[2]};
    }

    ImGui::Separator();
    ImGui::Text("Sivi Preset:");
    if (ImGui::Button("Su")) {
        g_waterColor = glm::vec3(0.10f, 0.40f, 0.80f);
        g_deepColor  = glm::vec3(0.02f, 0.12f, 0.40f);
        g_params.viscosityStrength = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Balcik")) {
        g_waterColor = glm::vec3(0.10f, 0.70f, 0.15f);
        g_deepColor  = glm::vec3(0.03f, 0.35f, 0.05f);
        g_params.viscosityStrength = 0.03f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Bal")) {
        g_waterColor = glm::vec3(0.92f, 0.62f, 0.08f);
        g_deepColor  = glm::vec3(0.58f, 0.32f, 0.02f);
        g_params.viscosityStrength = 0.07f;
    }
    if (ImGui::Button("Parametreleri Sifirla")) {
        glm::vec3 bMn=g_params.boxMin, bMx=g_params.boxMax;
        g_params=g_defaultParams; g_params.boxMin=bMn; g_params.boxMax=bMx;
    }
    ImGui::Separator();
    ImGui::TextDisabled("LMB=gizmo  RMB=orbit  MMB=pan  Scroll=zoom");
    ImGui::End();
}

void Cleanup(GLFWwindow* window) {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}
