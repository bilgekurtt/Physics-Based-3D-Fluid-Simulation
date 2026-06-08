// Camera: arc-ball camera with orbit, pan, and zoom controls.
// Produces view and projection matrices consumed by the render shaders.
#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    glm::vec3 target   = glm::vec3(0.0f, 0.5f, 0.0f);
    float     yaw      = 45.0f;
    float     pitch    = 25.0f;
    float     distance = 4.0f;
    float     fovY     = 45.0f;
    float     nearPlane = 0.01f;
    float     farPlane  = 100.0f;

    glm::mat4 view() const {
        return glm::lookAt(eye(), target, glm::vec3(0, 1, 0));
    }

    glm::mat4 projection(float aspectRatio) const {
        return glm::perspective(glm::radians(fovY), aspectRatio, nearPlane, farPlane);
    }

    glm::vec3 eye() const {
        float yR = glm::radians(yaw);
        float pR = glm::radians(pitch);
        return target + glm::vec3(
            distance * std::cos(pR) * std::sin(yR),
            distance * std::sin(pR),
            distance * std::cos(pR) * std::cos(yR));
    }

    void orbit(float dx, float dy) {
        yaw   += dx * orbitSpeed;
        pitch += dy * orbitSpeed;
        pitch  = glm::clamp(pitch, -89.0f, 89.0f);
    }

    void pan(float dx, float dy) {
        glm::vec3 forward = glm::normalize(target - eye());
        glm::vec3 right   = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
        glm::vec3 up      = glm::cross(right, forward);
        target -= right * dx * panSpeed  * distance;
        target += up    * dy * panSpeed  * distance;
    }

    void zoom(float delta) {
        distance -= delta * zoomSpeed * distance;
        distance  = glm::clamp(distance, 0.3f, 50.0f);
    }

private:
    float orbitSpeed = 0.4f;
    float panSpeed   = 0.001f;
    float zoomSpeed  = 0.1f;
};
