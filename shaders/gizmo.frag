// Gizmo fragment shader: passes through the axis colour; clips GL_POINTS to circles.
#version 430 core
uniform vec4 uColor;
out vec4 FragColor;
void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    if (dot(c, c) > 0.25) discard;
    FragColor = uColor;
}
