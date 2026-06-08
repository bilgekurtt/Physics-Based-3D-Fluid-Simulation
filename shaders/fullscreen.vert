// Fullscreen triangle vertex shader: generates a single large triangle that covers
// the entire viewport from gl_VertexID alone, with no vertex buffer required.
#version 430 core
out vec2 vUV;
void main() {
    vUV         = vec2((gl_VertexID & 1) * 2.0, (gl_VertexID >> 1) * 2.0);
    gl_Position = vec4(vUV * 2.0 - 1.0, 0.0, 1.0);
}
