// Wireframe box fragment shader: outputs a uniform colour passed from the CPU.
#version 430 core
uniform vec4 uColor = vec4(0.5, 0.8, 1.0, 0.35);
out vec4 FragColor;
void main() {
    FragColor = uColor;
}
