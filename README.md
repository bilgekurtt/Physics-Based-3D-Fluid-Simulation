# Physics Based 3D Fluid Simulation

Real-time 3D fluid simulation using Smoothed Particle Hydrodynamics (SPH) on the GPU. Built with C++17, OpenGL 4.3, GLFW, GLM, and Dear ImGui. All physics runs entirely on the GPU via compute shaders.

## Scenes
- Dam Break
- Staircase
- U-Tube
- Hourglass (rotatable gravity)

## Controls
| Input | Action |
|-------|--------|
| RMB + drag | Orbit camera |
| MMB + drag | Pan |
| Scroll | Zoom |
| LMB + drag | Move gizmo |

## Build
Requires Visual Studio 2022 with CMake and vcpkg.
```
cmake -B out/build -S . --toolchain <vcpkg>/scripts/buildsystems/vcpkg.cmake
```
