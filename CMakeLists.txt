cmake_minimum_required(VERSION 3.15)
project(SPH_FluidSim)

set(CMAKE_CXX_STANDARD 17)

# Bağımlılıkları vcpkg'den bul
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

# BURASI ÇOK ÖNEMLİ: Yeni eklediğin .cpp dosyalarını buraya tek tek yazmalısın
add_executable(${PROJECT_NAME} 
    "src/main.cpp"
    "src/imgui_impl_glfw.cpp"
    "src/imgui_impl_opengl3.cpp"
    "src/Sphsolver.cpp"
 "src/Shader.h" "src/Camera.h")

# Kütüphaneleri bağla
target_link_libraries(${PROJECT_NAME} PRIVATE 
    glm::glm 
    glfw 
    glad::glad
    imgui::imgui
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)