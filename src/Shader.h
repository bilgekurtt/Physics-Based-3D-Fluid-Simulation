// Shader: OpenGL shader program wrapper. Supports vertex+fragment pairs and
// standalone compute shaders. Provides typed uniform setters and a dispatch helper.
#pragma once
#include <glad/glad.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Shader {
public:
    GLuint ID = 0;

    Shader() = default;

    Shader(const char* vertPath, const char* fragPath) {
        std::string vertSrc = readFile(vertPath);
        std::string fragSrc = readFile(fragPath);
        if (vertSrc.empty() || fragSrc.empty()) return;

        GLuint vert = compile(GL_VERTEX_SHADER,   vertSrc.c_str(), vertPath);
        GLuint frag = compile(GL_FRAGMENT_SHADER, fragSrc.c_str(), fragPath);
        if (!vert || !frag) return;

        ID = glCreateProgram();
        glAttachShader(ID, vert);
        glAttachShader(ID, frag);
        glLinkProgram(ID);
        checkLink(ID);
        glDeleteShader(vert);
        glDeleteShader(frag);
    }

    static Shader compute(const char* compPath) {
        Shader s;
        std::string src = readFile(compPath);
        if (src.empty()) return s;

        GLuint comp = compile(GL_COMPUTE_SHADER, src.c_str(), compPath);
        if (!comp) return s;

        s.ID = glCreateProgram();
        glAttachShader(s.ID, comp);
        glLinkProgram(s.ID);
        checkLink(s.ID);
        glDeleteShader(comp);
        return s;
    }

    void dispatch(GLuint numGroups) const {
        glUseProgram(ID);
        glDispatchCompute(numGroups, 1, 1);
    }

    void use() const { glUseProgram(ID); }

    void setMat4 (const char* name, const float* value) const {
        glUniformMatrix4fv(glGetUniformLocation(ID, name), 1, GL_FALSE, value);
    }
    void setFloat(const char* name, float value) const {
        glUniform1f(glGetUniformLocation(ID, name), value);
    }
    void setInt  (const char* name, int value) const {
        glUniform1i(glGetUniformLocation(ID, name), value);
    }
    void setVec3 (const char* name, float x, float y, float z) const {
        glUniform3f(glGetUniformLocation(ID, name), x, y, z);
    }
    void setVec3 (const char* name, const float* v) const {
        glUniform3fv(glGetUniformLocation(ID, name), 1, v);
    }
    void setVec4 (const char* name, float x, float y, float z, float w) const {
        glUniform4f(glGetUniformLocation(ID, name), x, y, z, w);
    }
    void setIVec3(const char* name, int x, int y, int z) const {
        glUniform3i(glGetUniformLocation(ID, name), x, y, z);
    }

    bool valid() const { return ID != 0; }

    ~Shader() { if (ID) glDeleteProgram(ID); }

    Shader(const Shader&)            = delete;
    Shader& operator=(const Shader&) = delete;
    Shader(Shader&& o) noexcept : ID(o.ID) { o.ID = 0; }
    Shader& operator=(Shader&& o) noexcept {
        if (ID) glDeleteProgram(ID);
        ID = o.ID; o.ID = 0;
        return *this;
    }

private:
    static std::string readFile(const char* path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "[Shader] Cannot open: " << path << "\n";
            return {};
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    static GLuint compile(GLenum type, const char* src, const char* label) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[1024];
            glGetShaderInfoLog(s, sizeof(log), nullptr, log);
            std::cerr << "[Shader] Compile error (" << label << "):\n" << log << "\n";
            glDeleteShader(s);
            return 0;
        }
        return s;
    }

    static void checkLink(GLuint prog) {
        GLint ok = 0;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[1024];
            glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
            std::cerr << "[Shader] Link error:\n" << log << "\n";
        }
    }
};
