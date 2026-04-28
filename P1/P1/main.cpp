// ============================================================
//  GPS Scene - S4
//  Libraries: GLFW, GLAD, GLM, stb_image (procedural textures)
//  Build: see README_BUILD.txt
// ============================================================

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm/glm.hpp>
#include <glm/glm/gtc/matrix_transform.hpp>
#include <glm/glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
 
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>


// stb_image - header-only, define implementation in ONE .cpp file
#define STB_IMAGE_IMPLEMENTATION

// ═════════════════════════════════════════════
//  CONSTANTS
// ═════════════════════════════════════════════
const int   SCR_W = 1280, SCR_H = 720;
const float PI    = 3.14159265f;
 
// Shadow map resolution
const int SHADOW_W = 4096, SHADOW_H = 4096;
 
// ═════════════════════════════════════════════
//  STRUCTS
// ═════════════════════════════════════════════
 
// Axis-Aligned Bounding Box for collisions
struct AABB {
    glm::vec3 min, max;
};
 
// A static object in the scene
struct StaticObject {
    glm::vec3 position;
    glm::vec3 scale;
    float     rotationY;
    int       type; // 0=building, 1=tree, 2=streetlight
    AABB      bbox;
};
 
// The controllable car
struct Car {
    glm::vec3 position  = glm::vec3(0.f, 0.f, 15.f);
    float     angle     = 0.f;  // Y rotation in degrees
    float     speed     = 0.f;
    float     maxSpeed  = 12.f;
    float     accel     = 8.f;
    float     braking   = 15.f;
    float     friction  = 3.f;
    float     turnSpeed = 90.f; // degrees per second
    glm::vec3 size      = glm::vec3(1.2f, 0.6f, 2.4f); // half-extents for collision
 
    AABB getBBox() const {
        // Simplified AABB (doesn't rotate with car — using max dimension)
        float maxR = std::max(size.x, size.z);
        return {
            position - glm::vec3(maxR, 0.f, maxR),
            position + glm::vec3(maxR, size.y * 2.f, maxR)
        };
    }
};
 
// Camera modes
enum CameraMode { CAM_FREE, CAM_FOLLOW, CAM_TOP };
 
// ═════════════════════════════════════════════
//  GLOBALS
// ═════════════════════════════════════════════
// Free camera
glm::vec3 camPos   = glm::vec3(0.f, 25.f, 40.f);
glm::vec3 camFront = glm::vec3(0.f, -0.5f, -1.f);
glm::vec3 camUp    = glm::vec3(0.f, 1.f, 0.f);
float yaw = -90.f, pitch = -25.f;
float lastX = SCR_W / 2.f, lastY = SCR_H / 2.f;
bool  firstMouse = true;
float fov = 60.f;
 
float deltaTime = 0.f, lastFrame = 0.f;
 
CameraMode camMode = CAM_FOLLOW;
Car        car;
std::vector<StaticObject> objects;
 
bool keyStates[1024] = {false};
bool collisionOccurred = false;
float collisionTimer = 0.f;
 
// ═════════════════════════════════════════════
//  SHADER SOURCES
// ═════════════════════════════════════════════
 
// ── Main scene shader with shadow mapping ──
const char* mainVS = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;
layout(location=2) in vec3 aNorm;
 
uniform mat4 model, view, projection;
uniform mat4 lightSpaceMatrix;
 
out vec3 vFragPos;
out vec3 vNorm;
out vec2 vUV;
out vec4 vFragPosLightSpace;
 
void main(){
    vec4 worldPos = model * vec4(aPos, 1.0);
    vFragPos = worldPos.xyz;
    vNorm = mat3(transpose(inverse(model))) * aNorm;
    vUV = aUV;
    vFragPosLightSpace = lightSpaceMatrix * worldPos;
    gl_Position = projection * view * worldPos;
}
)";
 
const char* mainFS = R"(
#version 330 core
in vec3 vFragPos;
in vec3 vNorm;
in vec2 vUV;
in vec4 vFragPosLightSpace;
 
uniform sampler2D tex;
uniform sampler2D shadowMap;
uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 ambientColor;
uniform float tilingFactor;
uniform vec3 objectColor;    // if no texture, use solid color
uniform int  useTexture;     // 1=sample tex, 0=use objectColor
uniform vec3 viewPos;
 
// Point lights (streetlights)
#define MAX_POINT_LIGHTS 8
uniform int  numPointLights;
uniform vec3 pointLightPos[MAX_POINT_LIGHTS];
uniform vec3 pointLightColor[MAX_POINT_LIGHTS];
 
out vec4 FragColor;
 
float calcShadow(vec4 fragPosLS, vec3 normal, vec3 ldir){
    vec3 projCoords = fragPosLS.xyz / fragPosLS.w;
    projCoords = projCoords * 0.5 + 0.5;
    if(projCoords.z > 1.0) return 0.0;
    float currentDepth = projCoords.z;
    float bias = max(0.005 * (1.0 - dot(normal, -ldir)), 0.001);
    // PCF soft shadows
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -2; x <= 2; x++){
        for(int y = -2; y <= 2; y++){
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x,y)*texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 25.0;
    return shadow;
}
 
void main(){
    vec3 norm = normalize(vNorm);
    vec3 baseColor;
    if(useTexture == 1)
        baseColor = texture(tex, vUV * tilingFactor).rgb;
    else
        baseColor = objectColor;
 
    // Directional light
    float diff = max(dot(norm, normalize(-lightDir)), 0.0);
    vec3 diffuse = diff * lightColor;
 
    // Specular (Blinn-Phong)
    vec3 viewDir = normalize(viewPos - vFragPos);
    vec3 halfDir = normalize(normalize(-lightDir) + viewDir);
    float spec = pow(max(dot(norm, halfDir), 0.0), 32.0);
    vec3 specular = spec * lightColor * 0.3;
 
    // Shadow
    float shadow = calcShadow(vFragPosLightSpace, norm, lightDir);
 
    vec3 result = (ambientColor + (1.0 - shadow) * (diffuse + specular)) * baseColor;
 
    // Point lights (streetlights) - each creates its own lit area
    for(int i = 0; i < numPointLights; i++){
        vec3 lVec = pointLightPos[i] - vFragPos;
        float dist = length(lVec);
        vec3 lDir = normalize(lVec);
        float atten = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);
        float pDiff = max(dot(norm, lDir), 0.0);
        result += pDiff * atten * pointLightColor[i] * baseColor;
    }
 
    FragColor = vec4(result, 1.0);
}
)";
 
// ── Shadow depth shader ──
const char* shadowVS = R"(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 model;
uniform mat4 lightSpaceMatrix;
void main(){
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}
)";
 
const char* shadowFS = R"(
#version 330 core
void main(){ }
)";
 
// ── Skybox shader ──
const char* skyVS = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;
uniform mat4 view, projection;
out vec2 vUV;
out float vY;
void main(){
    vUV = aUV;
    vY  = aPos.y;
    vec4 pos = projection * mat4(mat3(view)) * vec4(aPos, 1.0);
    gl_Position = pos.xyww;
}
)";
 
const char* skyFS = R"(
#version 330 core
in vec2  vUV;
in float vY;
uniform sampler2D texSky;
uniform int faceID;
out vec4 FragColor;
void main(){
    vec3 col;
    if(faceID == 4)
        col = mix(vec3(0.40, 0.65, 0.95), vec3(0.15, 0.35, 0.75), vUV.y);
    else if(faceID == 5)
        col = vec3(0.22, 0.18, 0.12);
    else
        col = texture(texSky, vUV).rgb;
    FragColor = vec4(col, 1.0);
}
)";
 
// ── HUD shader (for collision flash) ──
const char* hudVS = R"(
#version 330 core
layout(location=0) in vec2 aPos;
void main(){
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";
 
const char* hudFS = R"(
#version 330 core
uniform vec4 color;
out vec4 FragColor;
void main(){
    FragColor = color;
}
)";
 
// ═════════════════════════════════════════════
//  UTILITY: compile + link shaders
// ═════════════════════════════════════════════
GLuint makeShader(const char* vs, const char* fs)
{
    auto compile = [](GLenum type, const char* src) -> GLuint {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if(!ok){
            char log[1024]; glGetShaderInfoLog(s, 1024, nullptr, log);
            std::cerr << "Shader compile error:\n" << log << "\n";
        }
        return s;
    };
    GLuint v = compile(GL_VERTEX_SHADER, vs);
    GLuint f = compile(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok){
        char log[1024]; glGetProgramInfoLog(p, 1024, nullptr, log);
        std::cerr << "Shader link error:\n" << log << "\n";
    }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}
 
// ═════════════════════════════════════════════
//  MESH
// ═════════════════════════════════════════════
struct Mesh {
    GLuint vao = 0, vbo = 0, ebo = 0;
    int indexCount = 0;
};
 
Mesh buildMesh(const std::vector<float>& verts, const std::vector<unsigned int>& idx)
{
    Mesh m;
    m.indexCount = (int)idx.size();
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glGenBuffers(1, &m.ebo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size()*sizeof(float), verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size()*sizeof(unsigned int), idx.data(), GL_STATIC_DRAW);
    // stride = 8 floats: pos(3) uv(2) norm(3)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(5*sizeof(float)));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
    return m;
}
 
// ═════════════════════════════════════════════
//  MESH BUILDERS
// ═════════════════════════════════════════════
 
// Flat ground
Mesh buildGround(float halfSize)
{
    std::vector<float> v = {
        -halfSize, 0, -halfSize,  0,0,  0,1,0,
         halfSize, 0, -halfSize,  1,0,  0,1,0,
         halfSize, 0,  halfSize,  1,1,  0,1,0,
        -halfSize, 0,  halfSize,  0,1,  0,1,0,
    };
    std::vector<unsigned int> idx = {0,2,1, 0,3,2};
    return buildMesh(v, idx);
}
 
// Box (for buildings, car body, etc.)
Mesh buildBox()
{
    // Unit cube centered at origin, from -0.5 to 0.5
    std::vector<float> v;
    std::vector<unsigned int> idx;
 
    // Helper: add a face (4 verts, 2 tris)
    auto addFace = [&](glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 n){
        unsigned int base = (unsigned int)(v.size() / 8);
        v.insert(v.end(), {p0.x,p0.y,p0.z, 0,0, n.x,n.y,n.z});
        v.insert(v.end(), {p1.x,p1.y,p1.z, 1,0, n.x,n.y,n.z});
        v.insert(v.end(), {p2.x,p2.y,p2.z, 1,1, n.x,n.y,n.z});
        v.insert(v.end(), {p3.x,p3.y,p3.z, 0,1, n.x,n.y,n.z});
        idx.insert(idx.end(), {base,base+1,base+2, base,base+2,base+3});
    };
 
    float s = 0.5f;
    // Front (+Z)
    addFace({-s,-s,s},{s,-s,s},{s,s,s},{-s,s,s}, {0,0,1});
    // Back (-Z)
    addFace({s,-s,-s},{-s,-s,-s},{-s,s,-s},{s,s,-s}, {0,0,-1});
    // Left (-X)
    addFace({-s,-s,-s},{-s,-s,s},{-s,s,s},{-s,s,-s}, {-1,0,0});
    // Right (+X)
    addFace({s,-s,s},{s,-s,-s},{s,s,-s},{s,s,s}, {1,0,0});
    // Top (+Y)
    addFace({-s,s,s},{s,s,s},{s,s,-s},{-s,s,-s}, {0,1,0});
    // Bottom (-Y)
    addFace({-s,-s,-s},{s,-s,-s},{s,-s,s},{-s,-s,s}, {0,-1,0});
 
    return buildMesh(v, idx);
}
 
// Cylinder (for tree trunks, lamppost poles)
Mesh buildCylinder(int segments, float radius, float height)
{
    std::vector<float> v;
    std::vector<unsigned int> idx;
 
    for(int i = 0; i <= segments; i++){
        float a = (float)i / segments * 2.f * PI;
        float x = cosf(a) * radius;
        float z = sinf(a) * radius;
        float u = (float)i / segments;
        glm::vec3 n = glm::normalize(glm::vec3(cosf(a), 0, sinf(a)));
        // Bottom vertex
        v.insert(v.end(), {x, 0.f, z, u, 0.f, n.x, n.y, n.z});
        // Top vertex
        v.insert(v.end(), {x, height, z, u, 1.f, n.x, n.y, n.z});
    }
    for(int i = 0; i < segments; i++){
        unsigned int b = i * 2;
        idx.insert(idx.end(), {b, b+1, b+3, b, b+3, b+2});
    }
    return buildMesh(v, idx);
}
 
// Cone (for tree canopy)
Mesh buildCone(int segments, float radius, float height)
{
    std::vector<float> v;
    std::vector<unsigned int> idx;
 
    // Apex
    v.insert(v.end(), {0.f, height, 0.f, 0.5f, 1.f, 0.f, 0.7f, 0.f});
    // Base ring
    for(int i = 0; i <= segments; i++){
        float a = (float)i / segments * 2.f * PI;
        float x = cosf(a) * radius;
        float z = sinf(a) * radius;
        float u = (float)i / segments;
        glm::vec3 n = glm::normalize(glm::vec3(cosf(a), radius/height, sinf(a)));
        v.insert(v.end(), {x, 0.f, z, u, 0.f, n.x, n.y, n.z});
    }
    for(int i = 0; i < segments; i++){
        idx.insert(idx.end(), {0, (unsigned int)(i+2), (unsigned int)(i+1)});
    }
    return buildMesh(v, idx);
}
 
// Oval track (a ring of quads)
Mesh buildOvalTrack(float outerA, float outerB, float innerA, float innerB, int segments)
{
    std::vector<float> v;
    std::vector<unsigned int> idx;
 
    for(int i = 0; i <= segments; i++){
        float a = (float)i / segments * 2.f * PI;
        float ca = cosf(a), sa = sinf(a);
 
        float ox = ca * outerA, oz = sa * outerB;
        float ix = ca * innerA, iz = sa * innerB;
        float u = (float)i / segments;
        glm::vec3 n(0, 1, 0);
 
        // Outer edge
        v.insert(v.end(), {ox, 0.01f, oz, u, 1.f, n.x,n.y,n.z});
        // Inner edge
        v.insert(v.end(), {ix, 0.01f, iz, u, 0.f, n.x,n.y,n.z});
    }
    for(int i = 0; i < segments; i++){
        unsigned int b = i * 2;
        idx.insert(idx.end(), {b, b+2, b+3, b, b+3, b+1});
    }
    return buildMesh(v, idx);
}
 
// Skybox face
Mesh buildSkyFace(int faceID, float s)
{
    glm::vec3 p[4];
    glm::vec2 uv[4] = {{0,0},{1,0},{1,1},{0,1}};
    switch(faceID){
        case 0: p[0]={-s,-s,s}; p[1]={s,-s,s}; p[2]={s,s,s}; p[3]={-s,s,s}; break;
        case 1: p[0]={s,-s,-s}; p[1]={-s,-s,-s}; p[2]={-s,s,-s}; p[3]={s,s,-s}; break;
        case 2: p[0]={-s,-s,-s}; p[1]={-s,-s,s}; p[2]={-s,s,s}; p[3]={-s,s,-s}; break;
        case 3: p[0]={s,-s,s}; p[1]={s,-s,-s}; p[2]={s,s,-s}; p[3]={s,s,s}; break;
        case 4: p[0]={-s,s,s}; p[1]={s,s,s}; p[2]={s,s,-s}; p[3]={-s,s,-s}; break;
        case 5: p[0]={-s,-s,-s}; p[1]={s,-s,-s}; p[2]={s,-s,s}; p[3]={-s,-s,s}; break;
    }
    std::vector<float> verts;
    for(int i = 0; i < 4; i++){
        verts.insert(verts.end(), {p[i].x,p[i].y,p[i].z, uv[i].x,uv[i].y, 0,0,0});
    }
    std::vector<unsigned int> idx = {0,1,2, 0,2,3};
    return buildMesh(verts, idx);
}
 
// HUD fullscreen quad
Mesh buildQuad2D()
{
    // Simple 2D quad covering the whole screen, only pos (no uv/norm)
    float v[] = {-1,-1, 1,-1, 1,1, -1,1};
    unsigned int idx[] = {0,1,2, 0,2,3};
    Mesh m;
    m.indexCount = 6;
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glGenBuffers(1, &m.ebo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    return m;
}
 
// ═════════════════════════════════════════════
//  PROCEDURAL TEXTURES
// ═════════════════════════════════════════════
GLuint makeTexture(int W, int H, const std::vector<unsigned char>& data, bool repeat = true)
{
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    return tex;
}
 
GLuint makeGrassTexture()
{
    const int W=256, H=256;
    std::vector<unsigned char> data(W*H*3);
    for(int y=0;y<H;y++) for(int x=0;x<W;x++){
        unsigned int h = (x*1664525u+y*1013904223u)^(x^y);
        float n = (h&0xFF)/255.f;
        int idx = (y*W+x)*3;
        data[idx+0] = (unsigned char)(30+n*20);
        data[idx+1] = (unsigned char)(100+n*60);
        data[idx+2] = (unsigned char)(20+n*15);
    }
    return makeTexture(W, H, data);
}
 
GLuint makeAsphaltTexture()
{
    const int W=256, H=256;
    std::vector<unsigned char> data(W*H*3);
    for(int y=0;y<H;y++) for(int x=0;x<W;x++){
        unsigned int h = (x*2246822519u+y*3266489917u);
        float n = (h&0xFF)/255.f;
        unsigned char g = (unsigned char)(50+n*25);
        int idx = (y*W+x)*3;
        data[idx+0]=g; data[idx+1]=g; data[idx+2]=g;
        // Road markings: dashed white center line
        float distFromCenter = fabs((float)x / W - 0.5f);
        if(distFromCenter < 0.015f && ((y/20)%3 != 0)){
            data[idx+0]=220; data[idx+1]=220; data[idx+2]=210;
        }
    }
    return makeTexture(W, H, data);
}
 
GLuint makeBuildingTexture()
{
    const int W=128, H=128;
    std::vector<unsigned char> data(W*H*3);
    for(int y=0;y<H;y++) for(int x=0;x<W;x++){
        int idx=(y*W+x)*3;
        // Brick-like pattern
        int brickW=16, brickH=8;
        int row = y / brickH;
        int offsetX = (row % 2 == 0) ? 0 : brickW/2;
        int bx = (x + offsetX) % brickW;
        int by = y % brickH;
 
        unsigned int h = ((x+y*37)*2654435761u);
        float noise = (h&0xFF)/255.f * 0.1f;
 
        if(bx == 0 || by == 0){
            // Mortar lines
            data[idx+0]=(unsigned char)(160); data[idx+1]=(unsigned char)(155); data[idx+2]=(unsigned char)(145);
        } else {
            // Brick color with variation
            data[idx+0]=(unsigned char)(150+noise*50);
            data[idx+1]=(unsigned char)(70+noise*30);
            data[idx+2]=(unsigned char)(55+noise*20);
        }
 
        // Windows (darker rectangles on brick)
        int winCol = (x % 32);
        int winRow = (y % 32);
        if(winCol > 8 && winCol < 24 && winRow > 6 && winRow < 22){
            // Glass
            data[idx+0]=(unsigned char)(60+noise*20);
            data[idx+1]=(unsigned char)(80+noise*30);
            data[idx+2]=(unsigned char)(120+noise*30);
        }
    }
    return makeTexture(W, H, data);
}
 
GLuint makeSkyTexture()
{
    const int W=1024, H=256;
    std::vector<unsigned char> data(W*H*3);
    auto mountainHeight = [](float u) -> float {
        float h = 0.18f*sinf(u*PI*2)+0.10f*sinf(u*PI*5.3f+1.2f)+0.06f*sinf(u*PI*9.7f+0.5f)+0.04f*sinf(u*PI*17.f+2.1f);
        return (h+0.38f)*0.55f;
    };
    for(int y=0;y<H;y++) for(int x=0;x<W;x++){
        float u=(float)x/W, v=(float)y/H;
        float skyT=v;
        unsigned char sr=(unsigned char)(220-skyT*80), sg=(unsigned char)(200-skyT*50), sb=255;
        float haze=std::max(0.f,1.f-v*6.f);
        sr=(unsigned char)std::min(255.f,(float)sr+haze*60);
        sg=(unsigned char)std::min(255.f,(float)sg+haze*40);
        sb=(unsigned char)std::max(0.f,(float)sb-haze*30);
        unsigned char r=sr,g=sg,b=sb;
        float mh=mountainHeight(u);
        if(v<mh){
            float snow=std::max(0.f,(v/mh-0.72f)*5.f);
            float rockT=v/mh;
            unsigned char mr=(unsigned char)(60+rockT*40+snow*180);
            unsigned char mg=(unsigned char)(70+rockT*50+snow*180);
            unsigned char mb=(unsigned char)(55+rockT*35+snow*190);
            float edge=std::min(1.f,(mh-v)/0.03f);
            r=(unsigned char)(mr*edge+sr*(1-edge));
            g=(unsigned char)(mg*edge+sg*(1-edge));
            b=(unsigned char)(mb*edge+sb*(1-edge));
        }
        int idx=(y*W+x)*3;
        data[idx+0]=r; data[idx+1]=g; data[idx+2]=b;
    }
    return makeTexture(W, H, data, false);
}
 
// ═════════════════════════════════════════════
//  SHADOW MAP SETUP
// ═════════════════════════════════════════════
GLuint shadowFBO, shadowDepthTex;
 
void setupShadowMap()
{
    glGenFramebuffers(1, &shadowFBO);
    glGenTextures(1, &shadowDepthTex);
    glBindTexture(GL_TEXTURE_2D, shadowDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_W, SHADOW_H, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = {1,1,1,1};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowDepthTex, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
 
// ═════════════════════════════════════════════
//  COLLISION DETECTION
// ═════════════════════════════════════════════
bool aabbOverlap(const AABB& a, const AABB& b)
{
    return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
           (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
           (a.min.z <= b.max.z && a.max.z >= b.min.z);
}
 
// ═════════════════════════════════════════════
//  SCENE SETUP: STATIC OBJECTS
// ═════════════════════════════════════════════
void setupObjects()
{
    // Track parameters (oval)
    float outerA = 22.f, outerB = 16.f;
 
    // ── BUILDINGS (type=0) ── placed outside the track
    struct { float x, z, sx, sy, sz, rot; } blds[] = {
        { 28, 0,   4, 8, 5,   0},
        {-28, 0,   5, 10, 4,  0},
        { 25, 14,  3, 6, 3,   15},
        {-25, 14,  4, 12, 4,  -10},
        { 25,-14,  5, 7, 5,   5},
        {-25,-14,  3, 9, 3,   -5},
        { 0,  22,  6, 5, 3,   0},
        { 0, -22,  4, 11, 4,  0},
    };
    for(auto& b : blds){
        StaticObject o;
        o.position = glm::vec3(b.x, b.sy/2.f, b.z);
        o.scale    = glm::vec3(b.sx, b.sy, b.sz);
        o.rotationY = b.rot;
        o.type = 0;
        o.bbox = {
            glm::vec3(b.x - b.sx/2.f, 0, b.z - b.sz/2.f),
            glm::vec3(b.x + b.sx/2.f, b.sy, b.z + b.sz/2.f)
        };
        objects.push_back(o);
    }
 
    // ── TREES (type=1) ── scattered around
    struct { float x, z; } trees[] = {
        {30, 10}, {-30, 10}, {30, -10}, {-30, -10},
        {15, 20}, {-15, 20}, {15, -20}, {-15, -20},
    };
    for(auto& t : trees){
        StaticObject o;
        o.position = glm::vec3(t.x, 0, t.z);
        o.scale    = glm::vec3(1, 1, 1); // handled in drawing
        o.rotationY = 0;
        o.type = 1;
        o.bbox = {
            glm::vec3(t.x - 0.3f, 0, t.z - 0.3f),
            glm::vec3(t.x + 0.3f, 4.f, t.z + 0.3f)
        };
        objects.push_back(o);
    }
 
    // ── STREETLIGHTS (type=2) ── along the track
    for(int i = 0; i < 8; i++){
        float a = (float)i / 8 * 2.f * PI;
        float x = cosf(a) * (outerA + 3.f);
        float z = sinf(a) * (outerB + 3.f);
        StaticObject o;
        o.position = glm::vec3(x, 0, z);
        o.scale    = glm::vec3(1, 1, 1);
        o.rotationY = 0;
        o.type = 2;
        o.bbox = {
            glm::vec3(x - 0.2f, 0, z - 0.2f),
            glm::vec3(x + 0.2f, 5.f, z + 0.2f)
        };
        objects.push_back(o);
    }
}
 
// ═════════════════════════════════════════════
//  INPUT
// ═════════════════════════════════════════════
void keyCallback(GLFWwindow* w, int key, int, int action, int)
{
    if(key >= 0 && key < 1024){
        if(action == GLFW_PRESS)   keyStates[key] = true;
        if(action == GLFW_RELEASE) keyStates[key] = false;
    }
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(w, true);
 
    // Toggle camera mode: 1=Free, 2=Follow, 3=Top
    if(action == GLFW_PRESS){
        if(key == GLFW_KEY_1) camMode = CAM_FREE;
        if(key == GLFW_KEY_2) camMode = CAM_FOLLOW;
        if(key == GLFW_KEY_3) camMode = CAM_TOP;
    }
}
 
void mouseCallback(GLFWwindow*, double xpos, double ypos)
{
    if(camMode != CAM_FREE) return;
    if(firstMouse){ lastX=(float)xpos; lastY=(float)ypos; firstMouse=false; }
    float dx=(float)xpos-lastX, dy=lastY-(float)ypos;
    lastX=(float)xpos; lastY=(float)ypos;
    yaw+=dx*0.12f; pitch+=dy*0.12f;
    pitch=glm::clamp(pitch,-89.f,89.f);
    glm::vec3 front;
    front.x=cosf(glm::radians(yaw))*cosf(glm::radians(pitch));
    front.y=sinf(glm::radians(pitch));
    front.z=sinf(glm::radians(yaw))*cosf(glm::radians(pitch));
    camFront=glm::normalize(front);
}
 
void scrollCallback(GLFWwindow*, double, double yoff){
    fov -= (float)yoff;
    fov = glm::clamp(fov, 20.f, 100.f);
}
 
void framebufferSizeCallback(GLFWwindow*, int w, int h){
    glViewport(0, 0, w, h);
}
 
// ═════════════════════════════════════════════
//  DRAWING HELPERS
// ═════════════════════════════════════════════
 
void setModelAndDraw(GLuint prog, const glm::mat4& model, Mesh& mesh)
{
    glUniformMatrix4fv(glGetUniformLocation(prog, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glBindVertexArray(mesh.vao);
    glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, 0);
}
 
// Draw all scene geometry (called for both shadow pass and main pass)
void drawScene(GLuint prog, Mesh& ground, Mesh& track, Mesh& box,
               Mesh& cylinder, Mesh& cone, Mesh& carBody,
               GLuint texGrass, GLuint texAsphalt, GLuint texBuilding,
               bool isShadowPass)
{
    glm::mat4 model;
 
    // ── Ground ──
    if(!isShadowPass){
        glUniform1i(glGetUniformLocation(prog, "useTexture"), 1);
        glUniform1f(glGetUniformLocation(prog, "tilingFactor"), 20.f);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texGrass);
    }
    model = glm::mat4(1.f);
    setModelAndDraw(prog, model, ground);
 
    // ── Track ──
    if(!isShadowPass){
        glBindTexture(GL_TEXTURE_2D, texAsphalt);
        glUniform1f(glGetUniformLocation(prog, "tilingFactor"), 8.f);
    }
    model = glm::mat4(1.f);
    setModelAndDraw(prog, model, track);
 
    // ── Static objects ──
    for(auto& obj : objects){
        model = glm::mat4(1.f);
 
        if(obj.type == 0){
            // BUILDING
            if(!isShadowPass){
                glBindTexture(GL_TEXTURE_2D, texBuilding);
                glUniform1f(glGetUniformLocation(prog, "tilingFactor"), 2.f);
                glUniform1i(glGetUniformLocation(prog, "useTexture"), 1);
            }
            model = glm::translate(model, obj.position);
            model = glm::rotate(model, glm::radians(obj.rotationY), glm::vec3(0,1,0));
            model = glm::scale(model, obj.scale);
            setModelAndDraw(prog, model, box);
        }
        else if(obj.type == 1){
            // TREE: trunk + canopy
            // Trunk
            if(!isShadowPass){
                glUniform1i(glGetUniformLocation(prog, "useTexture"), 0);
                glUniform3f(glGetUniformLocation(prog, "objectColor"), 0.35f, 0.22f, 0.1f);
            }
            model = glm::translate(glm::mat4(1.f), obj.position);
            model = glm::scale(model, glm::vec3(1, 1, 1));
            setModelAndDraw(prog, model, cylinder);
 
            // Canopy (3 stacked cones for fuller look)
            if(!isShadowPass){
                glUniform3f(glGetUniformLocation(prog, "objectColor"), 0.15f, 0.5f, 0.12f);
            }
            for(int c = 0; c < 3; c++){
                model = glm::translate(glm::mat4(1.f), obj.position + glm::vec3(0, 1.5f + c*0.8f, 0));
                model = glm::scale(model, glm::vec3(2.2f - c*0.5f, 2.0f - c*0.3f, 2.2f - c*0.5f));
                setModelAndDraw(prog, model, cone);
            }
        }
        else if(obj.type == 2){
            // STREETLIGHT: pole + lamp head
            // Pole
            if(!isShadowPass){
                glUniform1i(glGetUniformLocation(prog, "useTexture"), 0);
                glUniform3f(glGetUniformLocation(prog, "objectColor"), 0.3f, 0.3f, 0.32f);
            }
            model = glm::translate(glm::mat4(1.f), obj.position);
            model = glm::scale(model, glm::vec3(0.3f, 5.f, 0.3f));
            setModelAndDraw(prog, model, cylinder);
 
            // Lamp head (small box on top)
            if(!isShadowPass){
                glUniform3f(glGetUniformLocation(prog, "objectColor"), 1.f, 0.95f, 0.7f);
            }
            model = glm::translate(glm::mat4(1.f), obj.position + glm::vec3(0, 5.f, 0));
            model = glm::scale(model, glm::vec3(0.6f, 0.3f, 0.6f));
            setModelAndDraw(prog, model, box);
        }
    }
 
    // ── CAR ──
    if(!isShadowPass){
        glUniform1i(glGetUniformLocation(prog, "useTexture"), 0);
        // Flash red on collision
        if(collisionTimer > 0.f)
            glUniform3f(glGetUniformLocation(prog, "objectColor"), 1.f, 0.2f, 0.1f);
        else
            glUniform3f(glGetUniformLocation(prog, "objectColor"), 0.8f, 0.15f, 0.1f);
    }
    // Car body
    model = glm::translate(glm::mat4(1.f), car.position + glm::vec3(0, 0.35f, 0));
    model = glm::rotate(model, glm::radians(car.angle), glm::vec3(0,1,0));
    model = glm::scale(model, glm::vec3(1.2f, 0.5f, 2.4f));
    setModelAndDraw(prog, model, box);
 
    // Car cabin (smaller box on top)
    if(!isShadowPass){
        glUniform3f(glGetUniformLocation(prog, "objectColor"), 0.3f, 0.3f, 0.35f);
    }
    glm::vec3 cabinOffset = glm::vec3(
        sinf(glm::radians(car.angle)) * (-0.3f),
        0.75f,
        cosf(glm::radians(car.angle)) * (-0.3f)
    );
    // Simpler: just do it relative
    model = glm::translate(glm::mat4(1.f), car.position + glm::vec3(0, 0.75f, 0));
    model = glm::rotate(model, glm::radians(car.angle), glm::vec3(0,1,0));
    model = glm::scale(model, glm::vec3(1.0f, 0.4f, 1.4f));
    setModelAndDraw(prog, model, box);
 
    // Wheels (4 small dark cylinders)
    if(!isShadowPass){
        glUniform3f(glGetUniformLocation(prog, "objectColor"), 0.1f, 0.1f, 0.1f);
    }
    float rad = glm::radians(car.angle);
    glm::vec3 fwd(sinf(rad), 0, cosf(rad));  // note: GLM convention
    // Actually let's compute right vector
    glm::vec3 right = glm::normalize(glm::cross(glm::vec3(-sinf(rad), 0, -cosf(rad)), glm::vec3(0,1,0)));
    glm::vec3 forward = glm::vec3(-sinf(rad), 0, -cosf(rad));
 
    glm::vec3 wheelPositions[4] = {
        car.position + right*0.7f + forward*0.9f + glm::vec3(0, 0.2f, 0),
        car.position - right*0.7f + forward*0.9f + glm::vec3(0, 0.2f, 0),
        car.position + right*0.7f - forward*0.9f + glm::vec3(0, 0.2f, 0),
        car.position - right*0.7f - forward*0.9f + glm::vec3(0, 0.2f, 0),
    };
    for(int i = 0; i < 4; i++){
        model = glm::translate(glm::mat4(1.f), wheelPositions[i]);
        model = glm::rotate(model, glm::radians(car.angle + 90.f), glm::vec3(0,1,0));
        model = glm::scale(model, glm::vec3(0.15f, 0.3f, 0.15f));
        setModelAndDraw(prog, model, cylinder);
    }
}
 
// ═════════════════════════════════════════════
//  MAIN
// ═════════════════════════════════════════════
int main()
{
    // ── GLFW ──
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
 
    GLFWwindow* window = glfwCreateWindow(SCR_W, SCR_H, "GPS - Street Circuit", nullptr, nullptr);
    if(!window){ std::cerr << "Window failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
 
    // ── GLAD ──
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr << "GLAD failed\n"; return -1;
    }
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_CULL_FACE);
 
    // ── Shaders ──
    GLuint mainProg   = makeShader(mainVS, mainFS);
    GLuint shadowProg = makeShader(shadowVS, shadowFS);
    GLuint skyProg    = makeShader(skyVS, skyFS);
    GLuint hudProg    = makeShader(hudVS, hudFS);
 
    // ── Textures ──
    GLuint texGrass    = makeGrassTexture();
    GLuint texAsphalt  = makeAsphaltTexture();
    GLuint texBuilding = makeBuildingTexture();
    GLuint texSky      = makeSkyTexture();
 
    // ── Meshes ──
    Mesh ground   = buildGround(50.f);
    Mesh track    = buildOvalTrack(22.f, 16.f, 17.f, 11.f, 128);
    Mesh box      = buildBox();
    Mesh cyl      = buildCylinder(16, 0.5f, 1.f); // unit cylinder, scaled per use
    Mesh cone     = buildCone(16, 0.5f, 1.f);
    Mesh quad2D   = buildQuad2D();
    Mesh skyFaces[6];
    for(int i=0;i<6;i++) skyFaces[i] = buildSkyFace(i, 80.f);
 
    // ── Shadow map ──
    setupShadowMap();
 
    // ── Scene objects ──
    setupObjects();
 
    // ── Light ──
    glm::vec3 lightDir = glm::normalize(glm::vec3(-0.5f, -1.f, -0.3f));
 
    std::cout << "=== GPS Street Circuit ===\n"
              << "  Arrow Up/Down : accelerate / brake\n"
              << "  Arrow Left/Right : steer\n"
              << "  1 : Free camera (WASD + mouse)\n"
              << "  2 : Follow camera (behind car)\n"
              << "  3 : Top-down camera\n"
              << "  Scroll : zoom\n"
              << "  ESC : quit\n";
 
    // ── Render loop ──
    while(!glfwWindowShouldClose(window))
    {
        float now = (float)glfwGetTime();
        deltaTime = now - lastFrame;
        lastFrame = now;
        deltaTime = std::min(deltaTime, 0.05f); // cap
 
        // ────── UPDATE CAR ──────
        // Steering
        if(keyStates[GLFW_KEY_LEFT])  car.angle += car.turnSpeed * deltaTime * (car.speed != 0.f ? 1.f : 0.f);
        if(keyStates[GLFW_KEY_RIGHT]) car.angle -= car.turnSpeed * deltaTime * (car.speed != 0.f ? 1.f : 0.f);
 
        // Acceleration / braking
        if(keyStates[GLFW_KEY_UP])
            car.speed += car.accel * deltaTime;
        else if(keyStates[GLFW_KEY_DOWN])
            car.speed -= car.braking * deltaTime;
        else {
            // Friction
            if(car.speed > 0) car.speed -= car.friction * deltaTime;
            else if(car.speed < 0) car.speed += car.friction * deltaTime;
            if(fabs(car.speed) < 0.1f) car.speed = 0.f;
        }
        car.speed = glm::clamp(car.speed, -car.maxSpeed * 0.4f, car.maxSpeed);
 
        // Move car
        float rad = glm::radians(car.angle);
        glm::vec3 moveDir(-sinf(rad), 0, -cosf(rad));
        glm::vec3 newPos = car.position + moveDir * car.speed * deltaTime;
 
        // ────── COLLISION DETECTION ──────
        Car testCar = car;
        testCar.position = newPos;
        AABB carBox = testCar.getBBox();
        bool blocked = false;
 
        for(auto& obj : objects){
            if(aabbOverlap(carBox, obj.bbox)){
                blocked = true;
                collisionOccurred = true;
                collisionTimer = 0.3f;
                car.speed *= -0.3f; // bounce back
                break;
            }
        }
        if(!blocked){
            car.position = newPos;
        }
 
        if(collisionTimer > 0.f) collisionTimer -= deltaTime;
 
        // ────── UPDATE CAMERA ──────
        if(camMode == CAM_FREE){
            float spd = 10.f * deltaTime;
            if(keyStates[GLFW_KEY_W]) camPos += camFront * spd;
            if(keyStates[GLFW_KEY_S]) camPos -= camFront * spd;
            if(keyStates[GLFW_KEY_A]) camPos -= glm::normalize(glm::cross(camFront,camUp))*spd;
            if(keyStates[GLFW_KEY_D]) camPos += glm::normalize(glm::cross(camFront,camUp))*spd;
            if(keyStates[GLFW_KEY_SPACE])      camPos.y += spd;
            if(keyStates[GLFW_KEY_LEFT_SHIFT]) camPos.y -= spd;
        }
 
        // View matrix
        glm::mat4 view;
        if(camMode == CAM_FREE){
            view = glm::lookAt(camPos, camPos + camFront, camUp);
        }
        else if(camMode == CAM_FOLLOW){
            // Behind and above car
            glm::vec3 behind = car.position - moveDir * 8.f + glm::vec3(0, 4.f, 0);
            glm::vec3 target = car.position + glm::vec3(0, 1.f, 0);
            camPos = behind;
            view = glm::lookAt(behind, target, camUp);
        }
        else { // CAM_TOP
            glm::vec3 above = car.position + glm::vec3(0, 30.f, 0.01f);
            view = glm::lookAt(above, car.position, glm::vec3(0, 0, -1));
        }
 
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        if(fbH == 0) fbH = 1;
        glm::mat4 projection = glm::perspective(glm::radians(fov), (float)fbW/fbH, 0.1f, 300.f);
 
        // ────── SHADOW PASS ──────
        float orthoSize = 55.f;
        glm::mat4 lightProj = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, 0.1f, 120.f);
        glm::vec3 lightPos  = -lightDir * 50.f; // position the "sun"
        glm::mat4 lightView = glm::lookAt(lightPos, glm::vec3(0), glm::vec3(0,1,0));
        glm::mat4 lightSpaceMat = lightProj * lightView;
 
        glViewport(0, 0, SHADOW_W, SHADOW_H);
        glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
        glClear(GL_DEPTH_BUFFER_BIT);
        glUseProgram(shadowProg);
        glUniformMatrix4fv(glGetUniformLocation(shadowProg, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpaceMat));
        // Cull front faces to reduce peter-panning
        glCullFace(GL_FRONT);
        drawScene(shadowProg, ground, track, box, cyl, cone, box, texGrass, texAsphalt, texBuilding, true);
        glCullFace(GL_BACK);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
 
        // ────── MAIN PASS ──────
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.4f, 0.65f, 0.9f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
        // Scene
        glUseProgram(mainProg);
        glUniformMatrix4fv(glGetUniformLocation(mainProg, "view"),       1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(mainProg, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(mainProg, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpaceMat));
        glUniform3fv(glGetUniformLocation(mainProg, "lightDir"),   1, glm::value_ptr(lightDir));
        glUniform3fv(glGetUniformLocation(mainProg, "lightColor"), 1, &glm::vec3(1.f, 0.95f, 0.85f)[0]);
        glUniform3fv(glGetUniformLocation(mainProg, "ambientColor"),1, &glm::vec3(0.25f, 0.28f, 0.35f)[0]);
        glUniform3fv(glGetUniformLocation(mainProg, "viewPos"),    1, glm::value_ptr(camPos));
 
        // Point lights (streetlights)
        int numPL = 0;
        for(auto& obj : objects){
            if(obj.type == 2 && numPL < 8){
                glm::vec3 lp = obj.position + glm::vec3(0, 4.8f, 0);
                glm::vec3 lc(1.f, 0.85f, 0.5f);
                std::string base = "pointLightPos[" + std::to_string(numPL) + "]";
                std::string basec = "pointLightColor[" + std::to_string(numPL) + "]";
                glUniform3fv(glGetUniformLocation(mainProg, base.c_str()), 1, glm::value_ptr(lp));
                glUniform3fv(glGetUniformLocation(mainProg, basec.c_str()), 1, glm::value_ptr(lc));
                numPL++;
            }
        }
        glUniform1i(glGetUniformLocation(mainProg, "numPointLights"), numPL);
 
        // Bind shadow map to texture unit 1
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, shadowDepthTex);
        glUniform1i(glGetUniformLocation(mainProg, "shadowMap"), 1);
        glActiveTexture(GL_TEXTURE0);
        glUniform1i(glGetUniformLocation(mainProg, "tex"), 0);
 
        drawScene(mainProg, ground, track, box, cyl, cone, box, texGrass, texAsphalt, texBuilding, false);
 
        // ── Skybox ──
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);
        glUseProgram(skyProg);
        glUniformMatrix4fv(glGetUniformLocation(skyProg,"view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(skyProg,"projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texSky);
        glUniform1i(glGetUniformLocation(skyProg,"texSky"), 0);
        for(int i=0;i<6;i++){
            glUniform1i(glGetUniformLocation(skyProg,"faceID"), i);
            glBindVertexArray(skyFaces[i].vao);
            glDrawElements(GL_TRIANGLES, skyFaces[i].indexCount, GL_UNSIGNED_INT, 0);
        }
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);
 
        // ── HUD: collision flash ──
        if(collisionTimer > 0.f){
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDisable(GL_DEPTH_TEST);
            glUseProgram(hudProg);
            float alpha = collisionTimer / 0.3f * 0.3f;
            glUniform4f(glGetUniformLocation(hudProg, "color"), 1.f, 0.f, 0.f, alpha);
            glBindVertexArray(quad2D.vao);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            glEnable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
        }
 
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
 
    glfwTerminate();
    return 0;
}
 
