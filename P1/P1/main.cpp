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

// stb_image - header-only, define implementation in ONE .cpp file
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

// ─────────────────────────────────────────────
//  Window / camera globals
// ─────────────────────────────────────────────
const int SCR_W = 1280, SCR_H = 720;

glm::vec3 camPos   = glm::vec3(0.f, 2.5f, 8.f);
glm::vec3 camFront = glm::vec3(0.f, -0.15f, -1.f);
glm::vec3 camUp    = glm::vec3(0.f,  1.f,   0.f);

float yaw   = -90.f, pitch = -8.f;
float lastX = SCR_W / 2.f, lastY = SCR_H / 2.f;
bool  firstMouse = true;
float deltaTime = 0.f, lastFrame = 0.f;
float fov = 60.f;

// ─────────────────────────────────────────────
//  Shader source strings
// ─────────────────────────────────────────────

// --- SCENE shader (ground + terrain) ---
const char* sceneVS = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;
layout(location=2) in vec3 aNorm;

uniform mat4 model, view, projection;

out vec2  vUV;
out vec3  vNorm;
out vec3  vFragPos;

void main(){
    vec4 worldPos = model * vec4(aPos, 1.0);
    vFragPos = worldPos.xyz;
    vNorm    = mat3(transpose(inverse(model))) * aNorm;
    vUV      = aUV;
    gl_Position = projection * view * worldPos;
}
)";

const char* sceneFS = R"(
#version 330 core
in vec2  vUV;
in vec3  vNorm;
in vec3  vFragPos;

uniform sampler2D tex;
uniform vec3  lightDir;
uniform vec3  lightColor;
uniform vec3  ambientColor;
uniform float tilingFactor;

out vec4 FragColor;

void main(){
    vec3 norm    = normalize(vNorm);
    float diff   = max(dot(norm, normalize(-lightDir)), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 ambient = ambientColor;

    vec3 texColor = texture(tex, vUV * tilingFactor).rgb;
    vec3 result   = (ambient + diffuse) * texColor;
    FragColor     = vec4(result, 1.0);
}
)";

// --- SKYBOX shader ---
const char* skyVS = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;

uniform mat4 view, projection;

out vec2 vUV;
out float vY;   // used for sky gradient blend

void main(){
    vUV = aUV;
    vY  = aPos.y;
    // Remove translation from view
    vec4 pos = projection * mat4(mat3(view)) * vec4(aPos, 1.0);
    gl_Position = pos.xyww;  // depth = 1.0 (always behind)
}
)";

const char* skyFS = R"(
#version 330 core
in vec2  vUV;
in float vY;

uniform sampler2D texSky;   // horizon panorama
uniform int       faceID;   // 0=front,1=back,2=left,3=right,4=top,5=bottom

out vec4 FragColor;

void main(){
    vec3 col;

    if(faceID == 4){
        // Top face: deep sky blue gradient
        col = mix(vec3(0.40, 0.65, 0.95), vec3(0.15, 0.35, 0.75), vUV.y);
    } else if(faceID == 5){
        // Bottom: dark earth
        col = vec3(0.22, 0.18, 0.12);
    } else {
        // Side faces: use panorama texture
        col = texture(texSky, vUV).rgb;
    }
    FragColor = vec4(col, 1.0);
}
)";

// ─────────────────────────────────────────────
//  Utility: compile + link shaders
// ─────────────────────────────────────────────
GLuint makeShader(const char* vs, const char* fs)
{
    auto compile = [](GLenum type, const char* src) -> GLuint {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if(!ok){
            char log[512]; glGetShaderInfoLog(s, 512, nullptr, log);
            std::cerr << "Shader error:\n" << log << "\n";
        }
        return s;
    };
    GLuint v = compile(GL_VERTEX_SHADER,   vs);
    GLuint f = compile(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// ─────────────────────────────────────────────
//  Procedural texture generation (no files needed)
// ─────────────────────────────────────────────

// Grass texture: green base + noise
GLuint makeGrassTexture()
{
    const int W = 256, H = 256;
    std::vector<unsigned char> data(W * H * 3);
    for(int y = 0; y < H; y++)
    for(int x = 0; x < W; x++){
        // simple hash noise
        unsigned int h = (x * 1664525u + y * 1013904223u) ^ (x ^ y);
        float n = (h & 0xFF) / 255.f;

        // green grass with variation
        unsigned char r = (unsigned char)(30  + n * 20);
        unsigned char g = (unsigned char)(100 + n * 60);
        unsigned char b = (unsigned char)(20  + n * 15);

        // occasional darker blades
        if((h & 0x7) == 0){ g -= 30; }

        int idx = (y * W + x) * 3;
        data[idx+0] = r; data[idx+1] = g; data[idx+2] = b;
    }
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    return tex;
}

// Sky panorama texture: sky gradient + mountain silhouette + clouds
GLuint makeSkyTexture()
{
    const int W = 1024, H = 256;
    std::vector<unsigned char> data(W * H * 3);

    auto setPixel = [&](int x, int y, unsigned char r, unsigned char g, unsigned char b){
        int idx = (y * W + x) * 3;
        data[idx+0] = r; data[idx+1] = g; data[idx+2] = b;
    };

    // Mountain profile function (sum of sine waves)
    auto mountainHeight = [&](float u) -> float {
        float h = 0.f;
        h += 0.18f * sinf(u * 3.14159f * 2.f);
        h += 0.10f * sinf(u * 3.14159f * 5.3f + 1.2f);
        h += 0.06f * sinf(u * 3.14159f * 9.7f + 0.5f);
        h += 0.04f * sinf(u * 3.14159f * 17.f + 2.1f);
        h = (h + 0.38f) * 0.55f; // normalise to ~[0.12, 0.52]
        return h;
    };

    // Cloud noise
    auto cloudNoise = [](int x, int y) -> float {
        unsigned int h = (x * 374761393u) ^ (y * 668265263u);
        h = (h ^ (h >> 13)) * 1274126177u;
        return (h & 0xFF) / 255.f;
    };

    for(int y = 0; y < H; y++)
    for(int x = 0; x < W; x++){
        float u = (float)x / W;
        float v = (float)y / H;  // 0=bottom, 1=top of panorama

        // Sky gradient  (bottom of tex = horizon)
        float skyT = v;
        unsigned char sr = (unsigned char)(220 - skyT * 80);
        unsigned char sg = (unsigned char)(200 - skyT * 50);
        unsigned char sb = (unsigned char)(255);

        // Horizon haze (warm glow)
        float haze = std::max(0.f, 1.f - v * 6.f);
        sr = (unsigned char)(sr + haze * 60);
        sg = (unsigned char)(sg + haze * 40);
        sb = (unsigned char)(sb - haze * 30);

        unsigned char r = sr, g = sg, b = sb;

        // Mountain layer
        float mh = mountainHeight(u);
        if(v < mh){
            // mountain pixel
            float snow = std::max(0.f, (v / mh - 0.72f) * 5.f);
            float rockT = v / mh;

            unsigned char mr = (unsigned char)(60  + rockT * 40  + snow * 180);
            unsigned char mg = (unsigned char)(70  + rockT * 50  + snow * 180);
            unsigned char mb = (unsigned char)(55  + rockT * 35  + snow * 190);

            // blend mountain edge with sky
            float edge = std::min(1.f, (mh - v) / 0.03f);
            r = (unsigned char)(mr * edge + sr * (1.f - edge));
            g = (unsigned char)(mg * edge + sg * (1.f - edge));
            b = (unsigned char)(mb * edge + sb * (1.f - edge));
        }

        // Clouds (upper part of sky)
        if(v > 0.45f){
            float cn = 0.f;
            // multi-octave cloud noise
            cn += cloudNoise(x/8, y/8)   * 0.5f;
            cn += cloudNoise(x/4, y/4)   * 0.3f;
            cn += cloudNoise(x/2, y/2)   * 0.2f;
            float cloudMask = std::max(0.f, cn - 0.52f) * 3.5f;
            cloudMask = std::min(1.f, cloudMask);
            // cloud colour: white with slight blue tint
            r = (unsigned char)(r + cloudMask * (240 - r));
            g = (unsigned char)(g + cloudMask * (245 - g));
            b = (unsigned char)(b + cloudMask * (255 - b));
        }

        setPixel(x, y, r, g, b);
    }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    return tex;
}

// Rock/dirt texture for terrain hills
GLuint makeRockTexture()
{
    const int W = 256, H = 256;
    std::vector<unsigned char> data(W * H * 3);
    for(int y = 0; y < H; y++)
    for(int x = 0; x < W; x++){
        unsigned int h = (x * 2246822519u + y * 3266489917u) ^ ((x+y) * 668265263u);
        float n = (h & 0xFF) / 255.f;
        float n2 = ((h >> 8) & 0xFF) / 255.f;

        unsigned char r = (unsigned char)(100 + n * 60 + n2 * 20);
        unsigned char g = (unsigned char)(85  + n * 50 + n2 * 15);
        unsigned char b = (unsigned char)(65  + n * 40 + n2 * 10);

        int idx = (y * W + x) * 3;
        data[idx+0] = r; data[idx+1] = g; data[idx+2] = b;
    }
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    return tex;
}

// ─────────────────────────────────────────────
//  Mesh helpers
// ─────────────────────────────────────────────
struct Mesh {
    GLuint vao = 0, vbo = 0, ebo = 0;
    int    indexCount = 0;
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

    // pos(3) uv(2) norm(3) → stride = 8 floats
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(5*sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    return m;
}

// Flat ground plane (XZ, y=0)
Mesh buildGround(float halfSize, int divs)
{
    std::vector<float> verts;
    std::vector<unsigned int> idx;

    float step = 2.f * halfSize / divs;
    for(int z = 0; z <= divs; z++)
    for(int x = 0; x <= divs; x++){
        float px = -halfSize + x * step;
        float pz = -halfSize + z * step;
        float u  = (float)x / divs;
        float v  = (float)z / divs;
        // pos, uv, normal
        verts.insert(verts.end(), {px, 0.f, pz,  u, v,  0.f, 1.f, 0.f});
    }
    for(int z = 0; z < divs; z++)
    for(int x = 0; x < divs; x++){
        unsigned int tl = z*(divs+1)+x;
        unsigned int tr = tl+1;
        unsigned int bl = tl+(divs+1);
        unsigned int br = bl+1;
        idx.insert(idx.end(), {tl,bl,tr, tr,bl,br});
    }
    return buildMesh(verts, idx);
}

// Heightmap terrain inside the box
// Uses sine-wave hills with a noise layer
Mesh buildTerrain(float halfSize, int divs, float maxHeight)
{
    std::vector<float> verts;
    std::vector<unsigned int> idx;

    float step = 2.f * halfSize / divs;

    // Height function
    auto height = [&](float x, float z) -> float {
        float h = 0.f;
        h += 0.40f * sinf(x * 0.45f) * cosf(z * 0.35f);
        h += 0.25f * sinf(x * 0.9f  + 0.8f) * sinf(z * 0.7f + 1.1f);
        h += 0.15f * sinf(x * 1.8f  + 2.3f) * cosf(z * 1.5f + 0.4f);
        h += 0.10f * cosf(x * 3.0f  + 1.0f) * sinf(z * 2.8f + 2.0f);
        return h * maxHeight;
    };

    for(int z = 0; z <= divs; z++)
    for(int x = 0; x <= divs; x++){
        float px = -halfSize + x * step;
        float pz = -halfSize + z * step;
        float py = height(px, pz);
        float u  = (float)x / divs;
        float v  = (float)z / divs;

        // Approximate normal via finite difference
        float eps = 0.1f;
        float hL = height(px - eps, pz);
        float hR = height(px + eps, pz);
        float hD = height(px, pz - eps);
        float hU = height(px, pz + eps);
        glm::vec3 norm = glm::normalize(glm::vec3(hL - hR, 2.f*eps, hD - hU));

        verts.insert(verts.end(), {px, py, pz,  u, v,  norm.x, norm.y, norm.z});
    }
    for(int z = 0; z < divs; z++)
    for(int x = 0; x < divs; x++){
        unsigned int tl = z*(divs+1)+x;
        unsigned int tr = tl+1;
        unsigned int bl = tl+(divs+1);
        unsigned int br = bl+1;
        idx.insert(idx.end(), {tl,bl,tr, tr,bl,br});
    }
    return buildMesh(verts, idx);
}

// One quad face of the skybox cube
// faceID: 0=front(+Z) 1=back(-Z) 2=left(-X) 3=right(+X) 4=top(+Y) 5=bottom(-Y)
Mesh buildSkyFace(int faceID, float s)
{
    // Each face: 4 verts with pos + uv; normal unused for sky
    glm::vec3 p[4];
    glm::vec2 uv[4] = {{0,0},{1,0},{1,1},{0,1}};

    switch(faceID){
    case 0: // front +Z
        p[0]={-s,-s, s}; p[1]={ s,-s, s}; p[2]={ s, s, s}; p[3]={-s, s, s}; break;
    case 1: // back -Z
        p[0]={ s,-s,-s}; p[1]={-s,-s,-s}; p[2]={-s, s,-s}; p[3]={ s, s,-s}; break;
    case 2: // left -X
        p[0]={-s,-s,-s}; p[1]={-s,-s, s}; p[2]={-s, s, s}; p[3]={-s, s,-s}; break;
    case 3: // right +X
        p[0]={ s,-s, s}; p[1]={ s,-s,-s}; p[2]={ s, s,-s}; p[3]={ s, s, s}; break;
    case 4: // top +Y
        p[0]={-s, s, s}; p[1]={ s, s, s}; p[2]={ s, s,-s}; p[3]={-s, s,-s}; break;
    case 5: // bottom -Y
        p[0]={-s,-s,-s}; p[1]={ s,-s,-s}; p[2]={ s,-s, s}; p[3]={-s,-s, s}; break;
    }

    std::vector<float> verts;
    for(int i = 0; i < 4; i++){
        verts.insert(verts.end(), {
            p[i].x, p[i].y, p[i].z,
            uv[i].x, uv[i].y,
            0.f, 0.f, 0.f  // normal unused
        });
    }
    std::vector<unsigned int> idx = {0,1,2, 0,2,3};
    return buildMesh(verts, idx);
}

// ─────────────────────────────────────────────
//  Camera / input callbacks
// ─────────────────────────────────────────────
void updateCameraVectors(){
    glm::vec3 front;
    front.x = cosf(glm::radians(yaw)) * cosf(glm::radians(pitch));
    front.y = sinf(glm::radians(pitch));
    front.z = sinf(glm::radians(yaw)) * cosf(glm::radians(pitch));
    camFront = glm::normalize(front);
}

void mouseCallback(GLFWwindow*, double xpos, double ypos)
{
    if(firstMouse){ lastX = (float)xpos; lastY = (float)ypos; firstMouse = false; }
    float dx = (float)xpos - lastX;
    float dy = lastY - (float)ypos;
    lastX = (float)xpos; lastY = (float)ypos;
    float sens = 0.12f;
    yaw   += dx * sens;
    pitch += dy * sens;
    pitch  = glm::clamp(pitch, -89.f, 89.f);
    updateCameraVectors();
}

void scrollCallback(GLFWwindow*, double, double yoff){
    fov -= (float)yoff;
    fov  = glm::clamp(fov, 20.f, 90.f);
}

void processInput(GLFWwindow* w)
{
    float speed = 5.f * deltaTime;
    if(glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) camPos += speed * camFront;
    if(glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) camPos -= speed * camFront;
    if(glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS)
        camPos -= glm::normalize(glm::cross(camFront, camUp)) * speed;
    if(glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS)
        camPos += glm::normalize(glm::cross(camFront, camUp)) * speed;
    if(glfwGetKey(w, GLFW_KEY_SPACE)      == GLFW_PRESS) camPos.y += speed;
    if(glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) camPos.y -= speed;
    if(glfwGetKey(w, GLFW_KEY_ESCAPE)     == GLFW_PRESS) glfwSetWindowShouldClose(w, true);
}

void framebufferSizeCallback(GLFWwindow*, int w, int h){
    glViewport(0, 0, w, h);
}

// ─────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────
int main()
{
    // ----- GLFW init -----
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);  // MSAA

    GLFWwindow* window = glfwCreateWindow(SCR_W, SCR_H, "GPS Scene S4", nullptr, nullptr);
    if(!window){ std::cerr << "GLFW window failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // ----- GLAD init -----
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr << "GLAD init failed\n"; return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    // ----- Shaders -----
    GLuint sceneProg = makeShader(sceneVS, sceneFS);
    GLuint skyProg   = makeShader(skyVS,   skyFS);

    // ----- Textures -----
    GLuint texGrass = makeGrassTexture();
    GLuint texSky   = makeSkyTexture();
    GLuint texRock  = makeRockTexture();

    // ----- Meshes -----
    // Ground plane: big flat quad at y = 0
    Mesh ground  = buildGround(20.f, 1);

    // Terrain: hilly mesh, slightly above ground, inside the scene
    Mesh terrain = buildTerrain(18.f, 120, 2.8f);

    // Skybox faces
    Mesh skyFaces[6];
    for(int i = 0; i < 6; i++) skyFaces[i] = buildSkyFace(i, 50.f);

    // ----- Light -----
    glm::vec3 lightDir   = glm::normalize(glm::vec3(-0.4f, -1.f, -0.3f));
    glm::vec3 lightColor = glm::vec3(1.0f, 0.95f, 0.85f);
    glm::vec3 ambient    = glm::vec3(0.35f, 0.38f, 0.45f);

    std::cout << "=== GPS Scene S4 ===\n"
              << "  WASD        : move\n"
              << "  Mouse       : look\n"
              << "  Space/Shift : up/down\n"
              << "  Scroll      : zoom\n"
              << "  ESC         : quit\n";

    // ----- Render loop -----
    while(!glfwWindowShouldClose(window))
    {
        float now = (float)glfwGetTime();
        deltaTime = now - lastFrame;
        lastFrame = now;

        processInput(window);

        glClearColor(0.4f, 0.65f, 0.9f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        if(fbH == 0) fbH = 1;

        glm::mat4 view       = glm::lookAt(camPos, camPos + camFront, camUp);
        glm::mat4 projection = glm::perspective(glm::radians(fov),
                                                (float)fbW / fbH, 0.1f, 200.f);

        // ── Draw skybox LAST, with LEQUAL depth so it renders behind everything ──
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);
        glUseProgram(skyProg);
        glUniformMatrix4fv(glGetUniformLocation(skyProg, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(skyProg, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texSky);
        glUniform1i(glGetUniformLocation(skyProg, "texSky"), 0);
        for (int i = 0; i < 6; i++) {
            glUniform1i(glGetUniformLocation(skyProg, "faceID"), i);
            glBindVertexArray(skyFaces[i].vao);
            glDrawElements(GL_TRIANGLES, skyFaces[i].indexCount, GL_UNSIGNED_INT, 0);
        }
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);  // restore default

        // ── Draw ground (grass) ──
        glUseProgram(sceneProg);
        glUniformMatrix4fv(glGetUniformLocation(sceneProg,"view"),       1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(sceneProg,"projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform3fv(glGetUniformLocation(sceneProg,"lightDir"),   1, glm::value_ptr(lightDir));
        glUniform3fv(glGetUniformLocation(sceneProg,"lightColor"), 1, glm::value_ptr(lightColor));
        glUniform3fv(glGetUniformLocation(sceneProg,"ambientColor"),1, glm::value_ptr(ambient));
        glUniform1f (glGetUniformLocation(sceneProg,"tilingFactor"), 12.f);

        glm::mat4 model = glm::mat4(1.f);
        glUniformMatrix4fv(glGetUniformLocation(sceneProg,"model"), 1, GL_FALSE, glm::value_ptr(model));
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texGrass);
        glUniform1i(glGetUniformLocation(sceneProg,"tex"), 0);
        glBindVertexArray(ground.vao);
        glDrawElements(GL_TRIANGLES, ground.indexCount, GL_UNSIGNED_INT, 0);

        // ── Draw terrain (rock/grass hills) ──
        // blend grass on lower areas, rock on higher — done via tiling & same grass tex
        // We raise terrain slightly so it sits on top of ground
        model = glm::translate(glm::mat4(1.f), glm::vec3(0.f, 0.02f, 0.f));
        glUniformMatrix4fv(glGetUniformLocation(sceneProg,"model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniform1f(glGetUniformLocation(sceneProg,"tilingFactor"), 8.f);
        glBindTexture(GL_TEXTURE_2D, texRock);
        glBindVertexArray(terrain.vao);
        glDrawElements(GL_TRIANGLES, terrain.indexCount, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
