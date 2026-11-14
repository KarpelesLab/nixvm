/*
 * gpu_scene: a small scene through EGL (surfaceless) + GLES, read back to a
 * PPM, to compare nixvm's virtio-gpu against Mesa's software rasterizer.
 * Like gpu_triangle.c it is not part of the conformance set (it needs Mesa).
 *
 *   gcc -O2 -o gpu_scene gpu_scene.c -lEGL -lGLESv2 -lm
 *   nixvm run --root <root> --workdir <dir> -- /gpu_scene /work/ref.ppm        # llvmpipe
 *   NIXVM_GPU=egl nixvm run ... -- /gpu_scene /work/gpu.ppm                    # virgl
 *
 * It draws what es2gears draws — es2gears' own GLSL ES 1.00 lighting shader
 * on indexed, depth-tested geometry — plus a textured, alpha-blended quad
 * with a GLSL ES 3.00 shader, a uniform block, a scissored clear and a
 * second render target size, so the output exercises most of the state a
 * GLES 2/3 application sets.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 128
#define H 96

/* es2gears' shaders, verbatim. */
static const char *gears_vs =
    "attribute vec3 position;\n"
    "attribute vec3 normal;\n"
    "\n"
    "uniform mat4 ModelViewProjectionMatrix;\n"
    "uniform mat4 NormalMatrix;\n"
    "uniform vec4 LightSourcePosition;\n"
    "uniform vec4 MaterialColor;\n"
    "\n"
    "varying vec4 Color;\n"
    "\n"
    "void main(void)\n"
    "{\n"
    "    // Transform the normal to eye coordinates\n"
    "    vec3 N = normalize(vec3(NormalMatrix * vec4(normal, 1.0)));\n"
    "\n"
    "    // The LightSourcePosition is actually its direction for directional light\n"
    "    vec3 L = normalize(LightSourcePosition.xyz);\n"
    "\n"
    "    // Multiply the diffuse value by the vertex color (which is fixed in this case)\n"
    "    // to get the actual color that we will use to draw this vertex with\n"
    "    float diffuse = max(dot(N, L), 0.0);\n"
    "    Color = diffuse * MaterialColor;\n"
    "\n"
    "    // Transform the position to clip coordinates\n"
    "    gl_Position = ModelViewProjectionMatrix * vec4(position, 1.0);\n"
    "}";
static const char *gears_fs =
    "precision mediump float;\n"
    "varying vec4 Color;\n"
    "\n"
    "void main(void)\n"
    "{\n"
    "    gl_FragColor = Color;\n"
    "}";

static const char *tex_vs =
    "#version 300 es\n"
    "layout(location = 0) in vec2 pos;\n"
    "out vec2 uv;\n"
    "layout(std140) uniform Params { vec4 offset_scale; vec4 tint; };\n"
    "void main() {\n"
    "  uv = pos * 0.5 + 0.5;\n"
    "  gl_Position = vec4(pos * offset_scale.zw + offset_scale.xy, 0.0, 1.0);\n"
    "}\n";
static const char *tex_fs =
    "#version 300 es\n"
    "precision mediump float;\n"
    "in vec2 uv;\n"
    "uniform sampler2D tex;\n"
    "layout(std140) uniform Params { vec4 offset_scale; vec4 tint; };\n"
    "out vec4 o;\n"
    "void main() { o = texture(tex, uv) * tint; }\n";

static GLuint shader(GLenum kind, const char *src)
{
    GLuint s = glCreateShader(kind);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "shader: %s\n", log);
        exit(1);
    }
    return s;
}

static GLuint program(const char *vs, const char *fs, int bind_attribs)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, shader(GL_VERTEX_SHADER, vs));
    glAttachShader(p, shader(GL_FRAGMENT_SHADER, fs));
    if (bind_attribs) {
        glBindAttribLocation(p, 0, "position");
        glBindAttribLocation(p, 1, "normal");
    }
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        fprintf(stderr, "link failed\n");
        exit(1);
    }
    return p;
}

/* Column-major 4x4 helpers. */
static void mul(float *r, const float *a, const float *b)
{
    float t[16];
    for (int c = 0; c < 4; c++)
        for (int i = 0; i < 4; i++) {
            t[c * 4 + i] = 0;
            for (int k = 0; k < 4; k++)
                t[c * 4 + i] += a[k * 4 + i] * b[c * 4 + k];
        }
    memcpy(r, t, sizeof t);
}
static void rot(float *m, float deg, float x, float y, float z)
{
    float s = sinf(deg * (float)M_PI / 180), c = cosf(deg * (float)M_PI / 180);
    float r[16] = {x * x * (1 - c) + c, y * x * (1 - c) + z * s, x * z * (1 - c) - y * s, 0,
                   x * y * (1 - c) - z * s, y * y * (1 - c) + c, y * z * (1 - c) + x * s, 0,
                   x * z * (1 - c) + y * s, y * z * (1 - c) - x * s, z * z * (1 - c) + c, 0,
                   0, 0, 0, 1};
    mul(m, m, r);
}

int main(int argc, char **argv)
{
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay dpy = get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    if (!eglInitialize(dpy, NULL, NULL)) {
        fprintf(stderr, "eglInitialize failed\n");
        return 1;
    }
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint ctx_attrs[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attrs);
    if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
        fprintf(stderr, "no GLES 3 context\n");
        return 1;
    }
    printf("GL_RENDERER: %s\n", glGetString(GL_RENDERER));

    /* Colour + depth render targets. */
    GLuint fb, color, depth;
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glGenRenderbuffers(1, &color);
    glBindRenderbuffer(GL_RENDERBUFFER, color);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, W, H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color);
    glGenRenderbuffers(1, &depth);
    glBindRenderbuffer(GL_RENDERBUFFER, depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, W, H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "framebuffer incomplete\n");
        return 1;
    }
    glViewport(0, 0, W, H);
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    /* A scissored clear: a band across the top. */
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, H - 12, W, 12);
    glClearColor(0.6f, 0.3f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);

    /* A cube: 24 vertices (position, normal), 36 indices. */
    static const float n[6][3] = {{0, 0, 1}, {0, 0, -1}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
    float verts[24 * 6];
    unsigned short idx[36];
    for (int f = 0; f < 6; f++) {
        float u[3] = {n[f][1], n[f][2], n[f][0]}, v[3];
        v[0] = n[f][1] * u[2] - n[f][2] * u[1];
        v[1] = n[f][2] * u[0] - n[f][0] * u[2];
        v[2] = n[f][0] * u[1] - n[f][1] * u[0];
        for (int k = 0; k < 4; k++) {
            float su = (k & 1) ? 1 : -1, sv = (k & 2) ? 1 : -1;
            float *p = &verts[(f * 4 + k) * 6];
            for (int i = 0; i < 3; i++) {
                p[i] = 0.5f * (n[f][i] + su * u[i] + sv * v[i]);
                p[3 + i] = n[f][i];
            }
        }
        unsigned short b = f * 4, q[6] = {0, 1, 2, 2, 1, 3};
        for (int i = 0; i < 6; i++)
            idx[f * 6 + i] = b + q[i];
    }
    GLuint vbo, ibo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);

    GLuint gears = program(gears_vs, gears_fs, 1);
    glUseProgram(gears);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, (void *)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, (void *)12);
    static const float light[4] = {5.0f, 5.0f, 10.0f, 1.0f};
    glUniform4fv(glGetUniformLocation(gears, "LightSourcePosition"), 1, light);
    static const float colors[2][4] = {{0.8f, 0.1f, 0.0f, 1.0f}, {0.0f, 0.8f, 0.2f, 1.0f}};
    for (int i = 0; i < 2; i++) {
        float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        m[12] = i ? 0.35f : -0.35f;
        m[14] = i ? -0.2f : 0.0f;
        rot(m, 30.0f + i * 20, 1, 0, 0);
        rot(m, 40.0f + i * 15, 0, 1, 0);
        float s[16] = {0.9f * H / W, 0, 0, 0, 0, 0.9f, 0, 0, 0, 0, 0.5f, 0, 0, 0, 0, 1};
        float mvp[16];
        mul(mvp, s, m);
        glUniformMatrix4fv(glGetUniformLocation(gears, "ModelViewProjectionMatrix"), 1, GL_FALSE, mvp);
        glUniformMatrix4fv(glGetUniformLocation(gears, "NormalMatrix"), 1, GL_FALSE, m);
        glUniform4fv(glGetUniformLocation(gears, "MaterialColor"), 1, colors[i]);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
    }
    glDisable(GL_CULL_FACE);
    glDisableVertexAttribArray(1);

    /* A textured, blended quad in the lower-left corner. */
    static const unsigned char checker[4 * 4 * 4] = {
        255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255,
        0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255,
        0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255,
    };
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, checker);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    GLuint texprog = program(tex_vs, tex_fs, 0);
    glUseProgram(texprog);
    glUniform1i(glGetUniformLocation(texprog, "tex"), 0);
    static const float params[8] = {-0.7f, -0.6f, 0.25f, 0.33f, 1.0f, 1.0f, 0.3f, 0.6f};
    GLuint ubo;
    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof params, params, GL_STATIC_DRAW);
    glUniformBlockBinding(texprog, glGetUniformBlockIndex(texprog, "Params"), 1);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);
    static const float quad[8] = {-1, -1, 1, -1, -1, 1, 1, 1};
    GLuint qbo;
    glGenBuffers(1, &qbo);
    glBindBuffer(GL_ARRAY_BUFFER, qbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisable(GL_BLEND);

    static unsigned char px[W * H * 4];
    glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
    GLenum e = glGetError();
    printf("glGetError: 0x%x\n", e);
    unsigned long sum = 0;
    for (int i = 0; i < W * H * 4; i++)
        sum = sum * 31 + px[i];
    printf("checksum %08lx\n", sum & 0xffffffff);
    if (argc > 1) {
        FILE *f = fopen(argv[1], "wb");
        if (!f) {
            perror(argv[1]);
            return 1;
        }
        fprintf(f, "P6\n%d %d\n255\n", W, H);
        for (int y = H - 1; y >= 0; y--)
            for (int x = 0; x < W; x++)
                fwrite(&px[(y * W + x) * 4], 1, 3, f);
        fclose(f);
    }
    return e == GL_NO_ERROR ? 0 : 1;
}
