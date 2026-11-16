/*
 * gpu_gl33: desktop OpenGL 3.3 core through EGL (surfaceless), read back to
 * a PPM, to compare nixvm's virtio-gpu (which emulates desktop GL 3.3 on a
 * GLES 3.0 / WebGL2 host) against Mesa's software rasterizer. Like
 * gpu_scene.c it needs Mesa and is not part of the conformance set.
 *
 *   gcc -O2 -o gpu_gl33 gpu_gl33.c -lEGL -lGL -lm
 *   nixvm run --root <root> --workdir <dir> -- /gpu_gl33 /work/ref.ppm   # llvmpipe
 *   NIXVM_GPU=egl nixvm run ... -- /gpu_gl33 /work/gpu.ppm               # virgl
 *
 * Each 64x64 tile of the 256x192 picture exercises something desktop GL 3.x
 * has and GLES 3.0 lacks (or names differently), so the renderer has to
 * emulate it: texture buffers, texture swizzles (as GL_LUMINANCE needs),
 * BGRA and 2_10_10_10 vertex formats, dual-source blending, clip distances,
 * depth clamp, per-buffer blend enables and colour masks, multisampling
 * resolved by a blit, a multisampled texture fetched in a shader, transform
 * feedback (captured, then drawn from), point sizes from glPointSize, and
 * conditional rendering. Transform feedback's numbers and the queries'
 * results are printed too.
 */
#define GL_GLEXT_PROTOTYPES 1
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 256
#define H 192
#define T 64

/* On a GL 3.2 context (GL32 set), "#version 330 core" becomes GLSL 1.50
 * with explicit attribute locations. */
static int gl32;

static const char *glsl(const char *s)
{
    static const char v330[] = "#version 330 core\n";
    if (!gl32 || strncmp(s, v330, sizeof v330 - 1))
        return s;
    static char buf[8][4096];
    static int n;
    char *b = buf[n++ % 8];
    snprintf(b, sizeof buf[0], "#version 150 core\n#extension GL_ARB_explicit_attrib_location : require\n%s",
             s + sizeof v330 - 1);
    return b;
}

static GLuint prog(const char *vs, const char *fs, const char **xfb, int nxfb)
{
    GLuint p = glCreateProgram();
    const char *src[2] = {glsl(vs), glsl(fs)};
    GLenum kind[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    for (int i = 0; i < 2; i++) {
        GLuint s = glCreateShader(kind[i]);
        glShaderSource(s, 1, &src[i], NULL);
        glCompileShader(s);
        GLint ok;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetShaderInfoLog(s, sizeof log, NULL, log);
            fprintf(stderr, "compile: %s\n%s\n", log, src[i]);
            exit(1);
        }
        glAttachShader(p, s);
    }
    if (nxfb)
        glTransformFeedbackVaryings(p, nxfb, xfb, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        fprintf(stderr, "link: %s\n", log);
        exit(1);
    }
    return p;
}

/* Tile (tx, ty) as the viewport and scissor. */
static void tile(int tx, int ty)
{
    glViewport(tx * T, ty * T, T, T);
    glScissor(tx * T, ty * T, T, T);
}

static GLuint vao, vbo;

/* Draw `n` vec2 positions as `mode`, attribute 0. */
static void draw2(GLenum mode, const float *v, int n)
{
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, n * 2 * sizeof(float), v, GL_STREAM_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(mode, 0, n);
}

static const float quad[] = {-1, -1, 1, -1, -1, 1, 1, 1};

static const char *pos_vs =
    "#version 330 core\n"
    "layout(location = 0) in vec2 p;\n"
    "out vec2 uv;\n"
    "void main() { uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }\n";

int main(int argc, char **argv)
{
    EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    if (!eglInitialize(dpy, NULL, NULL)) {
        fprintf(stderr, "eglInitialize failed\n");
        return 1;
    }
    eglBindAPI(EGL_OPENGL_API);
    const EGLint cattr[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, cattr);
    if (!ctx && getenv("GL32")) {
        /* A host without dual-source blending: GL 3.2 (the dual-source
         * tile is skipped). */
        EGLint c32[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 2,
                        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
        ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, c32);
    }
    gl32 = getenv("GL32") != NULL;
    if (!ctx || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
        fprintf(stderr, "no GL 3.3 core context: 0x%x\n", eglGetError());
        return 1;
    }
    printf("GL_RENDERER: %s\nGL_VERSION: %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    /* The target: RGBA8 colour, depth. */
    GLuint fbo, color, depth;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &color);
    glBindTexture(GL_TEXTURE_2D, color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    glGenRenderbuffers(1, &depth);
    glBindRenderbuffer(GL_RENDERBUFFER, depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, W, H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "framebuffer incomplete\n");
        return 1;
    }
    glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_SCISSOR_TEST);

    /* (0,0) texture buffer: a gradient of 64 RGBA8 texels, size from
     * textureSize. */
    {
        unsigned char data[64 * 4];
        for (int i = 0; i < 64; i++) {
            data[i * 4 + 0] = i * 4;
            data[i * 4 + 1] = 255 - i * 4;
            data[i * 4 + 2] = (i & 8) ? 200 : 40;
            data[i * 4 + 3] = 255;
        }
        GLuint buf, tex;
        glGenBuffers(1, &buf);
        glBindBuffer(GL_TEXTURE_BUFFER, buf);
        glBufferData(GL_TEXTURE_BUFFER, sizeof data, data, GL_STATIC_DRAW);
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_BUFFER, tex);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA8, buf);
        GLuint p = prog(pos_vs,
                        "#version 330 core\n"
                        "uniform samplerBuffer tb;\n"
                        "in vec2 uv; out vec4 o;\n"
                        "void main() {\n"
                        "  int n = textureSize(tb);\n"
                        "  o = texelFetch(tb, int(uv.x * float(n)));\n"
                        "  if (n != 64) o = vec4(1.0, 0.0, 1.0, 1.0);\n"
                        "}\n",
                        NULL, 0);
        glUseProgram(p);
        tile(0, 0);
        draw2(GL_TRIANGLE_STRIP, quad, 4);
    }

    /* (1,0) texture swizzles: an R8 texture read as luminance (RRR1) on
     * the left, an RGBA one read as BGRA-with-alpha-one on the right. */
    {
        unsigned char lum[4 * 4], rgba[4 * 4 * 4];
        for (int i = 0; i < 16; i++) {
            lum[i] = i * 16;
            rgba[i * 4 + 0] = 255;
            rgba[i * 4 + 1] = i * 16;
            rgba[i * 4 + 2] = 30;
            rgba[i * 4 + 3] = 0;
        }
        GLuint t[2];
        glGenTextures(2, t);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, t[0]);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 4, 4, 0, GL_RED, GL_UNSIGNED_BYTE, lum);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        GLint sw[4] = {GL_RED, GL_RED, GL_RED, GL_ONE};
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, sw);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, t[1]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        GLint sw2[4] = {GL_BLUE, GL_GREEN, GL_RED, GL_ONE};
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, sw2);
        GLuint p = prog(pos_vs,
                        "#version 330 core\n"
                        "uniform sampler2D a, b;\n"
                        "in vec2 uv; out vec4 o;\n"
                        "void main() {\n"
                        "  vec2 c = vec2(fract(uv.x * 2.0), uv.y);\n"
                        "  o = uv.x < 0.5 ? texture(a, c) : texture(b, c);\n"
                        "}\n",
                        NULL, 0);
        glUseProgram(p);
        glUniform1i(glGetUniformLocation(p, "a"), 0);
        glUniform1i(glGetUniformLocation(p, "b"), 1);
        tile(1, 0);
        draw2(GL_TRIANGLE_STRIP, quad, 4);
        glActiveTexture(GL_TEXTURE0);
    }

    /* (2,0) vertex formats: BGRA colours (size GL_BGRA) and packed
     * 2_10_10_10 positions. */
    {
        struct { GLuint pos; unsigned char bgra[4]; } v[4];
        float ps[4][2] = {{-0.9f, -0.9f}, {0.9f, -0.9f}, {-0.9f, 0.9f}, {0.9f, 0.9f}};
        unsigned char cols[4][4] = {{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}, {255, 255, 0, 255}};
        for (int i = 0; i < 4; i++) {
            /* signed 10-bit normalized x, y; z 0; w 1 (2-bit 1). */
            int x = (int)lrintf(ps[i][0] * 511.0f) & 0x3ff, y = (int)lrintf(ps[i][1] * 511.0f) & 0x3ff;
            v[i].pos = (GLuint)x | ((GLuint)y << 10) | (1u << 30);
            memcpy(v[i].bgra, cols[i], 4);
        }
        GLuint p = prog("#version 330 core\n"
                        "layout(location = 0) in vec4 p;\n"
                        "layout(location = 1) in vec4 c;\n"
                        "out vec4 col;\n"
                        "void main() { col = c; gl_Position = p; }\n",
                        "#version 330 core\n"
                        "in vec4 col; out vec4 o;\n"
                        "void main() { o = col; }\n",
                        NULL, 0);
        glUseProgram(p);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_INT_2_10_10_10_REV, GL_TRUE, sizeof v[0], 0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, GL_BGRA, GL_UNSIGNED_BYTE, GL_TRUE, sizeof v[0], (void *)4);
        tile(2, 0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glDisableVertexAttribArray(1);
    }

    /* (3,0) dual-source blending: dst * src1 + src0. */
    if (!getenv("GL32")) {
        GLuint p = prog(pos_vs,
                        "#version 330 core\n"
                        "in vec2 uv;\n"
                        "layout(location = 0, index = 0) out vec4 c0;\n"
                        "layout(location = 0, index = 1) out vec4 c1;\n"
                        "void main() { c0 = vec4(uv.x * 0.5, 0.0, uv.y * 0.5, 1.0);"
                        " c1 = vec4(1.0, uv.y, 0.25, 1.0); }\n",
                        NULL, 0);
        tile(3, 0);
        glClearColor(0.5f, 0.8f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(p);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_SRC1_COLOR);
        draw2(GL_TRIANGLE_STRIP, quad, 4);
        glDisable(GL_BLEND);
        glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
    }

    /* (0,1) clip distances: a quad cut by two planes. */
    {
        GLuint p = prog("#version 330 core\n"
                        "layout(location = 0) in vec2 p;\n"
                        "out float gl_ClipDistance[2];\n"
                        "void main() {\n"
                        "  gl_ClipDistance[0] = p.x + p.y * 0.5;\n"
                        "  gl_ClipDistance[1] = 0.6 - p.y;\n"
                        "  gl_Position = vec4(p, 0.0, 1.0);\n"
                        "}\n",
                        "#version 330 core\nout vec4 o; void main() { o = vec4(0.9, 0.5, 0.1, 1.0); }\n",
                        NULL, 0);
        glUseProgram(p);
        glEnable(GL_CLIP_DISTANCE0);
        glEnable(GL_CLIP_DISTANCE1);
        tile(0, 1);
        draw2(GL_TRIANGLE_STRIP, quad, 4);
        glDisable(GL_CLIP_DISTANCE0);
        glDisable(GL_CLIP_DISTANCE1);
    }

    /* (1,1) depth clamp: a quad entirely beyond the far plane, kept and
     * depth-tested at the far plane against a nearer half quad. */
    {
        GLuint p = prog("#version 330 core\n"
                        "layout(location = 0) in vec2 p;\n"
                        "uniform float z;\n"
                        "void main() { gl_Position = vec4(p, z, 1.0); }\n",
                        "#version 330 core\nuniform vec4 c; out vec4 o; void main() { o = c; }\n",
                        NULL, 0);
        glUseProgram(p);
        tile(1, 1);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_DEPTH_CLAMP);
        const float half[] = {-1, -1, 0, -1, -1, 1, 0, 1};
        glUniform1f(glGetUniformLocation(p, "z"), 0.5f);
        glUniform4f(glGetUniformLocation(p, "c"), 0.2f, 0.9f, 0.3f, 1.0f);
        draw2(GL_TRIANGLE_STRIP, half, 4);
        glUniform1f(glGetUniformLocation(p, "z"), 3.0f);
        glUniform4f(glGetUniformLocation(p, "c"), 0.9f, 0.2f, 0.6f, 1.0f);
        draw2(GL_TRIANGLE_STRIP, quad, 4);
        glDisable(GL_DEPTH_CLAMP);
        glDisable(GL_DEPTH_TEST);
    }

    /* (2,1) instancing with a divisor, and points sized by glPointSize. */
    {
        GLuint p = prog("#version 330 core\n"
                        "layout(location = 0) in vec2 p;\n"
                        "layout(location = 1) in vec2 off;\n"
                        "flat out int id;\n"
                        "void main() { id = gl_InstanceID; gl_Position = vec4(p * 0.2 + off, 0.0, 1.0); }\n",
                        "#version 330 core\nflat in int id; out vec4 o;\n"
                        "void main() { o = vec4(float(id) / 3.0, 1.0 - float(id) / 3.0, 0.5, 1.0); }\n",
                        NULL, 0);
        glUseProgram(p);
        GLuint ob;
        glGenBuffers(1, &ob);
        const float offs[] = {-0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, ob);
        glBufferData(GL_ARRAY_BUFFER, sizeof offs, offs, GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glVertexAttribDivisor(1, 1);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        tile(2, 1);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, 4);
        glVertexAttribDivisor(1, 0);
        glDisableVertexAttribArray(1);
        GLuint pp = prog("#version 330 core\nlayout(location = 0) in vec2 p;\n"
                         "void main() { gl_Position = vec4(p, 0.0, 1.0); }\n",
                         "#version 330 core\nout vec4 o; void main() { o = vec4(1.0); }\n",
                         NULL, 0);
        glUseProgram(pp);
        glPointSize(6.0f);
        const float pts[] = {0.0f, 0.0f, -0.75f, 0.0f, 0.75f, 0.0f};
        draw2(GL_POINTS, pts, 3);
        glPointSize(1.0f);
    }

    /* (3,1) two render targets, blending only the second, writing only
     * green to the first. */
    {
        GLuint f2, t2[2];
        glGenFramebuffers(1, &f2);
        glBindFramebuffer(GL_FRAMEBUFFER, f2);
        glGenTextures(2, t2);
        for (int i = 0; i < 2; i++) {
            glBindTexture(GL_TEXTURE_2D, t2[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, T, T, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, t2[i], 0);
        }
        GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        glDrawBuffers(2, bufs);
        glDisable(GL_SCISSOR_TEST);
        glViewport(0, 0, T, T);
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        GLuint p = prog(pos_vs,
                        "#version 330 core\nin vec2 uv;\n"
                        "layout(location = 0) out vec4 a;\nlayout(location = 1) out vec4 b;\n"
                        "void main() { a = vec4(uv, 1.0, 1.0); b = vec4(1.0, uv.x, 0.0, uv.y); }\n",
                        NULL, 0);
        glUseProgram(p);
        glEnablei(GL_BLEND, 1);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColorMaski(0, GL_FALSE, GL_TRUE, GL_FALSE, GL_FALSE);
        draw2(GL_TRIANGLE_STRIP, quad, 4);
        glDisablei(GL_BLEND, 1);
        glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        /* Left half from target 0, right half from target 1. */
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0, 0, T / 2, T, 3 * T, T, 3 * T + T / 2, 2 * T, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        glBlitFramebuffer(T / 2, 0, T, T, 3 * T + T / 2, T, 4 * T, 2 * T, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glEnable(GL_SCISSOR_TEST);
        glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
    }

    /* (0,2) multisampling: a triangle in a 4x renderbuffer, resolved into
     * the tile with a blit. Edges are averaged differently by different
     * hardware, so the triangle is axis-aligned-ish but compared loosely. */
    {
        GLuint f2, rb;
        glGenFramebuffers(1, &f2);
        glBindFramebuffer(GL_FRAMEBUFFER, f2);
        glGenRenderbuffers(1, &rb);
        glBindRenderbuffer(GL_RENDERBUFFER, rb);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_RGBA8, T, T);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
        GLint samples = 0;
        glGetIntegerv(GL_SAMPLES, &samples);
        printf("msaa samples: %d\n", samples);
        glDisable(GL_SCISSOR_TEST);
        glViewport(0, 0, T, T);
        glClearColor(0.1f, 0.1f, 0.4f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        GLuint p = prog(pos_vs, "#version 330 core\nin vec2 uv; out vec4 o; void main() { o = vec4(1.0, uv, 1.0); }\n",
                        NULL, 0);
        glUseProgram(p);
        const float rect[] = {-0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
        draw2(GL_TRIANGLE_STRIP, rect, 4);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, f2);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
        glBlitFramebuffer(0, 0, T, T, 0, 2 * T, T, 3 * T, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glEnable(GL_SCISSOR_TEST);
        glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
    }

    /* (1,2) a multisampled texture, filled with a full-tile draw (every
     * sample alike) and fetched with texelFetch. */
    {
        GLuint f2, mt;
        glGenFramebuffers(1, &f2);
        glBindFramebuffer(GL_FRAMEBUFFER, f2);
        glGenTextures(1, &mt);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, mt);
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA8, T, T, GL_TRUE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, mt, 0);
        glDisable(GL_SCISSOR_TEST);
        glViewport(0, 0, T, T);
        GLuint p = prog(pos_vs, "#version 330 core\nin vec2 uv; out vec4 o; void main() { o = vec4(uv.y, 0.3, uv.x, 1.0); }\n",
                        NULL, 0);
        glUseProgram(p);
        draw2(GL_TRIANGLE_STRIP, quad, 4);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glEnable(GL_SCISSOR_TEST);
        GLuint q = prog(pos_vs,
                        "#version 330 core\nuniform sampler2DMS ms;\nin vec2 uv; out vec4 o;\n"
                        "void main() { o = texelFetch(ms, ivec2(gl_FragCoord.xy) - ivec2(64, 128), 2).bgra; }\n",
                        NULL, 0);
        glUseProgram(q);
        tile(1, 2);
        draw2(GL_TRIANGLE_STRIP, quad, 4);
    }

    /* (2,2) transform feedback: a vertex shader's outputs captured, then
     * drawn from; and the primitives-written query. */
    {
        const char *names[] = {"outp", "outc"};
        GLuint p = prog("#version 330 core\n"
                        "layout(location = 0) in vec2 p;\n"
                        "out vec2 outp; out vec3 outc;\n"
                        "void main() { outp = p.yx * vec2(0.8, -0.8); outc = vec3(p * 0.5 + 0.5, 0.2);"
                        " gl_Position = vec4(p, 0.0, 1.0); }\n",
                        "#version 330 core\nout vec4 o; void main() { o = vec4(1.0); }\n",
                        names, 2);
        GLuint xb, q;
        glGenBuffers(1, &xb);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xb);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 6 * 5 * sizeof(float), NULL, GL_STATIC_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xb);
        glGenQueries(1, &q);
        glUseProgram(p);
        glEnable(GL_RASTERIZER_DISCARD);
        glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
        glBeginTransformFeedback(GL_TRIANGLES);
        /* A strip: captured as two separate triangles. */
        const float strip[] = {-0.9f, -0.9f, 0.9f, -0.9f, -0.9f, 0.9f, 0.9f, 0.9f};
        draw2(GL_TRIANGLE_STRIP, strip, 4);
        glEndTransformFeedback();
        glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
        glDisable(GL_RASTERIZER_DISCARD);
        GLuint written = 0;
        glGetQueryObjectuiv(q, GL_QUERY_RESULT, &written);
        float got[30];
        glBindBuffer(GL_ARRAY_BUFFER, xb);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof got, got);
        printf("xfb primitives written: %u\nxfb:", written);
        for (int i = 0; i < 30; i++)
            printf(" %.2f", got[i]);
        printf("\n");
        GLuint d = prog("#version 330 core\n"
                        "layout(location = 0) in vec2 p;\nlayout(location = 1) in vec3 c;\n"
                        "out vec3 col; void main() { col = c; gl_Position = vec4(p, 0.0, 1.0); }\n",
                        "#version 330 core\nin vec3 col; out vec4 o; void main() { o = vec4(col, 1.0); }\n",
                        NULL, 0);
        glUseProgram(d);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(2 * sizeof(float)));
        tile(2, 2);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisableVertexAttribArray(1);
    }

    /* (3,2) conditional rendering on an occlusion query (which passes),
     * and a timer query around it. */
    {
        GLuint q[2];
        glGenQueries(2, q);
        GLuint p = prog(pos_vs, "#version 330 core\nin vec2 uv; out vec4 o; void main() { o = vec4(uv.x, 0.6, uv.y, 1.0); }\n",
                        NULL, 0);
        glUseProgram(p);
        tile(3, 2);
        glBeginQuery(GL_TIME_ELAPSED, q[1]);
        glBeginQuery(GL_SAMPLES_PASSED, q[0]);
        const float small[] = {-0.2f, -0.2f, 0.2f, -0.2f, -0.2f, 0.2f};
        draw2(GL_TRIANGLES, small, 3);
        glEndQuery(GL_SAMPLES_PASSED);
        glBeginConditionalRender(q[0], GL_QUERY_WAIT);
        draw2(GL_TRIANGLE_STRIP, quad, 4);
        glEndConditionalRender();
        glEndQuery(GL_TIME_ELAPSED);
        GLuint64 ns = 0;
        GLuint passed = 0;
        glGetQueryObjectuiv(q[0], GL_QUERY_RESULT, &passed);
        PFNGLGETQUERYOBJECTUI64VPROC get64 =
            (PFNGLGETQUERYOBJECTUI64VPROC)eglGetProcAddress("glGetQueryObjectui64v");
        get64(q[1], GL_QUERY_RESULT, &ns);
        printf("samples passed: %s, time elapsed: %s\n", passed ? "some" : "none", ns ? "nonzero" : "zero");
    }

    GLenum err = glGetError();
    if (err)
        printf("GL error 0x%x\n", err);
    static unsigned char px[W * H * 4];
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
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
    unsigned sum = 0;
    for (int i = 0; i < W * H * 4; i++)
        sum = sum * 31 + px[i];
    printf("checksum %08x\n", sum);
    return 0;
}
