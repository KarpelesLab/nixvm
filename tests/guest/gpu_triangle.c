/*
 * gpu_triangle: render one triangle into an FBO through EGL (surfaceless) +
 * GLES 3.0, read the pixels back and check them. Not part of the conformance
 * set (it needs Mesa, not a static libc): it is the end-to-end check for
 * nixvm's virtio-gpu / virgl device (`--features gpu`, `NIXVM_GPU=...`).
 *
 * Build in an Alpine root with mesa-dev:   gcc -O2 -o gpu_triangle gpu_triangle.c -lEGL -lGLESv2
 * Run:   NIXVM_GPU=egl nixvm run --root <root> -- /gpu_triangle [out.ppm]
 *
 * The left half of the frame is cleared red; the triangle is drawn with a
 * per-vertex colour (green at the bottom corners, blue at the top) scaled by
 * a uniform, so the clear, a varying, a uniform and the readback all show.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 64
#define H 64

static const char *vs_src =
    "#version 300 es\n"
    "layout(location = 0) in vec2 pos;\n"
    "layout(location = 1) in vec3 col;\n"
    "out vec3 v_col;\n"
    "void main() { v_col = col; gl_Position = vec4(pos, 0.0, 1.0); }\n";

static const char *fs_src =
    "#version 300 es\n"
    "precision mediump float;\n"
    "uniform float scale;\n"
    "in vec3 v_col;\n"
    "out vec4 o;\n"
    "void main() { o = vec4(v_col * scale, 1.0); }\n";

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

int main(int argc, char **argv)
{
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay dpy = get_display
        ? get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL)
        : eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major, minor;
    if (!eglInitialize(dpy, &major, &minor)) {
        fprintf(stderr, "eglInitialize failed: 0x%x\n", eglGetError());
        return 1;
    }
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint cfg_attrs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE};
    EGLConfig cfg = 0;
    EGLint n = 0;
    eglChooseConfig(dpy, cfg_attrs, &cfg, 1, &n);
    EGLint ctx_attrs[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, n ? cfg : EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attrs);
    if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
        fprintf(stderr, "no GLES 3 context: 0x%x\n", eglGetError());
        return 1;
    }
    printf("GL_VENDOR: %s\nGL_RENDERER: %s\nGL_VERSION: %s\n", glGetString(GL_VENDOR),
           glGetString(GL_RENDERER), glGetString(GL_VERSION));

    GLuint rb, fb;
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, W, H);
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "framebuffer incomplete\n");
        return 1;
    }

    GLuint prog = glCreateProgram();
    glAttachShader(prog, shader(GL_VERTEX_SHADER, vs_src));
    glAttachShader(prog, shader(GL_FRAGMENT_SHADER, fs_src));
    glLinkProgram(prog);
    glUseProgram(prog);
    glUniform1f(glGetUniformLocation(prog, "scale"), 1.0f);

    static const GLfloat verts[] = {
        /* x, y, r, g, b */
        -0.8f, -0.8f, 0.0f, 1.0f, 0.0f,
        0.8f, -0.8f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.8f, 0.0f, 0.0f, 1.0f,
    };
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat),
                          (void *)(2 * sizeof(GLfloat)));

    glViewport(0, 0, W, H);
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    static unsigned char px[W * H * 4];
    glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
    GLenum e = glGetError();
    if (e != GL_NO_ERROR)
        printf("glGetError: 0x%x\n", e);

    /* GL's row 0 is the bottom. */
#define AT(x, y) (&px[((y) * W + (x)) * 4])
    const unsigned char *corner = AT(1, H - 2);  /* top-left: outside → clear red */
    const unsigned char *bottom = AT(W / 2, 8);  /* near the base → mostly green */
    const unsigned char *top = AT(W / 2, H - 14); /* near the apex → more blue */
    printf("corner %u %u %u %u\n", corner[0], corner[1], corner[2], corner[3]);
    printf("bottom %u %u %u %u\n", bottom[0], bottom[1], bottom[2], bottom[3]);
    printf("top    %u %u %u %u\n", top[0], top[1], top[2], top[3]);

    if (argc > 1) {
        FILE *f = fopen(argv[1], "wb");
        if (f) {
            fprintf(f, "P6\n%d %d\n255\n", W, H);
            for (int y = H - 1; y >= 0; y--)
                for (int x = 0; x < W; x++)
                    fwrite(AT(x, y), 1, 3, f);
            fclose(f);
        }
    }

    int ok = corner[0] > 200 && corner[1] < 50 && corner[2] < 50 &&
             bottom[0] < 50 && bottom[1] > 150 && bottom[2] < 100 &&
             top[0] < 50 && top[2] > bottom[2] + 50;
    puts(ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
