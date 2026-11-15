/*
 * gpu_formats: clear a renderbuffer of each colour format a GLES 3.0 host
 * renders to, read a pixel back both ways GLES allows (the canonical
 * format/type and the implementation's preferred pair) and print it. Not a
 * pass/fail test: the output of one GPU backend is compared with another's
 * (NIXVM_GPU=egl natively vs. WebGL2 in a browser, web/scripts/gpu-check.mjs)
 * and with llvmpipe. It covers the readback path for formats WebGL2 can't
 * read directly (R8, RGB565, half floats, R11F_G11F_B10F, narrow integers).
 *
 * Build in an Alpine root with mesa-dev:  gcc -O2 -o gpu_formats gpu_formats.c -lEGL -lGLESv2
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdio.h>
#include <string.h>

enum kind { NORM, FLT, UINT, SINT };

static const struct {
    const char *name;
    GLenum internal;
    enum kind kind;
} formats[] = {
    {"R8", GL_R8, NORM},
    {"RG8", GL_RG8, NORM},
    {"RGB565", GL_RGB565, NORM},
    {"RGBA8", GL_RGBA8, NORM},
    {"RGB10_A2", GL_RGB10_A2, NORM},
    {"R16F", GL_R16F, FLT},
    {"RGBA16F", GL_RGBA16F, FLT},
    {"R32F", GL_R32F, FLT},
    {"R11F_G11F_B10F", GL_R11F_G11F_B10F, FLT},
    {"R8UI", GL_R8UI, UINT},
    {"RG16I", GL_RG16I, SINT},
    {"RGBA32UI", GL_RGBA32UI, UINT},
    {"RGB10_A2UI", GL_RGB10_A2UI, UINT},
};

static void dump(const char *what, const unsigned char *p, int n)
{
    printf("  %s:", what);
    for (int i = 0; i < n; i++)
        printf(" %02x", p[i]);
    printf("\n");
}

static int type_size(GLenum format, GLenum type)
{
    int ch = 4;
    switch (format) {
    case GL_RED: case GL_RED_INTEGER: ch = 1; break;
    case GL_RG: case GL_RG_INTEGER: ch = 2; break;
    case GL_RGB: case GL_RGB_INTEGER: ch = 3; break;
    }
    switch (type) {
    case GL_UNSIGNED_BYTE: case GL_BYTE: return ch;
    case GL_UNSIGNED_SHORT: case GL_SHORT: case GL_HALF_FLOAT: return 2 * ch;
    case GL_UNSIGNED_SHORT_5_6_5: case GL_UNSIGNED_SHORT_4_4_4_4: case GL_UNSIGNED_SHORT_5_5_5_1: return 2;
    case GL_UNSIGNED_INT_2_10_10_10_REV: case GL_UNSIGNED_INT_10F_11F_11F_REV: return 4;
    default: return 4 * ch;
    }
}

int main(void)
{
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay dpy = get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    if (!eglInitialize(dpy, NULL, NULL)) {
        fprintf(stderr, "eglInitialize failed\n");
        return 1;
    }
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint attrs[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attrs);
    if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
        fprintf(stderr, "no GLES 3 context\n");
        return 1;
    }
    printf("GL_RENDERER: %s\n", glGetString(GL_RENDERER));
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    for (unsigned i = 0; i < sizeof formats / sizeof *formats; i++) {
        GLuint fb, rb;
        glGenFramebuffers(1, &fb);
        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glGenRenderbuffers(1, &rb);
        glBindRenderbuffer(GL_RENDERBUFFER, rb);
        glRenderbufferStorage(GL_RENDERBUFFER, formats[i].internal, 4, 4);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
        printf("%s:", formats[i].name);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            printf(" incomplete\n");
            glGetError();
            continue;
        }
        printf("\n");
        static const GLfloat fv[4] = {0.25f, 0.5f, 1.0f, 0.75f};
        static const GLuint uv[4] = {3, 200, 1000, 1};
        static const GLint iv[4] = {-7, 300, 5, 1};
        GLenum cf = GL_RGBA, ct = GL_UNSIGNED_BYTE;
        switch (formats[i].kind) {
        case NORM: glClearBufferfv(GL_COLOR, 0, fv); break;
        case FLT: glClearBufferfv(GL_COLOR, 0, fv); ct = GL_FLOAT; break;
        case UINT: glClearBufferuiv(GL_COLOR, 0, uv); cf = GL_RGBA_INTEGER; ct = GL_UNSIGNED_INT; break;
        case SINT: glClearBufferiv(GL_COLOR, 0, iv); cf = GL_RGBA_INTEGER; ct = GL_INT; break;
        }
        unsigned char px[64];
        memset(px, 0xee, sizeof px);
        glReadPixels(1, 1, 1, 1, cf, ct, px);
        dump("canonical", px, type_size(cf, ct));
        GLint f = 0, t = 0;
        glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &f);
        glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &t);
        if ((GLenum)f != cf || (GLenum)t != ct) {
            memset(px, 0xee, sizeof px);
            glReadPixels(1, 1, 1, 1, f, t, px);
            char what[64];
            snprintf(what, sizeof what, "0x%x/0x%x", f, t);
            dump(what, px, type_size(f, t));
        }
        GLenum e = glGetError();
        if (e)
            printf("  glGetError 0x%x\n", e);
        glDeleteFramebuffers(1, &fb);
        glDeleteRenderbuffers(1, &rb);
    }
    return 0;
}
