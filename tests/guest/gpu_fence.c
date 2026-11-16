/*
 * gpu_fence: native fences, the way Chromium's GPU process uses them to
 * hand frames to a Wayland compositor. Not part of the conformance set (it
 * needs Mesa): an end-to-end check for nixvm's virtio-gpu device
 * (`--features gpu`, `NIXVM_GPU=...`).
 *
 * Build in an Alpine root with mesa-dev:
 *     gcc -O2 -o gpu_fence gpu_fence.c -lEGL -lGLESv2 -lgbm
 * Run:   NIXVM_GPU=egl nixvm run --root <root> -- /gpu_fence
 *
 * Why it exists: Mesa's virgl batches commands in the guest and sends them to
 * the device only on a flush. Chromium never flushes a frame for its own
 * sake: it asks for an EGL_ANDROID_native_fence_sync fence (which flushes),
 * and without that extension it commits buffers whose commands are still in
 * the guest — the compositor got transparent black wherever a frame wasn't
 * redrawn soon after (the browser's toolbar, any static page). Mesa offers the
 * extension only when execbuffers return fence fds (virtio_gpu 0.1).
 *
 * So: two contexts with separate command streams (as the GPU process and a
 * compositor are). A renders into a GBM buffer through a dma-buf EGLImage
 * and, with no glFlush, takes a native fence; once the fence fd polls
 * readable (and SYNC_IOC_FILE_INFO says signalled), B imports the same
 * dma-buf and must read the finished frame, not the old contents. B then
 * waits on the fence as an in-fence (eglWaitSyncKHR: Mesa merges in-fences
 * with SYNC_IOC_MERGE and passes them with the next execbuffer).
 */
#define EGL_EGLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
/* After gl3.h, which defines the types it uses. */
#include <GLES2/gl2ext.h>
#include <fcntl.h>
#include <gbm.h>
#include <linux/sync_file.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define W 64
#define H 64

static int failures;

static void check(int ok, const char *what)
{
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok)
        failures++;
}

static int has_ext(const char *list, const char *ext)
{
    size_t n = strlen(ext);
    for (const char *p = list; (p = strstr(p, ext)); p += n)
        if ((p == list || p[-1] == ' ') && (p[n] == ' ' || p[n] == 0))
            return 1;
    return 0;
}

static PFNEGLCREATESYNCKHRPROC create_sync;
static PFNEGLDESTROYSYNCKHRPROC destroy_sync;
static PFNEGLCLIENTWAITSYNCKHRPROC client_wait;
static PFNEGLWAITSYNCKHRPROC server_wait;
static PFNEGLDUPNATIVEFENCEFDANDROIDPROC dup_fence;
static PFNEGLCREATEIMAGEKHRPROC create_image;
static PFNEGLDESTROYIMAGEKHRPROC destroy_image;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target;

/* The dma-buf `fd` as an EGLImage, bound to a new texture and FBO. */
static GLuint import_fbo(EGLDisplay dpy, int fd, uint32_t stride, EGLImageKHR *img)
{
    const EGLint attrs[] = {
        EGL_WIDTH, W, EGL_HEIGHT, H,
        EGL_LINUX_DRM_FOURCC_EXT, GBM_FORMAT_ABGR8888,
        EGL_DMA_BUF_PLANE0_FD_EXT, fd,
        EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
        EGL_DMA_BUF_PLANE0_PITCH_EXT, (EGLint)stride,
        EGL_NONE,
    };
    *img = create_image(dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
    if (*img == EGL_NO_IMAGE_KHR) {
        fprintf(stderr, "eglCreateImageKHR(dma-buf): 0x%x\n", eglGetError());
        exit(1);
    }
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    image_target(GL_TEXTURE_2D, *img);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "dma-buf framebuffer incomplete\n");
        exit(1);
    }
    glViewport(0, 0, W, H);
    return fbo;
}

static int pixel_is(const uint8_t *px, int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    const uint8_t *p = px + (y * W + x) * 4;
    if (p[0] == r && p[1] == g && p[2] == b && p[3] == a)
        return 1;
    fprintf(stderr, "  (%d,%d) = %d,%d,%d,%d, want %d,%d,%d,%d\n", x, y, p[0], p[1], p[2],
            p[3], r, g, b, a);
    return 0;
}

int main(void)
{
    int node = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (node < 0) {
        perror("/dev/dri/renderD128");
        return 1;
    }
    struct gbm_device *gbm = gbm_create_device(node);
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay dpy = get_display(EGL_PLATFORM_GBM_KHR, gbm, NULL);
    if (!gbm || dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, NULL, NULL)) {
        fprintf(stderr, "no GBM EGL display: 0x%x\n", eglGetError());
        return 1;
    }
    const char *exts = eglQueryString(dpy, EGL_EXTENSIONS);
    printf("EGL vendor: %s\n", eglQueryString(dpy, EGL_VENDOR));
    check(has_ext(exts, "EGL_ANDROID_native_fence_sync"),
          "EGL_ANDROID_native_fence_sync (execbuffers return fence fds)");
    if (!has_ext(exts, "EGL_ANDROID_native_fence_sync") ||
        !has_ext(exts, "EGL_EXT_image_dma_buf_import") ||
        !has_ext(exts, "EGL_KHR_wait_sync")) {
        printf("FAIL: missing EGL extensions: %s\n", exts);
        return 1;
    }
    create_sync = (void *)eglGetProcAddress("eglCreateSyncKHR");
    destroy_sync = (void *)eglGetProcAddress("eglDestroySyncKHR");
    client_wait = (void *)eglGetProcAddress("eglClientWaitSyncKHR");
    server_wait = (void *)eglGetProcAddress("eglWaitSyncKHR");
    dup_fence = (void *)eglGetProcAddress("eglDupNativeFenceFDANDROID");
    create_image = (void *)eglGetProcAddress("eglCreateImageKHR");
    destroy_image = (void *)eglGetProcAddress("eglDestroyImageKHR");
    image_target = (void *)eglGetProcAddress("glEGLImageTargetTexture2DOES");

    eglBindAPI(EGL_OPENGL_ES_API);
    const EGLint cattr[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    EGLContext a = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, cattr);
    EGLContext b = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, cattr);
    if (a == EGL_NO_CONTEXT || b == EGL_NO_CONTEXT) {
        fprintf(stderr, "eglCreateContext: 0x%x\n", eglGetError());
        return 1;
    }

    /* The buffer a compositor would be handed. */
    struct gbm_bo *bo = gbm_bo_create(gbm, W, H, GBM_FORMAT_ABGR8888,
                                      GBM_BO_USE_RENDERING | GBM_BO_USE_SCANOUT);
    if (!bo) {
        perror("gbm_bo_create");
        return 1;
    }
    int buf = gbm_bo_get_fd(bo);
    uint32_t stride = gbm_bo_get_stride(bo);

    /* A: the buffer starts red, finished (glFinish). */
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, a);
    EGLImageKHR img_a;
    import_fbo(dpy, buf, stride, &img_a);
    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();

    /* A: the frame — green, the right half blue (scissored) — then a
     * native fence and no flush of our own. */
    glClearColor(0, 1, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_SCISSOR_TEST);
    glScissor(W / 2, 0, W / 2, H);
    glClearColor(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);
    const EGLint fattr[] = {EGL_SYNC_NATIVE_FENCE_FD_ANDROID, EGL_NO_NATIVE_FENCE_FD_ANDROID,
                            EGL_NONE};
    EGLSyncKHR sync = create_sync(dpy, EGL_SYNC_NATIVE_FENCE_ANDROID, fattr);
    check(sync != EGL_NO_SYNC_KHR, "eglCreateSyncKHR(NATIVE_FENCE)");
    int fence = dup_fence(dpy, sync);
    check(fence >= 0, "eglDupNativeFenceFDANDROID gives a sync file");
    if (fence < 0)
        return 1;
    struct pollfd p = {.fd = fence, .events = POLLIN};
    check(poll(&p, 1, 5000) == 1 && (p.revents & POLLIN), "the fence fd polls readable");
    struct sync_file_info info = {0};
    check(ioctl(fence, SYNC_IOC_FILE_INFO, &info) == 0 && info.status == 1 &&
              info.num_fences >= 1,
          "SYNC_IOC_FILE_INFO: signalled");
    check(client_wait(dpy, sync, 0, 0) == EGL_CONDITION_SATISFIED_KHR,
          "eglClientWaitSyncKHR: satisfied");
    destroy_sync(dpy, sync);

    /* B: another command stream, as a compositor's. It sees the frame. */
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, b);
    EGLImageKHR img_b;
    import_fbo(dpy, buf, stride, &img_b);
    /* The fence as an in-fence, twice (the second merges with the first). */
    for (int i = 0; i < 2; i++) {
        const EGLint iattr[] = {EGL_SYNC_NATIVE_FENCE_FD_ANDROID, dup(fence), EGL_NONE};
        EGLSyncKHR in = create_sync(dpy, EGL_SYNC_NATIVE_FENCE_ANDROID, iattr);
        check(in != EGL_NO_SYNC_KHR && server_wait(dpy, in, 0) == EGL_TRUE,
              "eglWaitSyncKHR on the imported fence");
        destroy_sync(dpy, in);
    }
    static uint8_t px[W * H * 4];
    glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
    check(pixel_is(px, 5, 5, 0, 255, 0, 255) && pixel_is(px, W / 2 - 1, H / 2, 0, 255, 0, 255) &&
              pixel_is(px, W / 2, H / 2, 0, 0, 255, 255) && pixel_is(px, W - 1, H - 1, 0, 0, 255, 255),
          "the other context reads the fenced frame (not the old red)");
    close(fence);

    destroy_image(dpy, img_b);
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    destroy_image(dpy, img_a);
    eglDestroyContext(dpy, a);
    eglDestroyContext(dpy, b);
    eglTerminate(dpy);
    close(buf);
    gbm_bo_destroy(bo);
    gbm_device_destroy(gbm);
    close(node);
    if (failures) {
        printf("FAIL (%d)\n", failures);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
