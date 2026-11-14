// A Wayland client speaking the wire protocol by hand (no libwayland): it
// binds the globals it needs, maps an xdg_toplevel filled with one color
// from a memfd-backed wl_shm buffer passed over SCM_RIGHTS, and waits for a
// key press, printing what it got. tests/display.rs runs it on nixvm's
// built-in display, checks the frame's pixels, then presses a key.
//
//   wayland_client [RRGGBB]      (default c03060)
//
// Prints CONFIGURED <w> <h>, COMMITTED, then KEY <code> and PASS.
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static int sock;
static uint8_t inbuf[65536];
static size_t inlen;
static int infds[16];
static int ninfds;

enum {
    ID_DISPLAY = 1,
    ID_REGISTRY,
    ID_SYNC,
    ID_COMPOSITOR,
    ID_SHM,
    ID_WM_BASE,
    ID_SEAT,
    ID_KEYBOARD,
    ID_SURFACE,
    ID_XDG_SURFACE,
    ID_TOPLEVEL,
    ID_POOL,
    ID_BUFFER,
    ID_FRAME,
};

static void die(const char *what) {
    printf("FAIL %s (errno %d)\n", what, errno);
    fflush(stdout);
    exit(1);
}

// A request being built.
static uint32_t msg[256];
static int msglen;

static void begin(uint32_t obj, uint32_t op) {
    msg[0] = obj;
    msg[1] = op;
    msglen = 2;
}
static void put(uint32_t v) { msg[msglen++] = v; }
static void put_str(const char *s) {
    uint32_t n = strlen(s) + 1;
    put(n);
    memset(&msg[msglen], 0, (n + 3) & ~3u);
    memcpy(&msg[msglen], s, n);
    msglen += (n + 3) / 4;
}
static void send_msg(int fd) {
    msg[1] |= (uint32_t)(msglen * 4) << 16;
    struct iovec iov = {msg, msglen * 4};
    char cbuf[CMSG_SPACE(sizeof(int))];
    struct msghdr mh = {0};
    mh.msg_iov = &iov;
    mh.msg_iovlen = 1;
    if (fd >= 0) {
        mh.msg_control = cbuf;
        mh.msg_controllen = sizeof cbuf;
        struct cmsghdr *c = CMSG_FIRSTHDR(&mh);
        c->cmsg_level = SOL_SOCKET;
        c->cmsg_type = SCM_RIGHTS;
        c->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(c), &fd, sizeof fd);
    }
    if (sendmsg(sock, &mh, 0) != msglen * 4)
        die("sendmsg");
}

// Read more from the socket (blocking), collecting any fds.
static void fill(void) {
    struct iovec iov = {inbuf + inlen, sizeof inbuf - inlen};
    char cbuf[CMSG_SPACE(sizeof(int) * 8)];
    struct msghdr mh = {0};
    mh.msg_iov = &iov;
    mh.msg_iovlen = 1;
    mh.msg_control = cbuf;
    mh.msg_controllen = sizeof cbuf;
    ssize_t n = recvmsg(sock, &mh, 0);
    if (n <= 0)
        die("recvmsg (the compositor hung up)");
    inlen += n;
    for (struct cmsghdr *c = CMSG_FIRSTHDR(&mh); c; c = CMSG_NXTHDR(&mh, c)) {
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
            int k = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
            memcpy(&infds[ninfds], CMSG_DATA(c), k * sizeof(int));
            ninfds += k;
        }
    }
}

// The next event: fills obj, op and a pointer to its arguments.
static uint32_t ev[16384];
static uint32_t ev_obj, ev_op, ev_words;
static void next_event(void) {
    while (inlen < 8 || inlen < (size_t)(((uint32_t *)inbuf)[1] >> 16))
        fill();
    uint32_t *h = (uint32_t *)inbuf;
    uint32_t size = h[1] >> 16;
    ev_obj = h[0];
    ev_op = h[1] & 0xffff;
    ev_words = (size - 8) / 4;
    memcpy(ev, inbuf + 8, size - 8);
    memmove(inbuf, inbuf + size, inlen - size);
    inlen -= size;
}

static int take_fd(void) {
    if (!ninfds)
        die("event without its fd");
    int fd = infds[0];
    memmove(infds, infds + 1, --ninfds * sizeof(int));
    return fd;
}

static void handle_common(void) {
    if (ev_obj == ID_DISPLAY && ev_op == 0) {
        printf("FAIL protocol error on %u code %u: %s\n", ev[0], ev[1], (char *)&ev[3]);
        exit(1);
    }
    if (ev_obj == ID_WM_BASE && ev_op == 0) { // ping
        begin(ID_WM_BASE, 3);
        put(ev[0]);
        send_msg(-1);
    }
    if (ev_obj == ID_KEYBOARD && ev_op == 0) { // keymap: format, fd, size
        int fd = take_fd();
        char *km = mmap(NULL, ev[1], PROT_READ, MAP_PRIVATE, fd, 0);
        if (km == MAP_FAILED)
            die("mmap keymap");
        if (strncmp(km, "xkb_keymap", 10) != 0)
            die("keymap text");
        printf("KEYMAP %u bytes\n", ev[1]);
        munmap(km, ev[1]);
        close(fd);
    }
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    uint32_t rgb = argc > 1 ? strtoul(argv[1], NULL, 16) : 0xc03060;
    const char *path = getenv("WAYLAND_DISPLAY");
    if (!path)
        die("no WAYLAND_DISPLAY");
    sock = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    struct sockaddr_un sa = {.sun_family = AF_UNIX};
    strncpy(sa.sun_path, path, sizeof sa.sun_path - 1);
    for (int tries = 0; connect(sock, (struct sockaddr *)&sa, sizeof sa) < 0; tries++) {
        if (tries > 100)
            die("connect");
        usleep(10000);
    }

    begin(ID_DISPLAY, 1); // get_registry
    put(ID_REGISTRY);
    send_msg(-1);
    begin(ID_DISPLAY, 0); // sync
    put(ID_SYNC);
    send_msg(-1);
    uint32_t g_comp = 0, g_shm = 0, g_wm = 0, g_seat = 0;
    for (;;) {
        next_event();
        if (ev_obj == ID_REGISTRY && ev_op == 0) {
            const char *name = (const char *)&ev[2];
            if (!strcmp(name, "wl_compositor"))
                g_comp = ev[0];
            if (!strcmp(name, "wl_shm"))
                g_shm = ev[0];
            if (!strcmp(name, "xdg_wm_base"))
                g_wm = ev[0];
            if (!strcmp(name, "wl_seat"))
                g_seat = ev[0];
        } else if (ev_obj == ID_SYNC) {
            break;
        } else {
            handle_common();
        }
    }
    if (!g_comp || !g_shm || !g_wm || !g_seat)
        die("missing globals");

    struct {
        uint32_t name, id, version;
        const char *iface;
    } binds[] = {
        {g_comp, ID_COMPOSITOR, 4, "wl_compositor"},
        {g_shm, ID_SHM, 1, "wl_shm"},
        {g_wm, ID_WM_BASE, 1, "xdg_wm_base"},
        {g_seat, ID_SEAT, 4, "wl_seat"},
    };
    for (int i = 0; i < 4; i++) {
        begin(ID_REGISTRY, 0);
        put(binds[i].name);
        put_str(binds[i].iface);
        put(binds[i].version);
        put(binds[i].id);
        send_msg(-1);
    }
    begin(ID_SEAT, 1); // get_keyboard
    put(ID_KEYBOARD);
    send_msg(-1);
    begin(ID_COMPOSITOR, 0); // create_surface
    put(ID_SURFACE);
    send_msg(-1);
    begin(ID_WM_BASE, 2); // get_xdg_surface
    put(ID_XDG_SURFACE);
    put(ID_SURFACE);
    send_msg(-1);
    begin(ID_XDG_SURFACE, 1); // get_toplevel
    put(ID_TOPLEVEL);
    send_msg(-1);
    begin(ID_TOPLEVEL, 2); // set_title
    put_str("raw wire client");
    send_msg(-1);
    begin(ID_SURFACE, 6); // commit: no buffer, asks for a configure
    send_msg(-1);

    int32_t w = 0, h = 0;
    uint32_t serial;
    for (;;) {
        next_event();
        if (ev_obj == ID_TOPLEVEL && ev_op == 0) {
            w = ev[0];
            h = ev[1];
        } else if (ev_obj == ID_XDG_SURFACE && ev_op == 0) {
            serial = ev[0];
            break;
        } else {
            handle_common();
        }
    }
    if (w <= 0 || h <= 0) {
        w = 200;
        h = 100;
    }
    printf("CONFIGURED %d %d\n", w, h);
    begin(ID_XDG_SURFACE, 4); // ack_configure
    put(serial);
    send_msg(-1);

    size_t size = (size_t)w * h * 4;
    int fd = memfd_create("wl_shm", MFD_CLOEXEC);
    if (fd < 0 || ftruncate(fd, size) < 0)
        die("memfd");
    uint32_t *px = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (px == MAP_FAILED)
        die("mmap");
    for (size_t i = 0; i < (size_t)w * h; i++)
        px[i] = 0xff000000u | rgb;
    begin(ID_SHM, 0); // create_pool
    put(ID_POOL);
    put(size);
    send_msg(fd);
    close(fd); // the compositor holds its own reference
    begin(ID_POOL, 0); // create_buffer
    put(ID_BUFFER);
    put(0);
    put(w);
    put(h);
    put(w * 4);
    put(1); // XRGB8888
    send_msg(-1);
    begin(ID_SURFACE, 1); // attach
    put(ID_BUFFER);
    put(0);
    put(0);
    send_msg(-1);
    begin(ID_SURFACE, 2); // damage
    put(0);
    put(0);
    put(w);
    put(h);
    send_msg(-1);
    begin(ID_SURFACE, 3); // frame
    put(ID_FRAME);
    send_msg(-1);
    begin(ID_SURFACE, 6); // commit
    send_msg(-1);
    printf("COMMITTED\n");

    int framed = 0;
    for (;;) {
        next_event();
        if (ev_obj == ID_FRAME && ev_op == 0 && !framed) {
            framed = 1;
            printf("FRAME DONE\n");
        } else if (ev_obj == ID_KEYBOARD && ev_op == 1) {
            printf("KEYBOARD ENTER\n");
        } else if (ev_obj == ID_KEYBOARD && ev_op == 3 && ev[3] == 1) {
            printf("KEY %u\n", ev[2]);
            printf(framed ? "PASS\n" : "FAIL no frame callback\n");
            return framed ? 0 : 1;
        } else {
            handle_common();
        }
    }
}
