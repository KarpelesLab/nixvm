// Protected paths in a writable host directory (`Passthrough::protect`).
//
// Not a Linux conformance program: it runs in a /work laid out and protected by
// tests/guest_conformance.rs (`.git/hooks` and `atelier.toml` with a hook and a
// config in them, and `bare/.git/hooks` in an empty `bare`), and checks that
// the guest can't change what's protected however it goes about it, starting
// with the rename-a-parent attack:
//
//     mv .git g && echo evil > g/hooks/pre-commit && mv g .git
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

static int failed;

#define REFUSED(call, want)                                                    \
    do {                                                                       \
        errno = 0;                                                             \
        long r_ = (long)(call);                                                \
        if (r_ != -1 || errno != (want)) {                                     \
            printf("FAIL %s: %ld (%s), want %s\n", #call, r_, strerror(errno), \
                   #want);                                                     \
            failed = 1;                                                        \
        }                                                                      \
    } while (0)

#define OK(call)                                                               \
    do {                                                                       \
        errno = 0;                                                             \
        long r_ = (long)(call);                                                \
        if (r_ < 0) {                                                          \
            printf("FAIL %s: %s\n", #call, strerror(errno));                   \
            failed = 1;                                                        \
        }                                                                      \
    } while (0)

int main(void) {
    const char *hook = ".git/hooks/pre-commit";

    // The attack: the parent can't move, so the hook can't be reached as
    // anything but itself.
    REFUSED(rename(".git", "g"), EPERM);
    REFUSED(open("g/hooks/pre-commit", O_WRONLY | O_TRUNC), ENOENT);

    // Nothing under a protected directory changes.
    REFUSED(open(hook, O_WRONLY), EROFS);
    REFUSED(open(hook, O_RDWR), EROFS);
    REFUSED(open(hook, O_RDONLY | O_TRUNC), EROFS);
    REFUSED(open(".git/hooks/post-merge", O_WRONLY | O_CREAT, 0755), EROFS);
    REFUSED(truncate(hook, 0), EROFS);
    REFUSED(unlink(hook), EROFS);
    REFUSED(chmod(hook, 0777), EROFS);
    REFUSED(utimes(hook, NULL), EROFS);
    REFUSED(symlink("/bin/sh", ".git/hooks/s"), EROFS);
    REFUSED(mkdir(".git/hooks/d", 0755), EROFS);
    REFUSED(mknod(".git/hooks/p", S_IFIFO | 0644, 0), EROFS);
    REFUSED(rename(hook, "stolen"), EROFS);
    REFUSED(rename(".git/hooks", "h"), EROFS);
    REFUSED(rmdir(".git/hooks"), EROFS);
    // ...and gets no second name to be written through.
    REFUSED(link(hook, "alias"), EPERM);

    // Read it, map it privately, but not shared and writable.
    int fd = open(hook, O_RDONLY);
    OK(fd);
    char buf[8] = {0};
    OK(read(fd, buf, sizeof buf));
    if (strncmp(buf, "good", 4) != 0) {
        printf("FAIL reading the hook: %.8s\n", buf);
        failed = 1;
    }
    errno = 0;
    void *m = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m != MAP_FAILED || errno != EACCES) {
        printf("FAIL shared writable mmap of the hook: %p (%s)\n", m, strerror(errno));
        failed = 1;
    }
    m = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) {
        printf("FAIL private mmap of the hook: %s\n", strerror(errno));
        failed = 1;
    }
    close(fd);

    // A symlink into a protected path leads to it, still protected.
    OK(symlink(".git/hooks", "h"));
    REFUSED(open("h/pre-commit", O_WRONLY), EROFS);
    REFUSED(open("h/new", O_WRONLY | O_CREAT, 0755), EROFS);
    OK(unlink("h"));

    // A protected single file.
    REFUSED(open("atelier.toml", O_WRONLY | O_APPEND), EROFS);
    OK(fd = open("other.toml", O_WRONLY | O_CREAT, 0644));
    close(fd);
    REFUSED(rename("other.toml", "atelier.toml"), EROFS);
    REFUSED(unlink("atelier.toml"), EROFS);

    // Everything else is as writable as ever.
    OK(fd = open(".git/config", O_WRONLY | O_TRUNC));
    OK(write(fd, "[user]\n", 7));
    close(fd);
    OK(rename("other.toml", ".git/other.toml"));
    OK(unlink(".git/other.toml"));
    OK(chmod(".git", 0755));

    // A repository without hooks can't gain them: `.git` may be made (as a
    // directory), hooks in it may not, however they'd arrive.
    OK(mkdir("bare/.git", 0755));
    REFUSED(mkdir("bare/.git/hooks", 0755), EROFS);
    REFUSED(symlink("../../g", "bare/.git/hooks"), EROFS);
    OK(mkdir("g", 0755));
    OK(mkdir("g/hooks", 0755));
    REFUSED(rename("g/hooks", "bare/.git/hooks"), EROFS);
    REFUSED(rmdir("bare/.git"), EPERM);
    OK(rmdir("g/hooks"));
    OK(rmdir("g"));

    // Another spelling of the name is the same name.
    REFUSED(rename(".GIT", "g"), EPERM);

    if (!failed)
        puts("PASS");
    return failed;
}
