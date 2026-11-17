/* What Wine asks of the kernel beyond what libc does (each check named after
 * the gap it once was):
 *  - arch_prctl(ARCH_SET_GS): x86-64 Wine keeps the TEB at %gs:0x30. The GS
 *    base is per thread, read back by ARCH_GET_GS, kept across a signal
 *    handler, inherited by a new thread and a fork child, and cleared by
 *    exec (checked by re-executing ourselves).
 *  - The top of the address space is a power of two: Wine finds it by mapping
 *    a page at 2^63, 2^62, ... with MAP_FIXED_NOREPLACE and takes twice the
 *    first address that maps as the end, then maps just below that end.
 *  - select/pselect6 with no descriptors is a sleep (Wine's Sleep() is
 *    select(0, 0, 0, 0, &tv)): it returns 0 when the timeout runs out, with
 *    the time left (none) written back, and with a ready descriptor it
 *    returns at once, leaving most of the timeout.
 *  - lstat("link/") follows the link, the trailing slash asking for the
 *    directory (Wine stats dosdevices/c:/ to see drive C's root).
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <time.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL " __VA_ARGS__);                                             \
      printf("\n");                                                            \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#if defined(__x86_64__)
#include <asm/prctl.h>

static uint64_t teb[8] = {0x1111, 0x2222, 0x3333};

static uint64_t gs_read8(void) {
  uint64_t v;
  __asm__ volatile("movq %%gs:8, %0" : "=r"(v));
  return v;
}

static uint64_t get_gs(void) {
  uint64_t base = 0xdead;
  if (syscall(SYS_arch_prctl, ARCH_GET_GS, &base) != 0)
    return 0xbad;
  return base;
}

static volatile uint64_t in_handler;
static void on_usr1(int sig) {
  (void)sig;
  in_handler = gs_read8();
}

static void *thread_main(void *arg) {
  (void)arg;
  uint64_t inherited = get_gs();
  static uint64_t own[2] = {0, 0x7777};
  syscall(SYS_arch_prctl, ARCH_SET_GS, own);
  return (void *)(uintptr_t)(inherited == (uintptr_t)teb && gs_read8() == 0x7777);
}

static void gs_base(const char *self) {
  CHECK(syscall(SYS_arch_prctl, ARCH_SET_GS, teb) == 0, "ARCH_SET_GS refused");
  CHECK(get_gs() == (uintptr_t)teb, "ARCH_GET_GS gave %#lx", (long)get_gs());
  CHECK(gs_read8() == 0x2222, "%%gs:8 read %#lx", (long)gs_read8());

  signal(SIGUSR1, on_usr1);
  raise(SIGUSR1);
  CHECK(in_handler == 0x2222, "GS base in a handler: %%gs:8 = %#lx", (long)in_handler);
  CHECK(gs_read8() == 0x2222, "GS base lost across a handler");

  pthread_t t;
  void *ok = 0;
  pthread_create(&t, 0, thread_main, 0);
  pthread_join(t, &ok);
  CHECK(ok, "a thread doesn't inherit (or can't set) its GS base");
  CHECK(gs_read8() == 0x2222, "a thread's ARCH_SET_GS changed ours");

  pid_t pid = fork();
  if (pid == 0)
    _exit(gs_read8() == 0x2222 ? 0 : 1);
  int st = 0;
  waitpid(pid, &st, 0);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "a fork child lost the GS base");

  CHECK(syscall(SYS_arch_prctl, ARCH_SET_GS, 0x8000000000000000ul) == -1,
        "a non-canonical GS base was accepted");

  pid = fork();
  if (pid == 0) {
    execl(self, self, "gs-after-exec", (char *)0);
    _exit(2);
  }
  waitpid(pid, &st, 0);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "exec kept the GS base (%#x)", st);
}
#endif

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

static void address_space_top(void) {
  uintptr_t addr = (uintptr_t)1 << 63;
  while (addr >> 32) {
    void *p = mmap((void *)addr, 4096, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (p != MAP_FAILED) {
      munmap(p, 4096);
      if ((uintptr_t)p >= addr)
        break;
    } else if (errno == EEXIST) {
      break;
    }
    addr >>= 1;
  }
  /* Wine's limit, (addr << 1) - 64K, and its last 64K granule below it, are
   * inside the address space: the granule maps, or is taken (the stack sits
   * at the top), but isn't beyond the end (ENOMEM). */
  uintptr_t top = (addr << 1) - 0x20000;
  void *p = mmap((void *)top, 0x10000, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  CHECK(p != MAP_FAILED || errno == EEXIST,
        "the address space doesn't end at a power of two: %#lx maps, %#lx "
        "is beyond the end (%s)", (long)addr, (long)top, strerror(errno));
  if (p != MAP_FAILED)
    munmap(p, 0x10000);
}

static long long now_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000000000LL + t.tv_nsec;
}

static void select_sleeps(void) {
  struct timeval tv = {0, 20000};
  long long t0 = now_ns();
  int r = select(0, NULL, NULL, NULL, &tv);
  long long dt = now_ns() - t0;
  CHECK(r == 0, "select(0, ..., 20ms) returned %d", r);
  CHECK(dt >= 20000000 && dt < 2000000000LL, "select(0, ..., 20ms) took %lldns", dt);
  CHECK(tv.tv_sec == 0 && tv.tv_usec == 0, "select left %ld.%06ld of its timeout",
        (long)tv.tv_sec, (long)tv.tv_usec);

  struct timespec ts = {0, 5000000};
  t0 = now_ns();
  r = pselect(0, NULL, NULL, NULL, &ts, NULL);
  dt = now_ns() - t0;
  CHECK(r == 0 && dt >= 5000000, "pselect(0, ..., 5ms) returned %d after %lldns", r, dt);

  int p[2];
  if (pipe(p) == 0) {
    write(p[1], "x", 1);
    fd_set rd;
    FD_ZERO(&rd);
    FD_SET(p[0], &rd);
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    r = select(p[0] + 1, &rd, NULL, NULL, &tv);
    CHECK(r == 1 && FD_ISSET(p[0], &rd), "select on a readable pipe returned %d", r);
    CHECK(tv.tv_sec >= 4 && tv.tv_sec <= 5, "a ready select left %ld s of 5", (long)tv.tv_sec);
    close(p[0]);
    close(p[1]);
  }

  tv.tv_sec = 0;
  tv.tv_usec = -1;
  errno = 0;
  r = select(0, NULL, NULL, NULL, &tv);
  CHECK(r == -1 && errno == EINVAL, "select with tv_usec -1 returned %d (errno %d)", r, errno);
}

static void trailing_slash_follows(void) {
  char dir[] = "/tmp/wine_abi.XXXXXX";
  if (!mkdtemp(dir)) {
    CHECK(0, "mkdtemp: %s", strerror(errno));
    return;
  }
  char sub[64], dlink[64], file[64], flink[64], dslash[64], fslash[64];
  snprintf(sub, sizeof sub, "%s/d", dir);
  snprintf(dlink, sizeof dlink, "%s/c:", dir);
  snprintf(file, sizeof file, "%s/f", dir);
  snprintf(flink, sizeof flink, "%s/fl", dir);
  snprintf(dslash, sizeof dslash, "%s/", dlink);
  snprintf(fslash, sizeof fslash, "%s/", flink);
  mkdir(sub, 0755);
  symlink("d", dlink);
  close(open(file, O_CREAT | O_WRONLY, 0644));
  symlink("f", flink);
  struct stat st;
  CHECK(lstat(dlink, &st) == 0 && S_ISLNK(st.st_mode), "lstat(link) isn't the link");
  CHECK(lstat(dslash, &st) == 0 && S_ISDIR(st.st_mode),
        "lstat(\"link/\") isn't the directory (%s)", strerror(errno));
  CHECK(fstatat(AT_FDCWD, dslash, &st, AT_SYMLINK_NOFOLLOW) == 0 && S_ISDIR(st.st_mode),
        "fstatat(\"link/\", NOFOLLOW) isn't the directory");
  errno = 0;
  CHECK(lstat(fslash, &st) == -1 && errno == ENOTDIR,
        "lstat(\"link-to-file/\") gave %s, not ENOTDIR", strerror(errno));
  unlink(flink);
  unlink(file);
  unlink(dlink);
  rmdir(sub);
  rmdir(dir);
}

int main(int argc, char **argv) {
#if defined(__x86_64__)
  if (argc > 1 && !strcmp(argv[1], "gs-after-exec"))
    return get_gs() == 0 ? 0 : 1;
  /* In the VM the program is /wine_abi, which /proc/self/exe may not name. */
  gs_base(access("/wine_abi", X_OK) == 0 ? "/wine_abi" : "/proc/self/exe");
#else
  (void)argc;
  (void)argv;
#endif
  address_space_top();
  select_sleeps();
  trailing_slash_follows();
  if (failures)
    return 1;
  printf("PASS\n");
  return 0;
}
