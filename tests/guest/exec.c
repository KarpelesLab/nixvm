/* execve: what fails before the point of no return leaves the caller running,
 * and a dynamic linker is found through relative symlinks in its directories
 * and in its last component, as Debian's merged /usr lays them out:
 *   /lib64 -> usr/lib64
 *   /usr/lib64/ld-linux-x86-64.so.2 -> ../lib/x86_64-linux-gnu/ld-linux-x86-64.so.2
 * The files exec'd are tiny ELF images written here; the "dynamic linker" is
 * one that exits 42, so its running is how the right one is seen to be found. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

#ifdef __aarch64__
#define MACHINE 183
#define OTHER_MACHINE 62
/* mov x0, #42; mov x8, #93 (exit); svc #0 */
static const uint8_t EXIT42[] = {0x40, 0x05, 0x80, 0xd2, 0xa8, 0x0b, 0x80, 0xd2, 0x01, 0x00, 0x00, 0xd4};
#else
#define MACHINE 62
#define OTHER_MACHINE 183
/* mov edi, 42; mov eax, 60 (exit); syscall */
static const uint8_t EXIT42[] = {0xbf, 0x2a, 0, 0, 0, 0xb8, 0x3c, 0, 0, 0, 0x0f, 0x05};
#endif

static void put16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static void put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void put64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }

/* An ELF64 executable loaded whole at `vaddr`, starting at EXIT42; naming
 * `interp` as its dynamic linker when that's not NULL. */
static void write_elf(const char *path, uint64_t vaddr, const char *interp, uint16_t machine) {
  uint8_t f[512] = {0};
  int nph = interp ? 2 : 1;
  size_t ph = 64, str = ph + 56 * nph, code = str + (interp ? strlen(interp) + 1 : 0);
  code = (code + 15) & ~(size_t)15;
  size_t size = code + sizeof EXIT42;
  memcpy(f, "\177ELF\2\1\1", 7);
  put16(f + 16, 2); /* ET_EXEC */
  put16(f + 18, machine);
  put32(f + 20, 1);
  put64(f + 24, vaddr + code);
  put64(f + 32, ph);
  put16(f + 52, 64);
  put16(f + 54, 56);
  put16(f + 56, nph);
  uint8_t *p = f + ph;
  if (interp) {
    put32(p, 3); /* PT_INTERP */
    put32(p + 4, 4);
    put64(p + 8, str);
    put64(p + 16, vaddr + str);
    put64(p + 32, strlen(interp) + 1);
    put64(p + 40, strlen(interp) + 1);
    put64(p + 48, 1);
    strcpy((char *)f + str, interp);
    p += 56;
  }
  put32(p, 1); /* PT_LOAD */
  put32(p + 4, 5); /* R+X */
  put64(p + 16, vaddr);
  put64(p + 24, vaddr);
  put64(p + 32, size);
  put64(p + 40, size);
  put64(p + 48, 0x1000);
  memcpy(f + code, EXIT42, sizeof EXIT42);
  FILE *o = fopen(path, "wb");
  CHECK(o && fwrite(f, 1, size, o) == size && fclose(o) == 0);
  CHECK(chmod(path, 0755) == 0);
}

static void write_bytes(const char *path, const void *b, size_t n) {
  FILE *o = fopen(path, "wb");
  CHECK(o && fwrite(b, 1, n, o) == n && fclose(o) == 0);
  CHECK(chmod(path, 0755) == 0);
}

/* execve `path` in this process: it must fail with `want` and come back. */
static void exec_fails(const char *path, int want) {
  char *argv[] = {(char *)path, 0};
  errno = 0;
  int r = execve(path, argv, (char *[]){0});
  printf("exec %s: %d, errno %d (want %d), and we're still here\n", path, r, errno, want);
  CHECK(r == -1 && errno == want);
}

/* What `path` exits with, exec'd in a child. */
static int exit_status(const char *path) {
  pid_t p = fork();
  CHECK(p >= 0);
  if (!p) {
    char *argv[] = {(char *)path, 0};
    execve(path, argv, (char *[]){0});
    _exit(100 + errno);
  }
  int st;
  CHECK(waitpid(p, &st, 0) == p);
  return WIFEXITED(st) ? WEXITSTATUS(st) : 200 + WTERMSIG(st);
}

int main(void) {
  /* Parallel runs share /tmp on the host: a name of our own. */
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  char d[64], p[160], q[160];
  snprintf(d, sizeof d, "/tmp/exec-%d-%ld", getpid(), now.tv_nsec);
  CHECK(mkdir(d, 0755) == 0);

  snprintf(p, sizeof p, "%s/nolinker", d);
  write_elf(p, 0x400000, "/nonexistent/ld.so", MACHINE);
  exec_fails(p, ENOENT);
  snprintf(p, sizeof p, "%s/short", d);
  write_bytes(p, "\177ELF", 4);
  exec_fails(p, ENOEXEC);
  snprintf(p, sizeof p, "%s/foreign", d);
  write_elf(p, 0x400000, 0, OTHER_MACHINE);
  exec_fails(p, ENOEXEC);
  snprintf(p, sizeof p, "%s/missing", d);
  exec_fails(p, ENOENT);

  /* A static image runs; one whose linker is found through relative links
   * runs that linker. */
  snprintf(p, sizeof p, "%s/static", d);
  write_elf(p, 0x400000, 0, MACHINE);
  int st = exit_status(p);
  printf("static image exits %d\n", st);
  CHECK(st == 42);
  snprintf(p, sizeof p, "%s/usr", d); CHECK(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/usr/lib", d); CHECK(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/usr/lib/multiarch", d); CHECK(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/usr/lib64", d); CHECK(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/lib64", d); CHECK(symlink("usr/lib64", p) == 0);
  snprintf(p, sizeof p, "%s/usr/lib64/ld.so", d);
  CHECK(symlink("../lib/multiarch/ld.so", p) == 0);
  snprintf(p, sizeof p, "%s/usr/lib/multiarch/ld.so", d);
  write_elf(p, 0x800000, 0, MACHINE);
  snprintf(q, sizeof q, "%s/lib64/ld.so", d);
  snprintf(p, sizeof p, "%s/dynamic", d);
  write_elf(p, 0x400000, q, MACHINE);
  st = exit_status(p);
  printf("image with linker %s exits %d\n", q, st);
  CHECK(st == 42);
  puts("PASS");
  return 0;
}
