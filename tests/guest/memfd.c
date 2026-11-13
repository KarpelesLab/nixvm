/* memfd and MAP_SHARED file mappings, as the heavy users of shared memory
 * (PostgreSQL, .NET's W^X double mapping, the JVM, runc's self-reexec) rely
 * on them: two mappings of the same pages with different protections are
 * coherent, with each other, with read()/write(), across fork and between
 * threads on different CPUs; mremap moves shared mappings without losing the
 * sharing; a shrinking ftruncate makes pages past EOF raise SIGBUS; the seals
 * (F_SEAL_WRITE, F_SEAL_FUTURE_WRITE, F_SEAL_EXEC/MFD_NOEXEC_SEAL) hold; a
 * memfd can be executed with fexecve/execveat; /proc names it. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/memfd.h>
#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef F_SEAL_FUTURE_WRITE
#define F_SEAL_FUTURE_WRITE 0x0010
#endif
#ifndef F_SEAL_EXEC
#define F_SEAL_EXEC 0x0020
#endif
#ifndef MFD_NOEXEC_SEAL
#define MFD_NOEXEC_SEAL 0x0008U
#endif
#ifndef MFD_EXEC
#define MFD_EXEC 0x0010U
#endif
#ifndef MREMAP_DONTUNMAP
#define MREMAP_DONTUNMAP 4
#endif

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

static long pg;

static int mfd(const char *name, unsigned flags, size_t size) {
  int fd = syscall(SYS_memfd_create, name, flags);
  CHECK(fd >= 0);
  if (size) CHECK(ftruncate(fd, size) == 0);
  return fd;
}

/* Touch `p`, reporting whether it raised SIGBUS (1), SIGSEGV (2) or not (0). */
static sigjmp_buf jb;
static void on_fault(int sig) { siglongjmp(jb, sig == SIGBUS ? 1 : 2); }
static int touch(volatile char *p, int write) {
  struct sigaction sa = {0};
  sa.sa_handler = on_fault;
  sigaction(SIGBUS, &sa, 0);
  sigaction(SIGSEGV, &sa, 0);
  int r = sigsetjmp(jb, 1);
  if (r == 0) {
    if (write) *p = 1;
    else (void)*p;
  }
  signal(SIGBUS, SIG_DFL);
  signal(SIGSEGV, SIG_DFL);
  return r;
}

/* /proc/self/maps has a line naming `what` that covers `addr`. */
static int maps_names(void *addr, const char *what) {
  FILE *f = fopen("/proc/self/maps", "r");
  CHECK(f);
  char line[512];
  int found = 0;
  while (fgets(line, sizeof line, f)) {
    unsigned long lo, hi;
    if (sscanf(line, "%lx-%lx", &lo, &hi) == 2 && (unsigned long)addr >= lo &&
        (unsigned long)addr < hi && strstr(line, what))
      found = 1;
  }
  fclose(f);
  return found;
}

static void double_mapping(void) {
  /* .NET's W^X: an RW view and an RX view of the same memfd pages. */
  int fd = mfd("doublemapper", MFD_CLOEXEC, 0);
  CHECK(ftruncate(fd, 1L << 40) == 0); /* a sparse terabyte, as .NET sizes it */
  char *res = mmap(0, 16 * pg, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  CHECK(res != MAP_FAILED);
  /* Commit an RX view over the reservation, then map an RW view elsewhere. */
  char *rx = mmap(res, 4 * pg, PROT_READ | PROT_EXEC, MAP_SHARED | MAP_FIXED, fd, 64 * pg);
  CHECK(rx == res);
  char *rw = mmap(0, 4 * pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 64 * pg);
  CHECK(rw != MAP_FAILED && rw != rx);
  /* x86-64: mov eax, 42; ret.  aarch64: mov w0, #42; ret. */
#if defined(__x86_64__)
  static const unsigned char code[] = {0xb8, 42, 0, 0, 0, 0xc3};
#elif defined(__aarch64__)
  static const unsigned char code[] = {0x40, 0x05, 0x80, 0x52, 0xc0, 0x03, 0x5f, 0xd6};
#endif
  memcpy(rw + pg, code, sizeof code);
  __builtin___clear_cache(rx + pg, rx + pg + sizeof code);
  CHECK(((int (*)(void))(rx + pg))() == 42);
  /* Rewrite through the RW view: the RX view runs the new code. */
  rw[pg + 1] = 7;
#if defined(__aarch64__)
  rw[pg] = 0xe0; /* mov w0, #7 */
  rw[pg + 1] = 0x00;
#endif
  __builtin___clear_cache(rx + pg, rx + pg + sizeof code);
  CHECK(((int (*)(void))(rx + pg))() == 7);
  CHECK(touch(rx, 1) == 2); /* the RX view isn't writable */
  /* pread sees the bytes the mapping stored, and pwrite lands in both views. */
  char b[4];
  CHECK(pread(fd, b, 1, 65 * pg + 1) == 1 && b[0] == rw[pg + 1]);
  CHECK(pwrite(fd, "hey", 3, 66 * pg) == 3);
  CHECK(memcmp(rx + 2 * pg, "hey", 3) == 0 && memcmp(rw + 2 * pg, "hey", 3) == 0);
  /* A third view at another offset doesn't alias the first. */
  char *other = mmap(0, pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  CHECK(other != MAP_FAILED && other[0] == 0);
  other[0] = 9;
  CHECK(rw[0] == 0);
  /* MAP_SHARED|MAP_FIXED over an existing view replaces it. */
  char *again = mmap(rx, pg, PROT_READ, MAP_SHARED | MAP_FIXED, fd, 0);
  CHECK(again == rx && rx[0] == 9);
  /* /proc/self/maps names the memfd. */
  CHECK(maps_names(rw, "/memfd:doublemapper (deleted)"));
  char link[256] = {0}, fdp[64];
  snprintf(fdp, sizeof fdp, "/proc/self/fd/%d", fd);
  CHECK(readlink(fdp, link, sizeof link - 1) > 0);
  CHECK(strcmp(link, "/memfd:doublemapper (deleted)") == 0);
  struct stat st;
  CHECK(fstat(fd, &st) == 0 && st.st_size == (1L << 40));
  CHECK(munmap(rw, 4 * pg) == 0 && munmap(other, pg) == 0 && munmap(res, 16 * pg) == 0);
  close(fd);
  puts("an RW and an RX view of one memfd are coherent");
}

static void across_fork(void) {
  int fd = mfd("fork", 0, 4 * pg);
  char *a = mmap(0, 4 * pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  char *priv = mmap(0, 4 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  CHECK(a != MAP_FAILED && priv != MAP_FAILED);
  strcpy(a, "parent");
  CHECK(strcmp(priv, "parent") == 0); /* an untouched private page reads the file */
  pid_t pid = fork();
  if (pid == 0) {
    /* The child's own new mapping and the inherited one both share. */
    char *c = mmap(0, 4 * pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (c == MAP_FAILED || strcmp(c, "parent") != 0) _exit(1);
    strcpy(a + pg, "child-inherited");
    strcpy(c + 2 * pg, "child-own");
    strcpy(priv + 3 * pg, "child-private");
    _exit(0);
  }
  int status;
  CHECK(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0);
  CHECK(strcmp(a + pg, "child-inherited") == 0);
  CHECK(strcmp(a + 2 * pg, "child-own") == 0);
  CHECK(a[3 * pg] == 0);
  /* A store to the private copy stays private… */
  priv[0] = 'P';
  CHECK(a[0] == 'p');
  /* …but its untouched pages follow the file. */
  strcpy(a + pg, "later");
  CHECK(strcmp(priv + pg, "later") == 0);
  munmap(a, 4 * pg);
  munmap(priv, 4 * pg);
  close(fd);
  puts("shared mappings are shared across fork, private ones aren't");
}

/* Two threads, likely on different CPUs: one stores through the RW view, the
 * other spins (yielding, for a single CPU) reading the RO view; then the
 * first replaces the page under the second's feet and it must see the new
 * one at once. */
static char *volatile seen_ro;
static atomic_int phase;
static void *reader(void *arg) {
  (void)arg;
  for (int i = 1; i <= 2000; i++) {
    while (*(volatile int *)seen_ro != i) {
      sched_yield();
    }
    atomic_store(&phase, i);
  }
  while (atomic_load(&phase) != -1) {
    sched_yield();
  }
  /* The page was replaced: wait to read the new contents. */
  while (*(volatile int *)seen_ro != 0x5eed) {
    sched_yield();
  }
  atomic_store(&phase, -2);
  return 0;
}

static void across_threads(void) {
  int fd = mfd("threads", 0, 2 * pg);
  int *rw = mmap(0, pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  char *ro = mmap(0, pg, PROT_READ, MAP_SHARED, fd, 0);
  CHECK(rw != MAP_FAILED && ro != MAP_FAILED);
  seen_ro = ro;
  pthread_t t;
  CHECK(pthread_create(&t, 0, reader, 0) == 0);
  for (int i = 1; i <= 2000; i++) {
    *(volatile int *)rw = i;
    while (atomic_load(&phase) != i) {
      sched_yield();
    }
  }
  /* Page 1 of the file holds the new value; map it over the reader's view. */
  ((int *)((char *)rw))[0] = 2000;
  CHECK(pwrite(fd, &(int){0x5eed}, 4, pg) == 4);
  CHECK(mmap(ro, pg, PROT_READ, MAP_SHARED | MAP_FIXED, fd, pg) == ro);
  atomic_store(&phase, -1);
  while (atomic_load(&phase) != -2) {
    sched_yield();
  }
  pthread_join(t, 0);
  munmap(rw, pg);
  munmap(ro, pg);
  close(fd);
  puts("threads see each other's stores through two views");
}

static void remapping(void) {
  int fd = mfd("remap", 0, 8 * pg);
  char *a = mmap(0, 2 * pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  char *view = mmap(0, 8 * pg, PROT_READ, MAP_SHARED, fd, 0);
  CHECK(a != MAP_FAILED && view != MAP_FAILED);
  strcpy(a, "zero");
  strcpy(a + pg, "one");
  /* Grow with MREMAP_MAYMOVE: the new pages are the file's next pages. */
  char *g = mremap(a, 2 * pg, 6 * pg, MREMAP_MAYMOVE);
  CHECK(g != MAP_FAILED);
  CHECK(strcmp(g, "zero") == 0 && strcmp(g + pg, "one") == 0);
  strcpy(g + 5 * pg, "five");
  CHECK(strcmp(view + 5 * pg, "five") == 0);
  g[0] = 'Z';
  CHECK(view[0] == 'Z');
  /* MREMAP_FIXED to a chosen place, replacing what was there. */
  char *dst = mmap(0, 6 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  CHECK(dst != MAP_FAILED);
  char *f = mremap(g, 6 * pg, 6 * pg, MREMAP_MAYMOVE | MREMAP_FIXED, dst);
  CHECK(f == dst);
  CHECK(touch(g, 0) == 2); /* the old range is gone */
  f[pg] = 'O';
  CHECK(view[pg] == 'O' && strcmp(f + 5 * pg, "five") == 0);
  /* Shrink: the tail goes away, the rest stays shared. */
  CHECK(mremap(f, 6 * pg, 3 * pg, 0) == f);
  CHECK(touch(f + 4 * pg, 0) == 2);
  f[2 * pg] = '2';
  CHECK(view[2 * pg] == '2');
  /* MREMAP_DONTUNMAP of a shared mapping: both ranges still map the file. */
  char *d = mremap(f, 3 * pg, 3 * pg, MREMAP_MAYMOVE | MREMAP_DONTUNMAP, (void *)0);
  CHECK(d != MAP_FAILED && d != f);
  d[0] = 'D';
  CHECK(view[0] == 'D' && f[0] == 'D');
  /* MREMAP_DONTUNMAP of private anonymous memory: the old range reads zero. */
  char *anon = mmap(0, 2 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  CHECK(anon != MAP_FAILED);
  strcpy(anon, "moved");
  char *m = mremap(anon, 2 * pg, 2 * pg, MREMAP_MAYMOVE | MREMAP_DONTUNMAP, (void *)0);
  CHECK(m != MAP_FAILED && strcmp(m, "moved") == 0 && anon[0] == 0);
  /* …and it must keep its size. */
  errno = 0;
  CHECK(mremap(m, 2 * pg, 4 * pg, MREMAP_MAYMOVE | MREMAP_DONTUNMAP, (void *)0) == MAP_FAILED && errno == EINVAL);
  /* A private file mapping grows with more of the file. */
  char *p = mmap(0, pg, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  CHECK(p != MAP_FAILED);
  p[0] = 'p';
  char *pg2 = mremap(p, pg, 6 * pg, MREMAP_MAYMOVE);
  CHECK(pg2 != MAP_FAILED && pg2[0] == 'p' && strcmp(pg2 + 5 * pg, "five") == 0);
  CHECK(view[0] == 'D');
  munmap(pg2, 6 * pg);
  munmap(m, 2 * pg);
  munmap(anon, 2 * pg);
  munmap(d, 3 * pg);
  munmap(f, 3 * pg);
  munmap(view, 8 * pg);
  close(fd);
  puts("mremap moves and resizes shared and private file mappings");
}

static void truncation(void) {
  int fd = mfd("trunc", 0, 4 * pg);
  char *a = mmap(0, 6 * pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  CHECK(a != MAP_FAILED);
  memset(a, 'x', 4 * pg);
  CHECK(touch(a + 4 * pg, 0) == 1); /* past EOF from the start */
  CHECK(touch(a + 5 * pg, 1) == 1);
  /* Shrink to a page and a half: the rest of the second page reads zero and
   * the pages after it raise SIGBUS. */
  CHECK(ftruncate(fd, pg + pg / 2) == 0);
  CHECK(a[0] == 'x' && a[pg] == 'x' && a[pg + pg / 2] == 0);
  CHECK(touch(a + 2 * pg, 0) == 1);
  CHECK(touch(a + 3 * pg, 1) == 1);
  /* A private mapping past EOF raises SIGBUS too. */
  char *p = mmap(0, 3 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  CHECK(p != MAP_FAILED && p[0] == 'x');
  CHECK(touch(p + 2 * pg, 0) == 1);
  /* Growing again brings the pages back, zero. */
  CHECK(ftruncate(fd, 4 * pg) == 0);
  CHECK(a[2 * pg] == 0 && a[3 * pg + 7] == 0);
  a[3 * pg] = 'y';
  char c;
  CHECK(pread(fd, &c, 1, 3 * pg) == 1 && c == 'y');
  /* A write() extending the file is seen through the mapping. */
  CHECK(pwrite(fd, "w", 1, 5 * pg) == 1);
  CHECK(a[5 * pg] == 'w' && a[4 * pg] == 0);
  munmap(a, 6 * pg);
  munmap(p, 3 * pg);
  close(fd);
  puts("pages past EOF raise SIGBUS, and come back when the file grows");
}

static void seals(void) {
  /* F_SEAL_WRITE: refused while a writable shared mapping exists. */
  int fd = mfd("seal", MFD_ALLOW_SEALING, 2 * pg);
  char *w = mmap(0, 2 * pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  CHECK(w != MAP_FAILED);
  w[0] = 's';
  errno = 0;
  CHECK(fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE) == -1 && errno == EBUSY);
  /* A read-only shared mapping of a writable descriptor could be made
   * writable, so it counts too... */
  char *maybe = mmap(0, 2 * pg, PROT_READ, MAP_SHARED, fd, 0);
  CHECK(maybe != MAP_FAILED);
  CHECK(munmap(w, 2 * pg) == 0);
  errno = 0;
  CHECK(fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE) == -1 && errno == EBUSY);
  CHECK(munmap(maybe, 2 * pg) == 0);
  /* ...but one of a read-only descriptor doesn't, nor does a private one. */
  char fdp[64];
  snprintf(fdp, sizeof fdp, "/proc/self/fd/%d", fd);
  int rofd = open(fdp, O_RDONLY);
  CHECK(rofd >= 0);
  char *r = mmap(0, 2 * pg, PROT_READ, MAP_SHARED, rofd, 0);
  char *pv = mmap(0, 2 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  CHECK(r != MAP_FAILED && pv != MAP_FAILED && r[0] == 's');
  errno = 0;
  CHECK(mprotect(r, pg, PROT_READ | PROT_WRITE) == -1 && errno == EACCES);
  close(rofd);
  CHECK(fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW) == 0);
  /* A read-only shared mapping made after sealing can't become writable. */
  char *after = mmap(0, pg, PROT_READ, MAP_SHARED, fd, 0);
  CHECK(after != MAP_FAILED && after[0] == 's');
  errno = 0;
  CHECK(mprotect(after, pg, PROT_READ | PROT_WRITE) == -1 && errno == EACCES);
  munmap(after, pg);
  CHECK(fcntl(fd, F_GET_SEALS) == (F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW));
  errno = 0;
  CHECK(mmap(0, pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) == MAP_FAILED && errno == EPERM);
  errno = 0;
  CHECK(mprotect(r, pg, PROT_READ | PROT_WRITE) == -1 && errno == EACCES);
  errno = 0;
  CHECK(write(fd, "x", 1) == -1 && errno == EPERM);
  errno = 0;
  CHECK(ftruncate(fd, pg) == -1 && errno == EPERM);
  errno = 0;
  CHECK(ftruncate(fd, 4 * pg) == -1 && errno == EPERM);
  errno = 0;
  CHECK(fallocate(fd, 0, 0, 8 * pg) == -1 && errno == EPERM);
  /* A private writable mapping still works, privately. */
  pv[0] = 'q';
  CHECK(r[0] == 's');
  CHECK(r[0] == 's');
  /* A memfd without MFD_ALLOW_SEALING can't be sealed. */
  int nos = mfd("noseal", 0, 0);
  errno = 0;
  CHECK(fcntl(nos, F_ADD_SEALS, F_SEAL_WRITE) == -1 && errno == EPERM);
  CHECK(fcntl(nos, F_GET_SEALS) == F_SEAL_SEAL);
  close(nos);
  munmap(r, 2 * pg);
  munmap(pv, 2 * pg);
  close(fd);

  /* F_SEAL_FUTURE_WRITE: existing writable mappings keep writing; nothing new may. */
  fd = mfd("future", MFD_ALLOW_SEALING, pg);
  w = mmap(0, pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  CHECK(w != MAP_FAILED);
  CHECK(fcntl(fd, F_ADD_SEALS, F_SEAL_FUTURE_WRITE) == 0);
  w[1] = 'f';
  char c;
  CHECK(pread(fd, &c, 1, 1) == 1 && c == 'f');
  errno = 0;
  CHECK(write(fd, "x", 1) == -1 && errno == EPERM);
  errno = 0;
  CHECK(mmap(0, pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) == MAP_FAILED && errno == EPERM);
  r = mmap(0, pg, PROT_READ, MAP_SHARED, fd, 0);
  CHECK(r != MAP_FAILED && r[1] == 'f');
  errno = 0;
  CHECK(mprotect(r, pg, PROT_READ | PROT_WRITE) == -1 && errno == EACCES);
  pv = mmap(0, pg, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  CHECK(pv != MAP_FAILED);
  pv[1] = 'g';
  CHECK(r[1] == 'f');
  munmap(w, pg);
  munmap(r, pg);
  munmap(pv, pg);
  close(fd);
  puts("F_SEAL_WRITE and F_SEAL_FUTURE_WRITE hold");
}

/* Copy this program into a memfd and run it again from there. */
static int exec_child(int how, int fd) {
  pid_t pid = fork();
  if (pid == 0) {
    char *argv[] = {"memfd", "child", 0};
    char *envp[] = {0};
    if (how == 0) fexecve(fd, argv, envp);
    else syscall(SYS_execveat, fd, "", argv, envp, AT_EMPTY_PATH);
    _exit(errno == EACCES ? 13 : 99);
  }
  int status;
  CHECK(waitpid(pid, &status, 0) == pid && WIFEXITED(status));
  return WEXITSTATUS(status);
}

static void executing(void) {
  int self = open("/proc/self/exe", O_RDONLY);
  CHECK(self >= 0);
  int fd = mfd("runc", MFD_CLOEXEC | MFD_ALLOW_SEALING, 0);
  char buf[65536];
  ssize_t n;
  while ((n = read(self, buf, sizeof buf)) > 0) CHECK(write(fd, buf, n) == n);
  close(self);
  struct stat st;
  CHECK(fstat(fd, &st) == 0 && (st.st_mode & 0777) == 0777);
  /* runc seals its copy before running it. */
  CHECK(fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL) == 0);
  CHECK(exec_child(0, fd) == 42); /* fexecve, even with MFD_CLOEXEC */
  CHECK(exec_child(1, fd) == 42); /* execveat(fd, "", AT_EMPTY_PATH) */
  close(fd);

  /* MFD_NOEXEC_SEAL: not executable, and sealed so it can't become so. */
  fd = mfd("noexec", MFD_NOEXEC_SEAL, 0);
  self = open("/proc/self/exe", O_RDONLY);
  while ((n = read(self, buf, sizeof buf)) > 0) CHECK(write(fd, buf, n) == n);
  close(self);
  CHECK(fstat(fd, &st) == 0 && (st.st_mode & 0777) == 0666);
  CHECK(fcntl(fd, F_GET_SEALS) == F_SEAL_EXEC);
  errno = 0;
  CHECK(fchmod(fd, 0777) == -1 && errno == EPERM);
  CHECK(exec_child(0, fd) == 13);
  /* MFD_NOEXEC_SEAL makes the memfd sealable. */
  CHECK(fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE) == 0);
  close(fd);
  /* MFD_EXEC with MFD_NOEXEC_SEAL is a contradiction. */
  errno = 0;
  CHECK(syscall(SYS_memfd_create, "x", MFD_EXEC | MFD_NOEXEC_SEAL) == -1 && errno == EINVAL);
  /* A memfd without the exec bit can't be run. */
  fd = mfd("chmod", 0, 0);
  self = open("/proc/self/exe", O_RDONLY);
  while ((n = read(self, buf, sizeof buf)) > 0) CHECK(write(fd, buf, n) == n);
  close(self);
  CHECK(fchmod(fd, 0644) == 0);
  CHECK(exec_child(0, fd) == 13);
  close(fd);
  puts("a memfd runs with fexecve and execveat, unless sealed against it");
}

static void secret(void) {
  int fd = syscall(447 /* memfd_secret */, 0);
  if (fd < 0) {
    CHECK(errno == ENOSYS);
    puts("memfd_secret isn't there");
    return;
  }
  CHECK(ftruncate(fd, pg) == 0);
  errno = 0;
  CHECK(mmap(0, pg, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0) == MAP_FAILED && errno == EINVAL);
  char *s = mmap(0, pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  CHECK(s != MAP_FAILED);
  strcpy(s, "secret");
  CHECK(strcmp(s, "secret") == 0);
  munmap(s, pg);
  close(fd);
  puts("memfd_secret maps shared only");
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "child") == 0) return 42;
  pg = sysconf(_SC_PAGESIZE);
  double_mapping();
  across_fork();
  across_threads();
  remapping();
  truncation();
  seals();
  executing();
  secret();
  puts("PASS");
  return 0;
}
