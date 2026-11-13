/* System V and POSIX shared memory as PostgreSQL and Python's multiprocessing
 * use them: a segment attached twice (once read-only) is one memory, shared
 * with a fork child that inherits the attachment; shm_nattch counts every
 * attachment; IPC_RMID with segments attached defers destruction (the key is
 * released at once); SHM_LOCK sets SHM_LOCKED; /proc/self/maps names an
 * attachment "/SYSV<key> (deleted)" and shared anonymous memory
 * "/dev/zero (deleted)"; a process-shared semaphore in /dev/shm wakes a
 * waiter in another process; and MAP_SHARED|MAP_ANONYMOUS stays shared
 * across fork even when it was mapped read-only first. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

static long pg;

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

static int nattch(int id) {
  struct shmid_ds ds;
  CHECK(shmctl(id, IPC_STAT, &ds) == 0);
  return (int)ds.shm_nattch;
}

static void sysv(void) {
  int id = shmget(IPC_PRIVATE, 3 * pg, IPC_CREAT | 0600);
  CHECK(id >= 0);
  char *a = shmat(id, 0, 0);
  char *r = shmat(id, 0, SHM_RDONLY);
  CHECK(a != (void *)-1 && r != (void *)-1 && a != r);
  CHECK(nattch(id) == 2);
  strcpy(a + pg, "seg");
  CHECK(strcmp(r + pg, "seg") == 0);
  CHECK(maps_names(a, "/SYSV00000000 (deleted)"));
  /* The read-only attachment can't be made writable. */
  errno = 0;
  CHECK(mprotect(r, pg, PROT_READ | PROT_WRITE) == -1 && errno == EACCES);
  /* A fork child inherits both attachments, and its stores are seen. */
  pid_t pid = fork();
  if (pid == 0) {
    if (nattch(id) != 4) _exit(1);
    strcpy(a + 2 * pg, "child");
    shmdt(r);
    _exit(nattch(id) == 3 ? 0 : 2);
  }
  int status;
  CHECK(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0);
  CHECK(strcmp(r + 2 * pg, "child") == 0);
  CHECK(nattch(id) == 2);
  /* SHM_LOCK marks the segment locked; SHM_UNLOCK clears it. */
  struct shmid_ds ds;
  CHECK(shmctl(id, SHM_LOCK, 0) == 0);
  CHECK(shmctl(id, IPC_STAT, &ds) == 0 && (ds.shm_perm.mode & SHM_LOCKED));
  CHECK(shmctl(id, SHM_UNLOCK, 0) == 0);
  CHECK(shmctl(id, IPC_STAT, &ds) == 0 && !(ds.shm_perm.mode & SHM_LOCKED));
  /* IPC_RMID while attached: still there, marked for destruction. */
  CHECK(shmctl(id, IPC_RMID, 0) == 0);
  CHECK(shmctl(id, IPC_STAT, &ds) == 0 && (ds.shm_perm.mode & SHM_DEST) && ds.shm_nattch == 2);
  strcpy(a, "after rmid");
  CHECK(strcmp(r, "after rmid") == 0);
  CHECK(shmdt(r) == 0 && shmdt(a) == 0);
  errno = 0;
  CHECK(shmctl(id, IPC_STAT, &ds) == -1 && (errno == EINVAL || errno == EIDRM));

  /* A keyed segment: the key is free again once removed, even attached. */
  key_t key = 0x5eed0000 + getpid();
  int k1 = shmget(key, pg, IPC_CREAT | IPC_EXCL | 0600);
  CHECK(k1 >= 0);
  char *k = shmat(k1, 0, 0);
  CHECK(k != (void *)-1);
  char want[64];
  snprintf(want, sizeof want, "/SYSV%08x (deleted)", (unsigned)key);
  CHECK(maps_names(k, want));
  CHECK(shmctl(k1, IPC_RMID, 0) == 0);
  CHECK(shmctl(k1, IPC_STAT, &ds) == 0 && ds.shm_perm.__key == IPC_PRIVATE);
  int k2 = shmget(key, pg, IPC_CREAT | IPC_EXCL | 0600);
  CHECK(k2 >= 0 && k2 != k1);
  CHECK(shmctl(k2, IPC_RMID, 0) == 0);
  CHECK(shmdt(k) == 0);
  /* Attaching over a mapping needs SHM_REMAP. */
  int id3 = shmget(IPC_PRIVATE, pg, IPC_CREAT | 0600);
  char *over = mmap(0, pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  CHECK(over != MAP_FAILED);
  errno = 0;
  CHECK(shmat(id3, over, 0) == (void *)-1 && errno == EINVAL);
  CHECK(shmat(id3, over, SHM_REMAP) == over);
  CHECK(shmdt(over) == 0 && shmctl(id3, IPC_RMID, 0) == 0);
  puts("SysV segments are shared, counted and removed as on Linux");
}

static void shared_anon(void) {
  /* Mapped read-only, made writable later: still one memory across fork. */
  char *s = mmap(0, 2 * pg, PROT_READ, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  CHECK(s != MAP_FAILED);
  CHECK(maps_names(s, "/dev/zero (deleted)"));
  CHECK(mprotect(s, 2 * pg, PROT_READ | PROT_WRITE) == 0);
  pid_t pid = fork();
  if (pid == 0) {
    strcpy(s + pg, "from the child");
    _exit(0);
  }
  int status;
  CHECK(waitpid(pid, &status, 0) == pid && WIFEXITED(status));
  CHECK(strcmp(s + pg, "from the child") == 0);
  /* MADV_REMOVE punches the memory out: zeros, for everyone. */
  CHECK(madvise(s + pg, pg, MADV_REMOVE) == 0);
  CHECK(s[pg] == 0);
  CHECK(munmap(s, 2 * pg) == 0);
  puts("shared anonymous memory stays shared");
}

static void posix(void) {
  char name[64];
  snprintf(name, sizeof name, "/sysv_shm_test.%d", getpid());
  int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
  CHECK(fd >= 0);
  CHECK(ftruncate(fd, pg) == 0);
  sem_t *sem = mmap(0, pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  CHECK(sem != MAP_FAILED);
  CHECK(sem_init(sem, 1, 0) == 0);
  pid_t pid = fork();
  if (pid == 0) {
    /* The child maps the object anew and waits; the parent posts. */
    int cfd = shm_open(name, O_RDWR, 0);
    if (cfd < 0) _exit(1);
    sem_t *cs = mmap(0, pg, PROT_READ | PROT_WRITE, MAP_SHARED, cfd, 0);
    if (cs == MAP_FAILED) _exit(2);
    if (sem_wait(cs) != 0) _exit(3);
    strcpy((char *)cs + 512, "woken");
    _exit(0);
  }
  usleep(100000);
  CHECK(sem_post(sem) == 0);
  int status;
  CHECK(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0);
  CHECK(strcmp((char *)sem + 512, "woken") == 0);
  struct stat st;
  CHECK(fstat(fd, &st) == 0 && st.st_size == pg);
  CHECK(shm_unlink(name) == 0);
  /* Unlinked: the name is gone, the mapping still works. */
  errno = 0;
  CHECK(shm_open(name, O_RDWR, 0) == -1 && errno == ENOENT);
  ((char *)sem)[600] = 1;
  CHECK(munmap(sem, pg) == 0);
  close(fd);
  puts("a process-shared semaphore in /dev/shm wakes another process");
}

static void send_fd(int s, int fd) {
  char b = 'x';
  struct iovec io = {&b, 1};
  char c[CMSG_SPACE(sizeof(int))];
  struct msghdr m = {0};
  m.msg_iov = &io;
  m.msg_iovlen = 1;
  m.msg_control = c;
  m.msg_controllen = sizeof c;
  struct cmsghdr *h = CMSG_FIRSTHDR(&m);
  h->cmsg_level = SOL_SOCKET;
  h->cmsg_type = SCM_RIGHTS;
  h->cmsg_len = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(h), &fd, sizeof fd);
  CHECK(sendmsg(s, &m, 0) == 1);
}

static int recv_fd(int s) {
  char b;
  struct iovec io = {&b, 1};
  char c[CMSG_SPACE(sizeof(int))];
  struct msghdr m = {0};
  m.msg_iov = &io;
  m.msg_iovlen = 1;
  m.msg_control = c;
  m.msg_controllen = sizeof c;
  if (recvmsg(s, &m, 0) != 1 || !CMSG_FIRSTHDR(&m)) return -1;
  int fd;
  memcpy(&fd, CMSG_DATA(CMSG_FIRSTHDR(&m)), sizeof fd);
  return fd;
}

/* Python's multiprocessing hands an unlinked /dev/shm file to a worker over
 * a socket: the receiver may be blocked in recvmsg when it arrives, or the
 * sender may have closed its own descriptor while it was in flight. Either
 * way the worker maps the same memory. */
static void passed_fd(void) {
  for (int late = 0; late < 2; late++) {
    int sv[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    pid_t pid = fork();
    if (pid == 0) {
      close(sv[0]);
      if (late) usleep(200000); /* the sender has closed its copy by now */
      int fd = recv_fd(sv[1]);
      if (fd < 0) _exit(1);
      char *m = mmap(0, pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (m == MAP_FAILED) _exit(2);
      strcpy(m, "worker");
      _exit(0);
    }
    close(sv[1]);
    char tmpl[] = "/dev/shm/pym-XXXXXX";
    int fd = mkstemp(tmpl);
    CHECK(fd >= 0 && unlink(tmpl) == 0 && ftruncate(fd, pg) == 0);
    char *m = mmap(0, pg, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    CHECK(m != MAP_FAILED);
    if (!late) usleep(200000); /* the receiver is blocked in recvmsg by now */
    send_fd(sv[0], fd);
    close(fd);
    int status;
    CHECK(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    CHECK(strcmp(m, "worker") == 0);
    munmap(m, pg);
    close(sv[0]);
  }
  puts("an unlinked shared file passed over a socket is the same memory");
}

int main(void) {
  pg = sysconf(_SC_PAGESIZE);
  sysv();
  shared_anon();
  posix();
  passed_fd();
  puts("PASS");
  return 0;
}
