/* What a newly created file, directory or socket name looks like: the umask
 * comes off the requested mode (a directory keeps the mode it was made with,
 * 0700 included — PostgreSQL refuses a data directory that's group- or
 * world-accessible), the creator owns it (its group from a set-group-ID
 * directory), and binding an AF_UNIX socket makes a socket node that chmod
 * can change, open() refuses and a second bind finds in use. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

static char dir[64];

static unsigned mode_of(const char *p) {
  struct stat st;
  CHECK(lstat(p, &st) == 0);
  return st.st_mode;
}

static void path(char *out, const char *name) { snprintf(out, 128, "%s/%s", dir, name); }

static void modes(void) {
  char p[128];
  umask(022);
  path(p, "f");
  int fd = open(p, O_CREAT | O_RDWR, 0666);
  CHECK(fd >= 0);
  close(fd);
  CHECK((mode_of(p) & 07777) == 0644);
  path(p, "d700");
  CHECK(mkdir(p, 0700) == 0);
  CHECK(mode_of(p) == (S_IFDIR | 0700));
  path(p, "d777");
  CHECK(mkdir(p, 0777) == 0);
  CHECK(mode_of(p) == (S_IFDIR | 0755));
  CHECK(chmod(p, 0750) == 0);
  CHECK(mode_of(p) == (S_IFDIR | 0750));
  umask(077);
  path(p, "g");
  fd = open(p, O_CREAT | O_RDWR, 0666);
  CHECK(fd >= 0);
  close(fd);
  CHECK((mode_of(p) & 07777) == 0600);
  path(p, "fifo");
  CHECK(mkfifo(p, 0666) == 0);
  CHECK(mode_of(p) == (S_IFIFO | 0600));
  /* A child inherits the umask; its change stays its own. */
  pid_t pid = fork();
  if (pid == 0) {
    _exit(umask(0) == 077 ? 0 : 1);
  }
  int status;
  CHECK(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0);
  CHECK(umask(022) == 077);
  puts("the umask comes off new files, directories and FIFOs");
}

static void sockets(void) {
  char p[128];
  path(p, "sock");
  struct sockaddr_un a = {.sun_family = AF_UNIX};
  strcpy(a.sun_path, p);
  int s = socket(AF_UNIX, SOCK_STREAM, 0);
  CHECK(s >= 0);
  CHECK(bind(s, (struct sockaddr *)&a, sizeof a) == 0);
  CHECK(mode_of(p) == (S_IFSOCK | 0755));
  CHECK(chmod(p, 0777) == 0);
  CHECK(mode_of(p) == (S_IFSOCK | 0777));
  CHECK(listen(s, 4) == 0);
  errno = 0;
  CHECK(open(p, O_RDWR) == -1 && errno == ENXIO);
  int t = socket(AF_UNIX, SOCK_STREAM, 0);
  errno = 0;
  CHECK(bind(t, (struct sockaddr *)&a, sizeof a) == -1 && errno == EADDRINUSE);
  int c = socket(AF_UNIX, SOCK_STREAM, 0);
  CHECK(connect(c, (struct sockaddr *)&a, sizeof a) == 0);
  close(c);
  /* The name outlives the socket, and must be unlinked before binding again. */
  close(s);
  CHECK(access(p, F_OK) == 0);
  errno = 0;
  CHECK(bind(t, (struct sockaddr *)&a, sizeof a) == -1 && errno == EADDRINUSE);
  CHECK(unlink(p) == 0);
  CHECK(bind(t, (struct sockaddr *)&a, sizeof a) == 0);
  close(t);
  CHECK(unlink(p) == 0);
  /* A failed bind leaves no name behind. */
  path(p, "nodir/sock");
  strcpy(a.sun_path, p);
  s = socket(AF_UNIX, SOCK_STREAM, 0);
  errno = 0;
  CHECK(bind(s, (struct sockaddr *)&a, sizeof a) == -1 && errno == ENOENT);
  close(s);
  puts("a bound socket has a name in the filesystem");
}

static void owners(void) {
  if (geteuid() != 0) {
    puts("not root: ownership not checked");
    return;
  }
  char p[128], sg[128];
  path(sg, "sgid");
  CHECK(mkdir(sg, 0777) == 0);
  CHECK(chown(sg, 0, 4321) == 0 && chmod(sg, 02777) == 0);
  CHECK(chmod(dir, 0777) == 0);
  pid_t pid = fork();
  if (pid == 0) {
    if (setgid(1234) || setuid(1234)) _exit(2);
    umask(0);
    path(p, "mine");
    int fd = open(p, O_CREAT | O_RDWR, 0640);
    if (fd < 0) _exit(3);
    struct stat st;
    if (fstat(fd, &st) || st.st_uid != 1234 || st.st_gid != 1234) _exit(4);
    snprintf(p, sizeof p, "%s/sub", sg);
    if (mkdir(p, 0700) || stat(p, &st)) _exit(5);
    /* A set-group-ID directory's group, and its bit, pass to a subdirectory. */
    if (st.st_uid != 1234 || st.st_gid != 4321 || (st.st_mode & 07777) != 02700) _exit(6);
    snprintf(p, sizeof p, "%s/file", sg);
    int f2 = open(p, O_CREAT | O_RDWR, 0644);
    if (f2 < 0 || fstat(f2, &st) || st.st_gid != 4321 || (st.st_mode & 07777) != 0644) _exit(7);
    _exit(0);
  }
  int status;
  CHECK(waitpid(pid, &status, 0) == pid && WIFEXITED(status));
  if (WEXITSTATUS(status)) printf("FAIL: child step %d\n", WEXITSTATUS(status));
  CHECK(WEXITSTATUS(status) == 0);
  puts("the creator owns what it makes");
}

int main(void) {
  strcpy(dir, "/tmp/creationXXXXXX");
  CHECK(mkdtemp(dir));
  modes();
  sockets();
  owners();
  puts("PASS");
  return 0;
}
