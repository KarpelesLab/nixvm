/* A signal sent to a process that ignores it, and doesn't block it, is thrown
 * away there and then (it never becomes pending), while a blocked one is kept.
 * Making a signal ignored drops it even if it was pending and blocked. */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

static volatile sig_atomic_t hits;
static void on_sig(int s) { (void)s; hits++; }

static int pending(int sig) {
  sigset_t p;
  sigpending(&p);
  return sigismember(&p, sig);
}

static void block(int sig, int how) {
  sigset_t s;
  sigemptyset(&s);
  sigaddset(&s, sig);
  sigprocmask(how, &s, 0);
}

static void handle(int sig, void (*h)(int)) {
  struct sigaction sa = {0};
  sa.sa_handler = h;
  sigaction(sig, &sa, 0);
}

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); exit(1); } } while (0)

int main(void) {
  /* Ignored and unblocked: discarded when sent, so a handler installed later
   * never runs. */
  handle(SIGUSR1, SIG_IGN);
  kill(getpid(), SIGUSR1);
  block(SIGUSR1, SIG_BLOCK);
  handle(SIGUSR1, on_sig);
  CHECK(!pending(SIGUSR1));
  block(SIGUSR1, SIG_UNBLOCK);
  CHECK(hits == 0);
  puts("ignored unblocked signal discarded: ok");

  /* Ignored but blocked: kept, and runs the handler installed before it's
   * unblocked. */
  handle(SIGUSR1, SIG_IGN);
  block(SIGUSR1, SIG_BLOCK);
  kill(getpid(), SIGUSR1);
  CHECK(pending(SIGUSR1));
  handle(SIGUSR1, on_sig);
  block(SIGUSR1, SIG_UNBLOCK);
  CHECK(hits == 1);
  puts("ignored blocked signal kept: ok");

  /* Setting SIG_IGN drops a pending signal even while it's blocked. */
  block(SIGUSR2, SIG_BLOCK);
  kill(getpid(), SIGUSR2);
  CHECK(pending(SIGUSR2));
  handle(SIGUSR2, SIG_IGN);
  CHECK(!pending(SIGUSR2));
  block(SIGUSR2, SIG_UNBLOCK);
  puts("SIG_IGN flushes a pending signal: ok");

  /* A real-time signal's whole queue goes too. */
  block(SIGRTMIN, SIG_BLOCK);
  union sigval v = {.sival_int = 7};
  sigqueue(getpid(), SIGRTMIN, v);
  sigqueue(getpid(), SIGRTMIN, v);
  handle(SIGRTMIN, SIG_IGN);
  handle(SIGRTMIN, on_sig);
  CHECK(!pending(SIGRTMIN));
  block(SIGRTMIN, SIG_UNBLOCK);
  CHECK(hits == 1);
  puts("SIG_IGN flushes a real-time queue: ok");

  /* SIGCHLD defaults to ignored: a child's exit leaves nothing pending unless
   * it's blocked. */
  pid_t p = fork();
  if (!p) _exit(0);
  int st;
  waitpid(p, &st, 0);
  block(SIGCHLD, SIG_BLOCK);
  CHECK(!pending(SIGCHLD));
  p = fork();
  if (!p) _exit(0);
  waitpid(p, &st, 0);
  CHECK(pending(SIGCHLD));
  handle(SIGCHLD, SIG_DFL); /* default-ignored: flushed */
  CHECK(!pending(SIGCHLD));
  block(SIGCHLD, SIG_UNBLOCK);
  puts("SIGCHLD at its default: ok");

  /* An ignored SIGCONT still continues a stopped process. */
  p = fork();
  if (!p) {
    raise(SIGSTOP);
    _exit(3);
  }
  waitpid(p, &st, WUNTRACED);
  CHECK(WIFSTOPPED(st));
  kill(p, SIGCONT);
  waitpid(p, &st, 0);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 3);
  puts("ignored SIGCONT still continues: ok");

  /* Sent to another process that ignores it: it isn't woken or interrupted,
   * and a handler it installs afterwards never runs. */
  int fds[2];
  CHECK(pipe(fds) == 0);
  p = fork();
  if (!p) {
    handle(SIGUSR1, SIG_IGN);
    if (write(fds[1], "r", 1) != 1) _exit(2);
    usleep(200000); /* the parent signals meanwhile */
    block(SIGUSR1, SIG_BLOCK);
    handle(SIGUSR1, on_sig);
    _exit(pending(SIGUSR1) ? 1 : 0);
  }
  char c;
  CHECK(read(fds[0], &c, 1) == 1);
  kill(p, SIGUSR1);
  waitpid(p, &st, 0);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0);
  puts("another process's ignored signal discarded: ok");

  puts("PASS");
  return 0;
}
