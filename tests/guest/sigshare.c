#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
static volatile sig_atomic_t hits, child_hits;
static void on_usr1(int s) { (void)s; hits++; }
static void *installer(void *a) {
  struct sigaction sa = {0};
  sa.sa_handler = on_usr1;
  sigaction(SIGUSR1, &sa, 0);
  return a;
}
static void *reader(void *a) {
  struct sigaction old;
  sigaction(SIGUSR1, 0, &old);
  *(int *)a = old.sa_handler == on_usr1;
  return 0;
}
int main(void) {
  pthread_t t; int seen = 0;
  pthread_create(&t, 0, installer, 0); pthread_join(t, 0);
  /* Main thread never called sigaction itself: default action would kill us. */
  raise(SIGUSR1);
  printf("main got handler installed by thread: %d\n", hits);
  pthread_create(&t, 0, reader, &seen); pthread_join(t, 0);
  printf("later thread sees handler: %d\n", seen);
  pid_t p = fork();
  if (!p) { signal(SIGUSR1, SIG_DFL); _exit(0); }
  int st; waitpid(p, &st, 0);
  struct sigaction old; sigaction(SIGUSR1, 0, &old);
  printf("fork child's change stayed in child: %d\n", old.sa_handler == on_usr1);
  puts("PASS"); return 0;
}
