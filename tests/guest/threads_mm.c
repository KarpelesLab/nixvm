/* A process's threads share one address space: the program break, and every
 * mapping change. A thread mustn't see a stale break (and grow the heap over
 * memory another thread's malloc handed out), nor keep using a page another
 * thread unmapped or replaced while it slept. */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

static char *main_heap;
static char *thread_brk_before, *thread_brk_after;
static pthread_mutex_t gmu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gcv = PTHREAD_COND_INITIALIZER;
static int go;

static void *grow(void *a) {
  (void)a;
  /* Started before the main thread grows the heap: a copy of the break taken
   * at clone() would be stale by the time this runs. */
  pthread_mutex_lock(&gmu);
  while (!go) pthread_cond_wait(&gcv, &gmu);
  pthread_mutex_unlock(&gmu);
  thread_brk_before = sbrk(0);
  char *p = sbrk(65536);
  if (p != (void *)-1) memset(p, 'T', 65536);
  thread_brk_after = sbrk(0);
  return p;
}

static volatile char *page;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
static int stage;
static char seen_before, seen_after;

static void *reader(void *a) {
  (void)a;
  seen_before = page[0]; /* the page is now in this thread's TLB */
  pthread_mutex_lock(&mu);
  stage = 1;
  pthread_cond_broadcast(&cv);
  while (stage != 2) pthread_cond_wait(&cv, &mu);
  pthread_mutex_unlock(&mu);
  seen_after = page[0];
  return 0;
}

int main(void) {
  /* The break: grown by the main thread, then by another thread. */
  pthread_t t;
  void *tp;
  CHECK(pthread_create(&t, 0, grow, 0) == 0);
  char *start = sbrk(0);
  main_heap = sbrk(65536);
  CHECK(main_heap != (void *)-1);
  memset(main_heap, 'M', 65536);
  pthread_mutex_lock(&gmu);
  go = 1;
  pthread_cond_broadcast(&gcv);
  pthread_mutex_unlock(&gmu);
  CHECK(pthread_join(t, &tp) == 0);
  printf("thread saw the break at +%ld, moved it to +%ld\n", (long)(thread_brk_before - start),
         (long)(thread_brk_after - start));
  CHECK(thread_brk_before == main_heap + 65536);
  CHECK(tp == main_heap + 65536);
  CHECK(sbrk(0) == main_heap + 131072);
  for (int i = 0; i < 65536; i++) CHECK(main_heap[i] == 'M');
  puts("threads share one program break");

  /* A page replaced while another thread that had used it was asleep. */
  long pg = sysconf(_SC_PAGESIZE);
  char *p = mmap(0, pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  CHECK(p != MAP_FAILED);
  p[0] = 'A';
  page = p;
  CHECK(pthread_create(&t, 0, reader, 0) == 0);
  pthread_mutex_lock(&mu);
  while (stage != 1) pthread_cond_wait(&cv, &mu);
  /* Free the page, let other memory take its frame (and fill it), then map
   * a new page at the same address: a stale translation in the other thread
   * would show the frame's new owner's bytes, not this page's. */
  CHECK(munmap(p, pg) == 0);
  for (int i = 0; i < 64; i++) {
    char *q = mmap(0, pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(q != MAP_FAILED);
    q[0] = 'X';
  }
  CHECK(mmap(p, pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == p);
  p[0] = 'B';
  stage = 2;
  pthread_cond_broadcast(&cv);
  pthread_mutex_unlock(&mu);
  CHECK(pthread_join(t, 0) == 0);
  printf("the other thread read %c, then %c after the page was replaced\n", seen_before, seen_after);
  CHECK(seen_before == 'A' && seen_after == 'B');
  puts("a page replaced while a thread slept is the new page there");
  puts("PASS");
  return 0;
}
