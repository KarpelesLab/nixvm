/* A process's threads share one address space: the program break, and every
 * mapping change. A thread mustn't see a stale break (and grow the heap over
 * memory another thread's malloc handed out), nor keep using a page another
 * thread unmapped or replaced while it slept. A shared file mapping is the
 * process's too: one thread maps a file and stores into it, another unmaps
 * it (Go's linker on Windows, through Wine), and the stores are the file's.
 * And threads first touching the same fresh pages at once each get to read
 * them, however their faults interleave (Go's garbage collector, on several
 * CPUs: the slower fault found the page already there and was a SIGSEGV). */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
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

static char *shared_map;
static size_t shared_len;
static int mapped, reused;

/* Map the file and fill it, then stay until the main thread has unmapped
 * it and put fresh memory there: this thread's exit mustn't write that
 * memory over the file either. */
static void *map_and_fill(void *arg) {
  int fd = *(int *)arg;
  shared_map = mmap(0, shared_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (shared_map != MAP_FAILED)
    for (size_t i = 0; i < shared_len; i++) shared_map[i] = (char)(i % 251);
  pthread_mutex_lock(&mu);
  mapped = 1;
  pthread_cond_broadcast(&cv);
  while (!reused) pthread_cond_wait(&cv, &mu);
  pthread_mutex_unlock(&mu);
  return 0;
}

#define TOUCHERS 3
#define FRESH_PAGES 256
static char *volatile fresh;
static atomic_int fresh_round, fresh_done;

static void *toucher(void *arg) {
  long id = (long)arg;
  int seen = 0;
  unsigned sum = 0;
  for (;;) {
    while (atomic_load(&fresh_round) == seen) ;
    seen = atomic_load(&fresh_round);
    if (seen < 0) break;
    for (long i = 0; i < FRESH_PAGES; i++) sum += fresh[i * 4096 + id * 8];
    atomic_fetch_add(&fresh_done, 1);
  }
  return (void *)(uintptr_t)sum;
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

  char tmpl[] = "/tmp/threads_mm.XXXXXX";
  int fd = mkstemp(tmpl);
  CHECK(fd >= 0);
  unlink(tmpl);
  shared_len = 64 * pg;
  CHECK(ftruncate(fd, shared_len) == 0);
  CHECK(pthread_create(&t, 0, map_and_fill, &fd) == 0);
  pthread_mutex_lock(&mu);
  while (!mapped) pthread_cond_wait(&cv, &mu);
  pthread_mutex_unlock(&mu);
  CHECK(shared_map != MAP_FAILED);
  CHECK(munmap(shared_map, shared_len) == 0);
  CHECK(mmap(shared_map, shared_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == shared_map);
  pthread_mutex_lock(&mu);
  reused = 1;
  pthread_cond_broadcast(&cv);
  pthread_mutex_unlock(&mu);
  CHECK(pthread_join(t, 0) == 0);
  char *back = malloc(shared_len);
  CHECK(pread(fd, back, shared_len, 0) == (ssize_t)shared_len);
  size_t wrong = 0;
  for (size_t i = 0; i < shared_len; i++) wrong += back[i] != (char)(i % 251);
  if (wrong) printf("%zu of %zu bytes didn't reach the file\n", wrong, shared_len);
  CHECK(wrong == 0);
  close(fd);
  puts("a shared file mapping one thread filled and another unmapped kept its stores");

  pthread_t tt[TOUCHERS];
  for (long i = 0; i < TOUCHERS; i++) CHECK(pthread_create(&tt[i], 0, toucher, (void *)i) == 0);
  for (int round = 1; round <= 20; round++) {
    fresh = mmap(0, FRESH_PAGES * 4096L, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(fresh != MAP_FAILED);
    atomic_store(&fresh_done, 0);
    atomic_store(&fresh_round, round);
    while (atomic_load(&fresh_done) != TOUCHERS) ;
    CHECK(munmap(fresh, FRESH_PAGES * 4096L) == 0);
  }
  atomic_store(&fresh_round, -1);
  for (int i = 0; i < TOUCHERS; i++) CHECK(pthread_join(tt[i], 0) == 0);
  puts("threads touching the same fresh pages at once all read them");
  puts("PASS");
  return 0;
}
