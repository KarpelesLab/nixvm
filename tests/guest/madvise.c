/* madvise(MADV_DONTNEED) on private anonymous memory: the pages read zeros
 * afterwards, whatever their protection was — runtimes discard PROT_NONE
 * pages and later make them writable again, expecting fresh zeros. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

static int all_zero(const char *p, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (p[i]) return 0;
  return 1;
}

int main(void) {
  long pg = sysconf(_SC_PAGESIZE);
  size_t n = 4 * pg;
  char *p = mmap(0, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  CHECK(p != MAP_FAILED);
  memset(p, 0x5a, n);
  CHECK(madvise(p, n, MADV_DONTNEED) == 0);
  CHECK(all_zero(p, n));
  puts("writable pages read zeros after MADV_DONTNEED");

  memset(p, 0x5a, n);
  CHECK(mprotect(p, n, PROT_NONE) == 0);
  CHECK(madvise(p, n, MADV_DONTNEED) == 0);
  CHECK(mprotect(p, n, PROT_READ | PROT_WRITE) == 0);
  CHECK(all_zero(p, n));
  puts("PROT_NONE pages discarded, then made writable, read zeros");

  memset(p, 0x5a, n);
  CHECK(mprotect(p, n, PROT_READ) == 0);
  CHECK(madvise(p + pg, pg, MADV_DONTNEED) == 0);
  CHECK(p[0] == 0x5a && all_zero(p + pg, pg) && p[2 * pg] == 0x5a);
  puts("only the advised read-only page is discarded");

  char *s = mmap(0, pg, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  CHECK(s != MAP_FAILED);
  memset(s, 0x33, pg);
  CHECK(madvise(s, pg, MADV_DONTNEED) == 0);
  CHECK(s[0] == 0x33 && s[pg - 1] == 0x33);
  puts("a shared page keeps its contents");
  puts("PASS");
  return 0;
}
