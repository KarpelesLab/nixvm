/* Where a 64-bit mmap puts what it isn't given a fixed place for, as
 * arch/x86/kernel/sys_x86_64.c and mm/mmap.c have it: MAP_32BIT in the
 * 1-2 GiB window (bottom up, a hint below 2 GiB taken, ENOMEM once the
 * window is full); an address hint taken when the range is free and clear of
 * the stack's guard gap, rounded down to a page and up to mmap_min_addr,
 * ignored when something is there, when it would cross the 47-bit boundary
 * or end past TASK_SIZE; nothing above 47 bits unless a hint asks for it;
 * and MAP_FIXED and MAP_FIXED_NOREPLACE's checks in get_unmapped_area's
 * order: ENOMEM past TASK_SIZE, EINVAL unaligned, EPERM below mmap_min_addr,
 * then EEXIST over a mapping.
 *
 * MAP_32BIT and the 47-bit rule are x86-64's; the rest is checked on any
 * architecture.
 *
 * It passes on Linux too: `gcc -static -O1 -o mmap_place mmap_place.c`. */
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static int failures;
#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            printf("FAIL %s:%d: ", __func__, __LINE__);                     \
            printf(__VA_ARGS__);                                              \
            printf(" (errno %d)\n", errno);                                   \
            failures++;                                                       \
        }                                                                     \
    } while (0)

#define PG 4096UL
#define WINDOW_LO 0x40000000UL
#define WINDOW_HI 0x80000000UL
#define BIT47 (1UL << 47)

static void *map(void *at, size_t len, int flags)
{
    return mmap(at, len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | flags, -1, 0);
}

/* Whether anything is mapped in [p, p + len). */
static int mapped(uintptr_t p, size_t len)
{
    unsigned char v;
    for (uintptr_t a = p; a < p + len; a += PG)
        if (mincore((void *)a, PG, &v) == 0 || errno != ENOMEM)
            return 1;
    return 0;
}

static unsigned long min_addr(void)
{
    unsigned long v = 65536;
    FILE *f = fopen("/proc/sys/vm/mmap_min_addr", "r");
    if (f) {
        if (fscanf(f, "%lu", &v) != 1)
            v = 65536;
        fclose(f);
    }
    return v;
}

/* Whether the CPU has 5-level paging, which lets a hint above 47 bits have
 * the whole 56-bit address space searched. */
static int la57(void)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    char line[4096];
    int yes = 0;
    while (f && !yes && fgets(line, sizeof line, f))
        yes = strncmp(line, "flags", 5) == 0 && strstr(line, " la57") != NULL;
    if (f)
        fclose(f);
    return yes;
}

/* Whether this process may map below mmap_min_addr (CAP_SYS_RAWIO). */
static int rawio(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    unsigned long long eff = 0;
    while (f && fgets(line, sizeof line, f))
        if (sscanf(line, "CapEff: %llx", &eff) == 1)
            break;
    if (f)
        fclose(f);
    return (eff >> 17) & 1;
}

#ifdef __x86_64__
static void test_map_32bit(void)
{
    /* Each lands in the window, whole. */
    uintptr_t got[4];
    for (int i = 0; i < 4; i++) {
        void *p = map(NULL, 1 << 20, MAP_32BIT);
        got[i] = (uintptr_t)p;
        CHECK(p != MAP_FAILED && got[i] >= WINDOW_LO && got[i] + (1 << 20) <= WINDOW_HI, "MAP_32BIT: %p", p);
        for (int j = 0; j < i; j++)
            CHECK(got[i] + (1 << 20) <= got[j] || got[j] + (1 << 20) <= got[i], "MAP_32BIT overlap %#lx %#lx",
                  (unsigned long)got[i], (unsigned long)got[j]);
    }
    for (int i = 0; i < 4; i++)
        munmap((void *)got[i], 1 << 20);
    /* A free hint that ends by 2 GiB is taken, inside the window or below
     * it; one past it is not. */
    void *h = map((void *)0x50000000, PG, MAP_32BIT);
    CHECK(h == (void *)0x50000000, "MAP_32BIT hint in the window: %p", h);
    munmap(h, PG);
    h = map((void *)0x30000000, PG, MAP_32BIT);
    CHECK(h == (void *)0x30000000, "MAP_32BIT hint below the window: %p", h);
    munmap(h, PG);
    h = map((void *)0x7ffff000, 2 * PG, MAP_32BIT);
    CHECK(h != MAP_FAILED && h != (void *)0x7ffff000 && (uintptr_t)h + 2 * PG <= WINDOW_HI,
          "MAP_32BIT hint ending past 2 GiB: %p", h);
    munmap(h, 2 * PG);
    h = map((void *)0x500000000, PG, MAP_32BIT);
    CHECK(h != MAP_FAILED && (uintptr_t)h >= WINDOW_LO && (uintptr_t)h < WINDOW_HI,
          "MAP_32BIT hint above 4 GiB: %p", h);
    munmap(h, PG);
    /* Too big for the window at all. */
    errno = 0;
    CHECK(map(NULL, 3UL << 30, MAP_32BIT) == MAP_FAILED && errno == ENOMEM, "MAP_32BIT of 3 GiB");
    /* Fill the window: then there's no room, and no falling back elsewhere. */
    uintptr_t chunks[40];
    int n = 0;
    for (; n < 40; n++) {
        void *p = map(NULL, 64 << 20, MAP_32BIT);
        if (p == MAP_FAILED) {
            CHECK(errno == ENOMEM, "MAP_32BIT when full: errno %d", errno);
            break;
        }
        chunks[n] = (uintptr_t)p;
        CHECK(chunks[n] >= WINDOW_LO && chunks[n] + (64 << 20) <= WINDOW_HI, "MAP_32BIT chunk %d at %#lx", n,
              (unsigned long)chunks[n]);
    }
    CHECK(n >= 10 && n < 17, "%d 64 MiB MAP_32BIT chunks fit in 1 GiB", n);
    /* What's left is smaller than 64 MiB, but a page still fits. */
    void *small = map(NULL, PG, MAP_32BIT);
    CHECK(small == MAP_FAILED || ((uintptr_t)small >= WINDOW_LO && (uintptr_t)small < WINDOW_HI),
          "a page MAP_32BIT when nearly full: %p", small);
    if (small != MAP_FAILED)
        munmap(small, PG);
    for (int i = 0; i < n; i++)
        munmap((void *)chunks[i], 64 << 20);
    /* A MAP_32BIT file mapping too. */
    FILE *f = fopen("/proc/self/exe", "r");
    if (f) {
        void *p = mmap(NULL, PG, PROT_READ, MAP_PRIVATE | MAP_32BIT, fileno(f), 0);
        CHECK(p != MAP_FAILED && (uintptr_t)p >= WINDOW_LO && (uintptr_t)p < WINDOW_HI, "file MAP_32BIT: %p", p);
        CHECK(p != MAP_FAILED && memcmp(p, "\177ELF", 4) == 0, "file MAP_32BIT reads the file");
        munmap(p, PG);
        fclose(f);
    }
}

#endif

static void test_hints(void)
{
    /* A free hint is taken, rounded down to its page. */
    uintptr_t at = 0x5000000000UL;
    CHECK(!mapped(at, 4 * PG), "%#lx is free", (unsigned long)at);
    void *p = map((void *)at, 2 * PG, 0);
    CHECK(p == (void *)at, "a free hint: %p", p);
    void *q = map((void *)(at + 2 * PG + 123), PG, 0);
    CHECK(q == (void *)(at + 2 * PG), "an unaligned hint: %p", q);
    /* Something there: placed elsewhere, clear of it. */
    void *r = map((void *)(at + PG), PG, 0);
    CHECK(r != MAP_FAILED && r != (void *)(at + PG) &&
              ((uintptr_t)r + PG <= at || (uintptr_t)r >= at + 3 * PG),
          "a hint over a mapping: %p", r);
    munmap(r, PG);
    /* Usable, as any mapping is. */
    CHECK(mprotect(p, 2 * PG, PROT_READ | PROT_WRITE) == 0, "mprotect");
    ((volatile char *)p)[PG] = 7;
    CHECK(((volatile char *)p)[PG] == 7, "write through a hinted mapping");
    munmap(p, 2 * PG);
    munmap(q, PG);
    /* A large reservation at a hint, as a JIT makes. */
    uintptr_t big = 0x6000000000UL;
    p = map((void *)big, 4UL << 30, 0);
    CHECK(p == (void *)big, "a 4 GiB reservation at a hint: %p", p);
    munmap(p, 4UL << 30);
    /* Low in the address space, and below mmap_min_addr: raised to it. */
    p = map((void *)0x10000000, PG, 0);
    CHECK(p == (void *)0x10000000, "a hint at 256 MiB: %p", p);
    munmap(p, PG);
    unsigned long lo = min_addr();
    if (!mapped(lo, PG)) {
        p = map((void *)PG, PG, 0);
        CHECK(p == (void *)lo, "a hint below mmap_min_addr: %p, mmap_min_addr %#lx", p, lo);
        munmap(p, PG);
    }
    /* In the stack's guard gap: not taken. */
    uintptr_t stack = 0;
    FILE *f = fopen("/proc/self/maps", "r");
    char line[512];
    while (f && fgets(line, sizeof line, f))
        if (strstr(line, "[stack]"))
            stack = strtoul(line, NULL, 16);
    if (f)
        fclose(f);
    CHECK(stack != 0, "a [stack] in /proc/self/maps");
    if (stack) {
        uintptr_t near = stack - 64 * PG;
        p = map((void *)near, PG, 0);
        CHECK(p != MAP_FAILED && p != (void *)near, "a hint in the stack's guard gap: %p (stack %#lx)", p,
              (unsigned long)stack);
        munmap(p, PG);
    }
#ifdef __x86_64__
    /* Across the 47-bit boundary: not taken, and placed below it. */
    p = map((void *)(BIT47 - PG), 2 * PG, 0);
    CHECK(p != MAP_FAILED && (uintptr_t)p + 2 * PG <= BIT47, "a hint across 47 bits: %p", p);
    munmap(p, 2 * PG);
    /* Above 47 bits: only there if the hint asks (5-level paging), never
     * otherwise. */
    p = map((void *)(BIT47 + (1UL << 40)), PG, 0);
    CHECK(p != MAP_FAILED && (p == (void *)(BIT47 + (1UL << 40)) || (uintptr_t)p < BIT47),
          "a hint above 47 bits: %p", p);
    munmap(p, PG);
    p = map(NULL, PG, 0);
    CHECK(p != MAP_FAILED && (uintptr_t)p < BIT47, "no hint: %p", p);
    munmap(p, PG);
#endif
    /* Past the end of the address space: placed below 47 bits instead
     * (with 5-level paging, the hint being above them, anywhere). */
    p = map((void *)(1UL << 62), 2 * PG, 0);
#ifdef __x86_64__
    CHECK(p != MAP_FAILED && ((uintptr_t)p < BIT47 || la57()), "a hint past the end: %p", p);
#else
    CHECK(p != MAP_FAILED && p != (void *)(1UL << 62), "a hint past the end: %p", p);
#endif
    munmap(p, 2 * PG);
}

static void test_fixed(void)
{
    uintptr_t at = 0x5100000000UL;
    void *p = map((void *)at, 2 * PG, MAP_FIXED_NOREPLACE);
    CHECK(p == (void *)at, "MAP_FIXED_NOREPLACE: %p", p);
    errno = 0;
    CHECK(map((void *)(at + PG), 2 * PG, MAP_FIXED_NOREPLACE) == MAP_FAILED && errno == EEXIST,
          "MAP_FIXED_NOREPLACE over a mapping");
    CHECK(mapped(at + PG, PG) && !mapped(at + 2 * PG, PG), "a failed MAP_FIXED_NOREPLACE changes nothing");
    /* Unaligned: EINVAL, before EEXIST. */
    errno = 0;
    CHECK(map((void *)(at + 1), PG, MAP_FIXED_NOREPLACE) == MAP_FAILED && errno == EINVAL,
          "MAP_FIXED_NOREPLACE unaligned over a mapping");
    errno = 0;
    CHECK(map((void *)(at + 2 * PG + 1), PG, MAP_FIXED) == MAP_FAILED && errno == EINVAL, "MAP_FIXED unaligned");
    /* MAP_FIXED replaces. */
    void *r = mmap((void *)at, PG, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    CHECK(r == (void *)at, "MAP_FIXED over a mapping: %p", r);
    munmap(p, 2 * PG);
    /* Past TASK_SIZE: ENOMEM, even unaligned. */
    errno = 0;
    CHECK(map((void *)(1UL << 62), 2 * PG, MAP_FIXED) == MAP_FAILED && errno == ENOMEM, "MAP_FIXED past the end");
    errno = 0;
    CHECK(map((void *)((1UL << 62) + 1), 2 * PG, MAP_FIXED_NOREPLACE) == MAP_FAILED && errno == ENOMEM,
          "MAP_FIXED_NOREPLACE past the end");
    /* Below mmap_min_addr, without CAP_SYS_RAWIO: EPERM, page 0 included
     * (MAP_FIXED_NOREPLACE is a fixed mapping too). */
    if (!rawio() && min_addr() >= PG) {
        errno = 0;
        CHECK(map(NULL, PG, MAP_FIXED) == MAP_FAILED && errno == EPERM, "MAP_FIXED at 0");
        errno = 0;
        CHECK(map(NULL, PG, MAP_FIXED_NOREPLACE) == MAP_FAILED && errno == EPERM, "MAP_FIXED_NOREPLACE at 0");
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
#ifdef __x86_64__
    test_map_32bit();
#endif
    test_hints();
    test_fixed();
    if (failures) {
        printf("%d failures\n", failures);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
