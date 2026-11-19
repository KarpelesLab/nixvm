/* Clock ids, as Linux's posix-timers.c, posix-cpu-timers.c, alarmtimer.c and
 * timerfd.c check them: which ids exist (EINVAL for the others), the CPU
 * clocks clock_getcpuclockid and pthread_getcpuclockid make — another
 * process's or thread's CPU time, and EINVAL once it is gone or never was —
 * and what each call refuses: clock_nanosleep on CLOCK_THREAD_CPUTIME_ID or
 * the raw and coarse clocks (EOPNOTSUPP), on one's own thread's CPU clock
 * (EINVAL), timers on clocks that have none, timerfd's five clocks, the
 * alarm clocks' CAP_WAKE_ALARM (EPERM), clock_settime and clock_adjtime. The
 * 64-bit vDSO hands every clock it doesn't compute to the system call and
 * gives back what it said. Also: CLOCK_TAI against adjtimex's TAI offset, a
 * sleep on a CPU clock waiting for CPU time, and an absolute timerfd expiry
 * on CLOCK_MONOTONIC.
 *
 * It passes on Linux too: `gcc -static -O1 -pthread -o clocks clocks.c`. */
#define _GNU_SOURCE
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/syscall.h>
#include <sys/timerfd.h>
#include <sys/timex.h>
#include <sys/wait.h>
#include <time.h>
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

#define CPUCLOCK_PROF 0
#define CPUCLOCK_VIRT 1
#define CPUCLOCK_SCHED 2
#define PROCESS_CLOCK(pid, which) ((clockid_t)((~(unsigned)(pid)) << 3) | (which))
#define THREAD_CLOCK(tid, which) PROCESS_CLOCK(tid, (which) | 4)
#define FD_CLOCK(fd) ((clockid_t)((~(unsigned)(fd)) << 3) | 3)

#ifndef CLOCK_TAI
#define CLOCK_TAI 11
#endif

/* A system call's result as the kernel gives it: 0 or -errno. */
static long raw(long r) { return r == -1 ? -errno : r; }

static long gettime(clockid_t c, struct timespec *t) { return raw(syscall(SYS_clock_gettime, c, t)); }
static long getres(clockid_t c, struct timespec *t) { return raw(syscall(SYS_clock_getres, c, t)); }
static long nsleep(clockid_t c, int flags, const struct timespec *t)
{
    return raw(syscall(SYS_clock_nanosleep, c, flags, t, NULL));
}
static long settime(clockid_t c, const struct timespec *t) { return raw(syscall(SYS_clock_settime, c, t)); }
static long mktimer(clockid_t c, timer_t *id)
{
    struct sigevent ev = {.sigev_notify = SIGEV_NONE};
    long r = raw(syscall(SYS_timer_create, c, &ev, id));
    return r;
}
static long rmtimer(timer_t id) { return raw(syscall(SYS_timer_delete, id)); }

static long long ns(const struct timespec *t) { return t->tv_sec * 1000000000LL + t->tv_nsec; }
static long long now(clockid_t c)
{
    struct timespec t;
    if (clock_gettime(c, &t) != 0)
        return -1;
    return ns(&t);
}

/* Whether the machine's alarm clocks work (an RTC): Linux's say EINVAL to
 * clock_getres without one. */
static int have_alarm;

/* Whether this process has CAP_WAKE_ALARM over the machine (Linux's
 * capable(): effective, and in the initial user namespace). */
static int wake_alarm;

static int initial_user_ns(void)
{
    FILE *f = fopen("/proc/self/uid_map", "r");
    unsigned long inner, outer, count;
    int n = f ? fscanf(f, "%lu %lu %lu", &inner, &outer, &count) : 0;
    if (f)
        fclose(f);
    return n == 3 && inner == 0 && outer == 0 && count == 4294967295UL;
}

static int has_wake_alarm(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    unsigned long long eff = 0;
    while (f && fgets(line, sizeof line, f))
        if (sscanf(line, "CapEff: %llx", &eff) == 1)
            break;
    if (f)
        fclose(f);
    return (eff >> 35) & 1 && initial_user_ns();
}

/* Burn `ms` of this thread's CPU time. */
static void burn(int ms)
{
    long long start = now(CLOCK_THREAD_CPUTIME_ID);
    while (now(CLOCK_THREAD_CPUTIME_ID) - start < ms * 1000000LL)
        ;
}

static void test_ids(void)
{
    for (int c = 0; c < 16; c++) {
        struct timespec t = {-1, -1}, r = {-1, -1};
        long g = gettime(c, &t), s = getres(c, &r);
        int exists = c <= 11 && c != 10;
        if (c == CLOCK_REALTIME_ALARM || c == CLOCK_BOOTTIME_ALARM) {
            CHECK(g == s && (g == 0 || g == -EINVAL), "alarm clock %d: gettime %ld getres %ld", c, g, s);
            continue;
        }
        if (exists) {
            CHECK(g == 0 && t.tv_nsec >= 0 && t.tv_nsec < 1000000000, "clock_gettime(%d) = %ld", c, g);
            CHECK(s == 0 && r.tv_sec == 0 && r.tv_nsec > 0, "clock_getres(%d) = %ld, %ld.%09ld", c, s,
                  (long)r.tv_sec, r.tv_nsec);
            CHECK(getres(c, NULL) == 0, "clock_getres(%d, NULL)", c);
        } else {
            CHECK(g == -EINVAL && s == -EINVAL, "clock %d: gettime %ld getres %ld, want EINVAL", c, g, s);
        }
    }
    struct timespec r;
    have_alarm = getres(CLOCK_REALTIME_ALARM, &r) == 0;
    /* The hrtimer clocks and the run-time CPU clocks resolve to the
     * nanosecond. */
    static const int fine[] = {CLOCK_REALTIME, CLOCK_MONOTONIC, CLOCK_BOOTTIME, CLOCK_TAI, CLOCK_MONOTONIC_RAW,
                               CLOCK_PROCESS_CPUTIME_ID, CLOCK_THREAD_CPUTIME_ID};
    for (unsigned i = 0; i < sizeof fine / sizeof fine[0]; i++)
        CHECK(getres(fine[i], &r) == 0 && r.tv_sec == 0 && r.tv_nsec == 1, "clock_getres(%d) = %ld", fine[i],
              r.tv_nsec);
    /* Monotonic time and boot time move together, wall time and TAI too. */
    long long m = now(CLOCK_MONOTONIC), b = now(CLOCK_BOOTTIME);
    CHECK(b >= m, "boottime %lld before monotonic %lld", b, m);
    struct timex tx = {0};
    int state = adjtimex(&tx);
    long long rt = now(CLOCK_REALTIME), tai = now(CLOCK_TAI);
    long long off = (tai - rt) / 1000000;
    CHECK(state >= 0 && off >= tx.tai * 1000LL - 50 && off <= tx.tai * 1000LL + 50,
          "TAI - REALTIME is %lld ms, adjtimex's tai %d s", off, tx.tai);

    /* Negative ids: CPU clocks by pid and the clock devices. */
    struct timespec t;
    CHECK(gettime(PROCESS_CLOCK(0, CPUCLOCK_SCHED), &t) == 0, "own process clock");
    CHECK(gettime(PROCESS_CLOCK(0, CPUCLOCK_PROF), &t) == 0, "own PROF clock");
    CHECK(gettime(PROCESS_CLOCK(0, CPUCLOCK_VIRT), &t) == 0, "own VIRT clock");
    CHECK(gettime(THREAD_CLOCK(0, CPUCLOCK_SCHED), &t) == 0, "own thread clock");
    CHECK(gettime(THREAD_CLOCK(gettid(), CPUCLOCK_SCHED), &t) == 0, "own thread clock by tid");
    CHECK(gettime(PROCESS_CLOCK(getpid(), CPUCLOCK_SCHED), &t) == 0, "own process clock by pid");
    CHECK(gettime(THREAD_CLOCK(0, 3), &t) == -EINVAL, "a CPU clock past SCHED"); /* -1 */
    CHECK(getres(THREAD_CLOCK(0, 3), &t) == -EINVAL, "getres of a CPU clock past SCHED");
    int fd = open("/dev/null", O_RDONLY);
    CHECK(gettime(FD_CLOCK(fd), &t) == -EINVAL, "a file that isn't a clock device");
    CHECK(getres(FD_CLOCK(fd), &t) == -EINVAL, "getres of a file that isn't a clock device");
    CHECK(gettime(FD_CLOCK(999), &t) == -EINVAL, "a clock device that isn't open");
    close(fd);
    clockid_t mine;
    CHECK(clock_getcpuclockid(getpid(), &mine) == 0 && mine == PROCESS_CLOCK(getpid(), CPUCLOCK_SCHED),
          "clock_getcpuclockid(self) = %d", mine);
    long long a = now(CLOCK_PROCESS_CPUTIME_ID), via = now(mine), z = now(CLOCK_PROCESS_CPUTIME_ID);
    CHECK(a <= via && via <= z, "the process clock by pid %lld, by id %lld..%lld", via, a, z);
}

static volatile int spin_stop;
static volatile pid_t spin_tid;

static void *spinner(void *arg)
{
    (void)arg;
    spin_tid = gettid();
    while (!spin_stop)
        ;
    return NULL;
}

static volatile int helper_done;
static long helper_get, helper_res;

static void *helper(void *arg)
{
    (void)arg;
    struct timespec t;
    /* By its own thread id, a thread names its process's clock to
     * clock_gettime, but nothing else does. */
    helper_get = gettime(PROCESS_CLOCK(gettid(), CPUCLOCK_SCHED), &t);
    helper_res = getres(PROCESS_CLOCK(gettid(), CPUCLOCK_SCHED), &t);
    helper_done = 1;
    return NULL;
}

/* Another thread's clock, and sleeping on CPU clocks. */
static void test_threads(void)
{
    pthread_t th;
    spin_stop = 0;
    spin_tid = 0;
    CHECK(pthread_create(&th, NULL, spinner, NULL) == 0, "pthread_create");
    while (!spin_tid)
        sched_yield();
    clockid_t tc;
    CHECK(pthread_getcpuclockid(th, &tc) == 0, "pthread_getcpuclockid");
    /* Sleep 20 ms of its CPU time, then 20 ms of the process's: both come
     * from the spinner. */
    struct timespec t20 = {0, 20000000};
    long long t0 = now(tc), p0 = now(CLOCK_PROCESS_CPUTIME_ID);
    CHECK(nsleep(tc, 0, &t20) == 0, "sleep on another thread's clock");
    long long t1 = now(tc);
    CHECK(t1 - t0 >= 20000000, "the thread ran %lld ns of the 20 ms slept on", t1 - t0);
    CHECK(nsleep(THREAD_CLOCK(spin_tid, CPUCLOCK_SCHED), 0, &t20) == 0, "sleep on a thread clock by tid");
    CHECK(nsleep(CLOCK_PROCESS_CPUTIME_ID, 0, &t20) == 0, "sleep on the process clock");
    long long p1 = now(CLOCK_PROCESS_CPUTIME_ID);
    CHECK(p1 - p0 >= 60000000, "the process ran %lld ns over the 60 ms slept on", p1 - p0);
    /* An absolute sleep on it, to a time already past, is over at once. */
    struct timespec past = {0, 1};
    CHECK(nsleep(CLOCK_PROCESS_CPUTIME_ID, TIMER_ABSTIME, &past) == 0, "an absolute CPU sleep in the past");
    /* A non-leader's tid isn't a process. */
    struct timespec t;
    CHECK(gettime(PROCESS_CLOCK(spin_tid, CPUCLOCK_SCHED), &t) == -EINVAL, "a process clock by a thread's tid");
    CHECK(gettime(THREAD_CLOCK(spin_tid, CPUCLOCK_SCHED), &t) == 0, "a thread clock by its tid");
    spin_stop = 1;
    pthread_join(th, NULL);

    pthread_t h;
    helper_done = 0;
    CHECK(pthread_create(&h, NULL, helper, NULL) == 0, "pthread_create");
    pthread_join(h, NULL);
    CHECK(helper_get == 0 && helper_res == -EINVAL, "own tid as a process clock: gettime %ld getres %ld",
          helper_get, helper_res);

    /* Never on one's own thread clock, and never on CLOCK_THREAD_CPUTIME_ID,
     * which has no sleep at all (checked before the time is). */
    CHECK(nsleep(THREAD_CLOCK(0, CPUCLOCK_SCHED), 0, &t20) == -EINVAL, "sleep on own thread clock");
    CHECK(nsleep(THREAD_CLOCK(gettid(), CPUCLOCK_SCHED), 0, &t20) == -EINVAL, "sleep on own thread clock by tid");
    CHECK(nsleep(CLOCK_THREAD_CPUTIME_ID, 0, &t20) == -EOPNOTSUPP, "sleep on CLOCK_THREAD_CPUTIME_ID");
    CHECK(nsleep(CLOCK_THREAD_CPUTIME_ID, 0, NULL) == -EOPNOTSUPP, "sleep on CLOCK_THREAD_CPUTIME_ID, no time");
}

/* Sleeps on the other clocks. */
static void test_sleeps(void)
{
    struct timespec t1ms = {0, 1000000}, bad = {0, 1000000000};
    static const int nosleep[] = {CLOCK_MONOTONIC_RAW, CLOCK_REALTIME_COARSE, CLOCK_MONOTONIC_COARSE};
    for (unsigned i = 0; i < 3; i++)
        CHECK(nsleep(nosleep[i], 0, &t1ms) == -EOPNOTSUPP, "sleep on clock %d", nosleep[i]);
    CHECK(nsleep(10, 0, &t1ms) == -EINVAL, "sleep on clock 10");
    CHECK(nsleep(12, 0, &t1ms) == -EINVAL, "sleep on clock 12");
    CHECK(nsleep(FD_CLOCK(0), 0, &t1ms) == -EOPNOTSUPP, "sleep on a clock device");
    CHECK(nsleep(CLOCK_MONOTONIC, 0, NULL) == -EFAULT, "sleep with no time");
    CHECK(nsleep(CLOCK_MONOTONIC, 0, &bad) == -EINVAL, "sleep with a bad time");
    static const int sleeps[] = {CLOCK_REALTIME, CLOCK_MONOTONIC, CLOCK_BOOTTIME, CLOCK_TAI};
    for (unsigned i = 0; i < 4; i++) {
        int c = sleeps[i];
        long long a = now(CLOCK_MONOTONIC);
        CHECK(nsleep(c, 0, &t1ms) == 0, "relative sleep on clock %d", c);
        /* 20 ms from now, on the clock itself. */
        struct timespec at;
        clock_gettime(c, &at);
        at.tv_nsec += 20000000;
        if (at.tv_nsec >= 1000000000) {
            at.tv_sec++;
            at.tv_nsec -= 1000000000;
        }
        long long b = now(CLOCK_MONOTONIC);
        CHECK(nsleep(c, TIMER_ABSTIME, &at) == 0, "absolute sleep on clock %d", c);
        long long d = now(CLOCK_MONOTONIC);
        CHECK(b - a >= 1000000, "clock %d: a 1 ms sleep took %lld ns", c, b - a);
        CHECK(d - b >= 19000000 && d - b < 2000000000LL, "clock %d: an absolute 20 ms sleep took %lld ns", c,
              d - b);
    }
    /* The alarm clocks: no flags but TIMER_ABSTIME, then CAP_WAKE_ALARM. */
    if (have_alarm) {
        CHECK(nsleep(CLOCK_REALTIME_ALARM, 2, &t1ms) == -EINVAL, "alarm sleep with a bad flag");
        long r = nsleep(CLOCK_BOOTTIME_ALARM, 0, &t1ms);
        CHECK(wake_alarm ? r == 0 : r == -EPERM, "alarm sleep (CAP_WAKE_ALARM %d): %ld", wake_alarm, r);
        CHECK(nsleep(CLOCK_BOOTTIME_ALARM, 0, &bad) == -EINVAL, "alarm sleep with a bad time");
    }
}

/* Another process's clock, alive, a zombie, and gone. */
static void test_process(void)
{
    int up[2], down[2];
    CHECK(pipe(up) == 0 && pipe(down) == 0, "pipe");
    pid_t pid = fork();
    if (pid == 0) {
        burn(30);
        char c = 1;
        write(up[1], &c, 1);
        read(down[0], &c, 1);
        if (c == 2) {
            /* Spin until killed, for a sleep on this process's clock. */
            for (;;)
                ;
        }
        _exit(0);
    }
    char c;
    CHECK(read(up[0], &c, 1) == 1, "child ready");
    clockid_t cc;
    CHECK(clock_getcpuclockid(pid, &cc) == 0 && cc == PROCESS_CLOCK(pid, CPUCLOCK_SCHED),
          "clock_getcpuclockid(child)");
    struct timespec t;
    CHECK(gettime(cc, &t) == 0 && ns(&t) >= 25000000, "the child's CPU time: %lld", ns(&t));
    CHECK(getres(cc, &t) == 0, "getres of the child's clock");
    /* Its thread clock isn't the caller's to read. */
    CHECK(gettime(THREAD_CLOCK(pid, CPUCLOCK_SCHED), &t) == -EINVAL, "another process's thread clock");
    timer_t id;
    CHECK(mktimer(cc, &id) == 0 && rmtimer(id) == 0, "a timer on the child's clock");
    CHECK(mktimer(THREAD_CLOCK(pid, CPUCLOCK_SCHED), &id) == -EINVAL, "a timer on another process's thread");
    /* Sleep 20 ms of its CPU time while it spins. */
    c = 2;
    write(down[1], &c, 1);
    struct timespec t20 = {0, 20000000};
    long long before = now(cc);
    CHECK(nsleep(cc, 0, &t20) == 0, "sleep on the child's clock");
    CHECK(now(cc) - before >= 20000000, "the child ran %lld ns of the 20 ms slept on", now(cc) - before);
    kill(pid, SIGKILL);
    /* A zombie's clock still reads. */
    siginfo_t si;
    CHECK(waitid(P_PID, pid, &si, WEXITED | WNOWAIT) == 0, "waitid");
    long long last = now(cc);
    CHECK(last >= 45000000, "the zombie's CPU time: %lld", last);
    CHECK(waitpid(pid, NULL, 0) == pid, "reap");
    /* Once reaped, there's nobody for it to be the clock of. */
    CHECK(gettime(cc, &t) == -EINVAL, "a reaped process's clock");
    CHECK(getres(cc, &t) == -EINVAL, "getres of a reaped process's clock");
    CHECK(nsleep(cc, 0, &t20) == -EINVAL, "sleep on a reaped process's clock");
    CHECK(mktimer(cc, &id) == -EINVAL, "a timer on a reaped process's clock");
    CHECK(settime(cc, &t20) == -EINVAL, "set a reaped process's clock");
    CHECK(clock_getcpuclockid(pid, &cc) == ESRCH, "clock_getcpuclockid(reaped)");
    close(up[0]), close(up[1]), close(down[0]), close(down[1]);
}

static void test_timers(void)
{
    timer_t id;
    static const int ok[] = {CLOCK_REALTIME, CLOCK_MONOTONIC, CLOCK_PROCESS_CPUTIME_ID, CLOCK_THREAD_CPUTIME_ID,
                             CLOCK_BOOTTIME, CLOCK_TAI};
    for (unsigned i = 0; i < sizeof ok / sizeof ok[0]; i++)
        CHECK(mktimer(ok[i], &id) == 0 && rmtimer(id) == 0, "a timer on clock %d", ok[i]);
    CHECK(mktimer(PROCESS_CLOCK(0, CPUCLOCK_PROF), &id) == 0 && rmtimer(id) == 0, "a timer on own PROF clock");
    CHECK(mktimer(THREAD_CLOCK(gettid(), CPUCLOCK_SCHED), &id) == 0 && rmtimer(id) == 0,
          "a timer on own thread clock by tid");
    static const int none[] = {CLOCK_MONOTONIC_RAW, CLOCK_REALTIME_COARSE, CLOCK_MONOTONIC_COARSE};
    for (unsigned i = 0; i < 3; i++)
        CHECK(mktimer(none[i], &id) == -EOPNOTSUPP, "a timer on clock %d", none[i]);
    CHECK(mktimer(10, &id) == -EINVAL, "a timer on clock 10");
    CHECK(mktimer(THREAD_CLOCK(0, 3), &id) == -EINVAL, "a timer on a CPU clock past SCHED");
    CHECK(mktimer(FD_CLOCK(0), &id) == -EOPNOTSUPP, "a timer on a clock device");
    /* The sigevent is read before the clock is looked at. */
    CHECK(raw(syscall(SYS_timer_create, 10, (void *)8, &id)) == -EFAULT, "a timer with an unreadable sigevent");
    if (have_alarm) {
        long r = mktimer(CLOCK_REALTIME_ALARM, &id);
        CHECK(wake_alarm ? r == 0 : r == -EPERM, "an alarm timer (CAP_WAKE_ALARM %d): %ld", wake_alarm, r);
        if (r == 0)
            rmtimer(id);
    }
}

static void test_timerfd(void)
{
    static const int bad[] = {CLOCK_PROCESS_CPUTIME_ID, CLOCK_THREAD_CPUTIME_ID, CLOCK_MONOTONIC_RAW,
                              CLOCK_REALTIME_COARSE, CLOCK_MONOTONIC_COARSE, 10, CLOCK_TAI, 12,
                              PROCESS_CLOCK(0, CPUCLOCK_SCHED)};
    for (unsigned i = 0; i < sizeof bad / sizeof bad[0]; i++)
        CHECK(raw(timerfd_create(bad[i], 0)) == -EINVAL, "timerfd on clock %d", bad[i]);
    CHECK(raw(timerfd_create(CLOCK_MONOTONIC, 1)) == -EINVAL, "timerfd with a bad flag");
    static const int good[] = {CLOCK_REALTIME, CLOCK_MONOTONIC, CLOCK_BOOTTIME};
    for (unsigned i = 0; i < 3; i++) {
        int fd = timerfd_create(good[i], TFD_CLOEXEC | TFD_NONBLOCK);
        CHECK(fd >= 0, "timerfd on clock %d", good[i]);
        close(fd);
    }
    /* The alarm clocks want CAP_WAKE_ALARM (an RTC or not). */
    int afd = timerfd_create(CLOCK_BOOTTIME_ALARM, 0);
    CHECK(wake_alarm ? afd >= 0 : (afd == -1 && errno == EPERM), "alarm timerfd (CAP_WAKE_ALARM %d): %d", wake_alarm,
          afd);
    if (afd >= 0)
        close(afd);
    /* An absolute expiry is a time on the timer's clock: 50 ms from now
     * on CLOCK_MONOTONIC is 50 ms away, not long past. */
    int fd = timerfd_create(CLOCK_MONOTONIC, 0);
    struct itimerspec its = {0};
    clock_gettime(CLOCK_MONOTONIC, &its.it_value);
    its.it_value.tv_nsec += 50000000;
    if (its.it_value.tv_nsec >= 1000000000) {
        its.it_value.tv_sec++;
        its.it_value.tv_nsec -= 1000000000;
    }
    long long a = now(CLOCK_MONOTONIC);
    CHECK(timerfd_settime(fd, TFD_TIMER_ABSTIME, &its, NULL) == 0, "timerfd_settime");
    struct itimerspec cur;
    CHECK(timerfd_gettime(fd, &cur) == 0 && ns(&cur.it_value) > 30000000 && ns(&cur.it_value) <= 50000000,
          "an absolute monotonic expiry has %lld ns to go", ns(&cur.it_value));
    uint64_t n = 0;
    CHECK(read(fd, &n, 8) == 8 && n == 1, "timerfd read %llu", (unsigned long long)n);
    long long b = now(CLOCK_MONOTONIC);
    CHECK(b - a >= 45000000 && b - a < 2000000000LL, "an absolute 50 ms timerfd took %lld ns", b - a);
    CHECK(raw(timerfd_settime(fd, 4, &its, NULL)) == -EINVAL, "timerfd_settime with a bad flag");
    struct itimerspec neg = {{0, 0}, {-1, 0}};
    CHECK(raw(timerfd_settime(fd, 0, &neg, NULL)) == -EINVAL, "timerfd_settime with a negative time");
    close(fd);
    /* An absolute CLOCK_REALTIME expiry a second ago is due at once. */
    fd = timerfd_create(CLOCK_REALTIME, 0);
    clock_gettime(CLOCK_REALTIME, &its.it_value);
    its.it_value.tv_sec--;
    CHECK(timerfd_settime(fd, TFD_TIMER_ABSTIME, &its, NULL) == 0, "timerfd_settime");
    n = 0;
    CHECK(read(fd, &n, 8) == 8 && n == 1, "a past realtime expiry read %llu", (unsigned long long)n);
    close(fd);
}

static void test_set_adjust(void)
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    /* Only CLOCK_REALTIME and the CPU clocks have a setter at all. */
    CHECK(settime(CLOCK_MONOTONIC, &t) == -EINVAL, "set CLOCK_MONOTONIC");
    CHECK(settime(CLOCK_MONOTONIC, NULL) == -EINVAL, "set CLOCK_MONOTONIC, no time");
    CHECK(settime(CLOCK_PROCESS_CPUTIME_ID, &t) == -EINVAL, "set CLOCK_PROCESS_CPUTIME_ID");
    CHECK(settime(10, &t) == -EINVAL, "set clock 10");
    /* A CPU clock by pid can't be set either, but says so with EPERM. */
    CHECK(settime(PROCESS_CLOCK(0, CPUCLOCK_SCHED), &t) == -EPERM, "set own process clock");
    CHECK(settime(PROCESS_CLOCK(0, CPUCLOCK_SCHED), NULL) == -EFAULT, "set own process clock, no time");
    CHECK(settime(FD_CLOCK(0), &t) == -EINVAL, "set a clock device");
    CHECK(settime(CLOCK_REALTIME, NULL) == -EFAULT, "set the time from nowhere");
    struct timespec neg = {-1, 0}, bad = {t.tv_sec, 1000000000};
    CHECK(settime(CLOCK_REALTIME, &neg) == -EINVAL, "set a negative time");
    CHECK(settime(CLOCK_REALTIME, &bad) == -EINVAL, "set a bad time");
    if (geteuid() != 0)
        CHECK(settime(CLOCK_REALTIME, &t) == -EPERM, "set the time unprivileged");

    struct timex tx = {0};
    CHECK(raw(syscall(SYS_clock_adjtime, CLOCK_REALTIME, &tx)) >= 0, "clock_adjtime(CLOCK_REALTIME)");
    static const int noadj[] = {CLOCK_MONOTONIC, CLOCK_BOOTTIME, CLOCK_TAI, CLOCK_PROCESS_CPUTIME_ID};
    for (unsigned i = 0; i < 4; i++) {
        memset(&tx, 0, sizeof tx);
        CHECK(raw(syscall(SYS_clock_adjtime, noadj[i], &tx)) == -EOPNOTSUPP, "clock_adjtime(%d)", noadj[i]);
    }
    memset(&tx, 0, sizeof tx);
    CHECK(raw(syscall(SYS_clock_adjtime, PROCESS_CLOCK(0, CPUCLOCK_SCHED), &tx)) == -EOPNOTSUPP,
          "clock_adjtime(own process clock)");
    CHECK(raw(syscall(SYS_clock_adjtime, 10, &tx)) == -EINVAL, "clock_adjtime(10)");
    CHECK(raw(syscall(SYS_clock_adjtime, FD_CLOCK(0), &tx)) == -EINVAL, "clock_adjtime(a clock device)");
    CHECK(raw(syscall(SYS_clock_adjtime, 10, NULL)) == -EFAULT, "clock_adjtime(10, NULL)");
}

/* The 64-bit vDSO's symbol `name`, through DT_HASH and DT_VERSYM as musl
 * looks it up. */
static void *vdso_sym(const char *ver, const char *name)
{
    unsigned long base = getauxval(AT_SYSINFO_EHDR);
    if (!base)
        return NULL;
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)base;
    const Elf64_Phdr *ph = (const Elf64_Phdr *)(base + eh->e_phoff);
    const Elf64_Dyn *dyn = NULL;
    long load = 0;
    for (int i = 0; i < eh->e_phnum; i++) {
        if (ph[i].p_type == PT_LOAD)
            load = (long)ph[i].p_offset - (long)ph[i].p_vaddr;
        if (ph[i].p_type == PT_DYNAMIC)
            dyn = (const Elf64_Dyn *)(base + ph[i].p_offset);
    }
    if (!dyn)
        return NULL;
    const uint32_t *hash = NULL;
    const Elf64_Sym *syms = NULL;
    const char *strs = NULL;
    const uint16_t *versym = NULL;
    const unsigned char *verdef = NULL;
    for (; dyn->d_tag; dyn++) {
        const unsigned char *at = (const unsigned char *)(base + dyn->d_un.d_ptr + load);
        switch (dyn->d_tag) {
        case DT_HASH: hash = (const uint32_t *)at; break;
        case DT_STRTAB: strs = (const char *)at; break;
        case DT_SYMTAB: syms = (const Elf64_Sym *)at; break;
        case DT_VERSYM: versym = (const uint16_t *)at; break;
        case DT_VERDEF: verdef = at; break;
        }
    }
    if (!hash || !syms || !strs || !versym || !verdef)
        return NULL;
    for (uint32_t i = 1; i < hash[1]; i++) {
        if (strcmp(strs + syms[i].st_name, name) != 0)
            continue;
        uint16_t want = versym[i] & 0x7fff;
        for (const unsigned char *d = verdef;; d += ((const Elf64_Verdef *)d)->vd_next) {
            const Elf64_Verdef *vd = (const Elf64_Verdef *)d;
            if (vd->vd_ndx == want) {
                const Elf64_Verdaux *aux = (const Elf64_Verdaux *)(d + vd->vd_aux);
                if (strcmp(strs + aux->vda_name, ver) == 0)
                    return (void *)(base + syms[i].st_value);
                break;
            }
            if (vd->vd_next == 0)
                break;
        }
    }
    return NULL;
}

/* The vDSO's clock functions give what the system calls give, for the
 * clocks they compute and those they don't (negative ids too), and take
 * the id as the 32-bit clockid_t it is. */
static void test_vdso(void)
{
    int (*cgt)(long, struct timespec *) = vdso_sym("LINUX_2.6", "__vdso_clock_gettime");
    int (*cgr)(long, struct timespec *) = vdso_sym("LINUX_2.6", "__vdso_clock_getres");
    CHECK(cgt && cgr, "the vDSO's clock functions %p %p", cgt, cgr);
    if (!cgt || !cgr)
        return;
    const long ids[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 100, -1, -2, -6, -8,
                        PROCESS_CLOCK(getpid(), CPUCLOCK_SCHED), THREAD_CLOCK(gettid(), CPUCLOCK_SCHED),
                        PROCESS_CLOCK(0x7fffff, CPUCLOCK_SCHED), FD_CLOCK(0)};
    for (unsigned i = 0; i < sizeof ids / sizeof ids[0]; i++) {
        long c = ids[i];
        struct timespec v, k;
        long rv = cgt(c, &v), rk = gettime((clockid_t)c, &k);
        CHECK(rv == rk, "clock %ld: vDSO %ld, system call %ld", c, rv, rk);
        rv = cgr(c, &v), rk = getres((clockid_t)c, &k);
        CHECK(rv == rk && (rv != 0 || (v.tv_sec == k.tv_sec && v.tv_nsec == k.tv_nsec)),
              "getres %ld: vDSO %ld %ld.%09ld, system call %ld %ld.%09ld", c, rv, (long)v.tv_sec, v.tv_nsec, rk,
              (long)k.tv_sec, k.tv_nsec);
    }
    /* Whatever the upper half of the register holds. */
    struct timespec a, v, z;
    clock_gettime(CLOCK_MONOTONIC, &a);
    long r = cgt((long)0x5a5a5a5a00000000L | CLOCK_MONOTONIC, &v);
    clock_gettime(CLOCK_MONOTONIC, &z);
    CHECK(r == 0 && ns(&v) >= ns(&a) && ns(&v) <= ns(&z), "CLOCK_MONOTONIC with high bits: %ld %lld (%lld..%lld)",
          r, ns(&v), ns(&a), ns(&z));
    r = cgt((long)0x5a5a5a5a00000000L | 10, &v);
    CHECK(r == -EINVAL, "clock 10 with high bits: %ld", r);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    wake_alarm = has_wake_alarm();
    test_ids();
    test_threads();
    test_sleeps();
    test_process();
    test_timers();
    test_timerfd();
    test_set_adjust();
    test_vdso();
    if (failures) {
        printf("%d failures\n", failures);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
