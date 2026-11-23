/* Timers on CPU time, as Linux's posix-cpu-timers.c and itimer.c run them:
 * POSIX timers on CLOCK_PROCESS_CPUTIME_ID, CLOCK_THREAD_CPUTIME_ID and the
 * CPU clocks of another thread or process expire on the CPU time that clock's
 * task uses, not on wall time — a process clock on the sum over its threads,
 * the exited ones included; a thread clock on that thread's alone — and
 * signal with SIGEV_SIGNAL or SIGEV_THREAD_ID, or not at all with SIGEV_NONE.
 * timer_gettime counts down with CPU time, a periodic timer re-arms, and the
 * expirations that come while its signal is pending are overruns. A timer on
 * a process that is gone can't be set (ESRCH). ITIMER_VIRTUAL and ITIMER_PROF
 * count CPU time too, aren't inherited by a fork child, and survive exec,
 * which deletes the POSIX timers; RLIMIT_CPU's SIGXCPU moves the soft limit
 * on a second.
 *
 * It passes on Linux too: `gcc -static -O1 -pthread -o cpu_timers cpu_timers.c`. */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef sigev_notify_thread_id
#define sigev_notify_thread_id _sigev_un._tid
#endif

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

#define MS 1000000LL

static long long ns(const struct timespec *t) { return t->tv_sec * 1000000000LL + t->tv_nsec; }
static long long now(clockid_t c)
{
    struct timespec t;
    if (clock_gettime(c, &t) != 0)
        return -1;
    return ns(&t);
}
static long long cpu(void) { return now(CLOCK_PROCESS_CPUTIME_ID); }
static pid_t gettid_(void) { return (pid_t)syscall(SYS_gettid); }

static void sleep_ms(long ms)
{
    struct timespec t = {ms / 1000, (ms % 1000) * MS};
    while (nanosleep(&t, &t) != 0 && errno == EINTR)
        ;
}

/* Use CPU until `*flag` is set, or `limit` of this thread's CPU time, or
 * ten seconds of wall time, whichever comes first. */
static void burn_until(volatile sig_atomic_t *flag, long long limit)
{
    long long start = now(CLOCK_THREAD_CPUTIME_ID), wall = now(CLOCK_MONOTONIC);
    while (!(flag && *flag) && now(CLOCK_THREAD_CPUTIME_ID) - start < limit &&
           now(CLOCK_MONOTONIC) - wall < 10000 * MS)
        ;
}
static void burn(long long limit) { burn_until(NULL, limit); }

static timer_t make(clockid_t c, int notify, int sig, pid_t tid)
{
    struct sigevent ev;
    memset(&ev, 0, sizeof ev);
    ev.sigev_notify = notify;
    ev.sigev_signo = sig;
    ev.sigev_value.sival_int = 42;
    ev.sigev_notify_thread_id = tid;
    timer_t id = 0;
    CHECK(timer_create(c, &ev, &id) == 0, "timer_create(%d)", c);
    return id;
}

static int arm(timer_t id, long long value, long long interval, int flags)
{
    struct itimerspec it = {{interval / 1000000000LL, interval % 1000000000LL},
                            {value / 1000000000LL, value % 1000000000LL}};
    return timer_settime(id, flags, &it, NULL);
}

static long long left(timer_t id, long long *interval)
{
    struct itimerspec it;
    if (timer_gettime(id, &it) != 0)
        return -1;
    if (interval)
        *interval = ns(&it.it_interval);
    return ns(&it.it_value);
}

static volatile sig_atomic_t fired, fired_tid, fired_code, fired_value;
static volatile long long fired_cpu;
static void on_timer(int sig, siginfo_t *si, void *uc)
{
    (void)sig, (void)uc;
    fired_cpu = cpu();
    fired_tid = gettid_();
    fired_code = si->si_code;
    fired_value = si->si_value.sival_int;
    fired++;
}

static void handle(int sig, void (*fn)(int, siginfo_t *, void *))
{
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = fn;
    sa.sa_flags = SA_SIGINFO;
    sigaction(sig, &sa, NULL);
}

/* A process timer waits for CPU time: a sleep, however long, doesn't
 * bring it closer; running does, and its signal comes once the process has
 * run that long. */
static void process_timer_counts_cpu_time(void)
{
    handle(SIGUSR1, on_timer);
    fired = 0;
    timer_t id = make(CLOCK_PROCESS_CPUTIME_ID, SIGEV_SIGNAL, SIGUSR1, 0);
    long long c0 = cpu();
    CHECK(arm(id, 50 * MS, 0, 0) == 0, "arm");
    sleep_ms(150);
    CHECK(!fired, "fired after 150 ms asleep");
    long long l = left(id, NULL);
    CHECK(l > 40 * MS && l <= 50 * MS, "%lld ns left after a sleep", l);
    burn_until(&fired, 5000 * MS);
    CHECK(fired == 1, "fired %d times", fired);
    CHECK(fired_cpu - c0 >= 50 * MS, "fired after %lld ns of CPU time", fired_cpu - c0);
    CHECK(fired_code == SI_TIMER && fired_value == 42, "si_code %d, si_value %d", fired_code,
          fired_value);
    /* A one-shot timer that expired is disarmed. */
    long long iv = -1;
    CHECK(left(id, &iv) == 0 && iv == 0, "an expired one-shot timer reads %lld/%lld", left(id, NULL), iv);
    timer_delete(id);
}

/* timer_gettime counts down with the CPU time used, and only with it; a
 * one-shot SIGEV_NONE timer then reads 0, and a periodic one moves on. */
static void gettime_counts_down_with_cpu_time(void)
{
    timer_t id = make(CLOCK_PROCESS_CPUTIME_ID, SIGEV_NONE, 0, 0);
    long long c0 = cpu();
    CHECK(arm(id, 200 * MS, 0, 0) == 0, "arm");
    long long c0b = cpu();
    burn(50 * MS);
    long long c1b = cpu();
    long long l = left(id, NULL);
    long long c1 = cpu();
    CHECK(l >= 200 * MS - (c1 - c0) && l <= 200 * MS - (c1b - c0b),
          "%lld ns left after %lld..%lld ns of CPU time", l, c1b - c0b, c1 - c0);
    burn(200 * MS);
    long long iv = -1;
    CHECK(left(id, &iv) == 0 && iv == 0, "an expired SIGEV_NONE timer reads %lld", left(id, NULL));

    CHECK(arm(id, 20 * MS, 20 * MS, 0) == 0, "arm periodic");
    burn(50 * MS);
    l = left(id, &iv);
    CHECK(l > 0 && l <= 20 * MS && iv == 20 * MS, "periodic SIGEV_NONE reads %lld/%lld", l, iv);
    /* An absolute expiry is a time on the clock. */
    CHECK(arm(id, cpu() + 100 * MS, 0, TIMER_ABSTIME) == 0, "arm absolute");
    l = left(id, NULL);
    CHECK(l > 90 * MS && l <= 100 * MS, "absolute: %lld left", l);
    /* The old setting comes back. */
    struct itimerspec zero = {{0, 0}, {0, 0}}, old;
    CHECK(timer_settime(id, 0, &zero, &old) == 0 && ns(&old.it_value) > 80 * MS &&
              ns(&old.it_value) <= 100 * MS,
          "old setting %lld", ns(&old.it_value));
    CHECK(left(id, &iv) == 0 && iv == 0, "disarmed");
    timer_delete(id);
}

/* A periodic timer re-arms; while its signal is pending, expirations are
 * overruns, which the signal carries when it's taken. */
static void periodic_timer_overruns(void)
{
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGUSR2);
    sigprocmask(SIG_BLOCK, &set, NULL);
    timer_t id = make(CLOCK_PROCESS_CPUTIME_ID, SIGEV_SIGNAL, SIGUSR2, 0);
    long long c0 = cpu();
    CHECK(arm(id, 10 * MS, 10 * MS, 0) == 0, "arm");
    burn(65 * MS);
    long long c1 = cpu();
    siginfo_t si;
    struct timespec zero = {0, 0};
    CHECK(sigtimedwait(&set, &si, &zero) == SIGUSR2, "no SIGUSR2 after %lld ns", c1 - c0);
    /* Expirations at 10, 20, ... ms: all but the first are overruns. */
    int expected = (int)((c1 - c0) / (10 * MS)) - 1;
    CHECK(si.si_code == SI_TIMER && si.si_overrun >= 4 && si.si_overrun <= expected + 1,
          "si_code %d, si_overrun %d (about %d)", si.si_code, si.si_overrun, expected);
    CHECK(timer_getoverrun(id) == si.si_overrun, "timer_getoverrun %d", timer_getoverrun(id));
    /* It goes on. */
    burn(25 * MS);
    CHECK(sigtimedwait(&set, &si, &zero) == SIGUSR2, "re-armed");
    long long iv;
    long long l = left(id, &iv);
    CHECK(l > 0 && l <= 10 * MS && iv == 10 * MS, "periodic reads %lld/%lld", l, iv);
    timer_delete(id);
    sigtimedwait(&set, &si, &zero);
    sigprocmask(SIG_UNBLOCK, &set, NULL);
}

static volatile sig_atomic_t stop;
struct worker {
    pthread_t th;
    pid_t tid;
    long long used;
    int wait; /* a pipe to wait on before running, or -1 */
};
static void *spin(void *arg)
{
    struct worker *w = arg;
    w->tid = gettid_();
    /* Asleep until told to go: blocked, not waking up now and then, which
     * would use CPU time of its own. */
    char c;
    if (w->wait >= 0 && read(w->wait, &c, 1) != 1)
        return NULL;
    burn_until(&stop, 5000 * MS);
    w->used = now(CLOCK_THREAD_CPUTIME_ID);
    return NULL;
}

/* A process timer counts every thread's time, and the process clock keeps
 * what exited threads used. */
static void process_timer_sums_threads(void)
{
    handle(SIGUSR1, on_timer);
    fired = 0;
    stop = 0;
    long long c0 = cpu();
    timer_t id = make(CLOCK_PROCESS_CPUTIME_ID, SIGEV_SIGNAL, SIGUSR1, 0);
    CHECK(arm(id, 100 * MS, 0, 0) == 0, "arm");
    struct worker w[2] = {{.wait = -1}, {.wait = -1}};
    for (int i = 0; i < 2; i++)
        pthread_create(&w[i].th, NULL, spin, &w[i]);
    long long wall = now(CLOCK_MONOTONIC);
    while (!fired && now(CLOCK_MONOTONIC) - wall < 10000 * MS)
        sleep_ms(2);
    stop = 1;
    for (int i = 0; i < 2; i++)
        pthread_join(w[i].th, NULL);
    CHECK(fired == 1, "fired %d times", fired);
    CHECK(fired_cpu - c0 >= 100 * MS, "fired at %lld ns", fired_cpu - c0);
    CHECK(w[0].used > 0 && w[1].used > 0, "threads used %lld and %lld", w[0].used, w[1].used);
    long long total = cpu() - c0;
    CHECK(total >= w[0].used + w[1].used, "the process clock (%lld) lost exited threads' time (%lld + %lld)",
          total, w[0].used, w[1].used);
    timer_delete(id);
}

/* A timer on a thread's clock counts that thread's time only, and
 * SIGEV_THREAD_ID signals the thread it names. */
static void thread_timer_counts_its_thread(void)
{
    handle(SIGUSR1, on_timer);
    fired = 0;
    stop = 0;
    int go[2];
    CHECK(pipe(go) == 0, "pipe");
    struct worker w = {.wait = go[0]};
    pthread_create(&w.th, NULL, spin, &w);
    while (!w.tid)
        sleep_ms(1);
    clockid_t tc;
    CHECK(pthread_getcpuclockid(w.th, &tc) == 0, "pthread_getcpuclockid");
    timer_t id = make(tc, SIGEV_THREAD_ID, SIGUSR1, w.tid);
    CHECK(arm(id, 30 * MS, 0, 0) == 0, "arm");
    /* This thread's time isn't the clock's. */
    burn(80 * MS);
    CHECK(!fired, "fired on another thread's time");
    long long l = left(id, NULL);
    CHECK(l > 25 * MS && l <= 30 * MS, "%lld left", l);
    long long t0 = now(tc);
    CHECK(write(go[1], "x", 1) == 1, "write");
    long long wall = now(CLOCK_MONOTONIC);
    while (!fired && now(CLOCK_MONOTONIC) - wall < 10000 * MS)
        sleep_ms(2);
    stop = 1;
    pthread_join(w.th, NULL);
    CHECK(fired == 1 && fired_tid == w.tid, "fired %d times, on %d (the thread is %d)", fired, fired_tid,
          w.tid);
    CHECK(w.used - t0 >= 30 * MS, "the thread used %lld", w.used - t0);
    /* Its thread gone, the timer is still there, but not set. */
    CHECK(left(id, NULL) == 0, "the timer of a thread that exited");
    CHECK(timer_delete(id) == 0, "timer_delete");
    close(go[0]);
    close(go[1]);
}

/* A timer on another process's clock counts that process's time, and
 * signals this one, even asleep; once the process is reaped, the timer
 * can't be set. */
static void timer_on_another_process(void)
{
    int go[2];
    CHECK(pipe(go) == 0, "pipe");
    pid_t pid = fork();
    if (pid == 0) {
        char c;
        if (read(go[0], &c, 1) < 0)
            _exit(2);
        burn(100 * MS);
        if (read(go[0], &c, 1) < 0)
            _exit(2);
        _exit(0);
    }
    clockid_t cc;
    CHECK(clock_getcpuclockid(pid, &cc) == 0, "clock_getcpuclockid");
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGUSR2);
    sigprocmask(SIG_BLOCK, &set, NULL);
    timer_t id = make(cc, SIGEV_SIGNAL, SIGUSR2, 0);
    CHECK(arm(id, 40 * MS, 0, 0) == 0, "arm");
    burn(60 * MS);
    siginfo_t si;
    struct timespec zero = {0, 0}, five = {5, 0};
    CHECK(sigtimedwait(&set, &si, &zero) == -1, "fired on this process's time");
    CHECK(write(go[1], "x", 1) == 1, "write");
    long long w0 = now(CLOCK_MONOTONIC);
    CHECK(sigtimedwait(&set, &si, &five) == SIGUSR2 && si.si_code == SI_TIMER,
          "no signal from the child's clock after %lld ns", now(CLOCK_MONOTONIC) - w0);
    CHECK(now(cc) >= 40 * MS, "the child used %lld", now(cc));
    CHECK(write(go[1], "x", 1) == 1, "write");
    int st;
    waitpid(pid, &st, 0);
    CHECK(arm(id, 40 * MS, 0, 0) == -1 && errno == ESRCH, "set a timer on a reaped process");
    long long iv = -1;
    CHECK(left(id, &iv) == 0 && iv == 0, "the timer of a reaped process");
    timer_delete(id);
    sigprocmask(SIG_UNBLOCK, &set, NULL);
    close(go[0]);
    close(go[1]);
}

static volatile sig_atomic_t prof, vtalrm;
static void on_prof(int sig) { prof += sig == SIGPROF; }
static void on_vtalrm(int sig) { vtalrm += sig == SIGVTALRM; }

static long long itimer_left(int which, long long *interval)
{
    struct itimerval v;
    if (getitimer(which, &v) != 0)
        return -1;
    if (interval)
        *interval = v.it_interval.tv_sec * 1000000000LL + v.it_interval.tv_usec * 1000LL;
    return v.it_value.tv_sec * 1000000000LL + v.it_value.tv_usec * 1000LL;
}

static int set_itimer(int which, long usec, long interval_usec, struct itimerval *old)
{
    struct itimerval v = {{interval_usec / 1000000, interval_usec % 1000000}, {usec / 1000000, usec % 1000000}};
    return setitimer(which, &v, old);
}

/* ITIMER_PROF and ITIMER_VIRTUAL count CPU time, a tick more than asked. */
static void cpu_itimers(void)
{
    signal(SIGPROF, on_prof);
    signal(SIGVTALRM, on_vtalrm);
    prof = vtalrm = 0;
    CHECK(set_itimer(ITIMER_PROF, 50000, 0, NULL) == 0, "setitimer(ITIMER_PROF)");
    CHECK(set_itimer(ITIMER_VIRTUAL, 60000, 0, NULL) == 0, "setitimer(ITIMER_VIRTUAL)");
    sleep_ms(150);
    CHECK(!prof && !vtalrm, "an itimer fired asleep");
    long long l = itimer_left(ITIMER_PROF, NULL);
    CHECK(l > 45 * MS && l <= 70 * MS, "ITIMER_PROF: %lld left", l);
    burn_until(&vtalrm, 5000 * MS);
    CHECK(prof == 1 && vtalrm == 1, "SIGPROF %d, SIGVTALRM %d", prof, vtalrm);
    CHECK(itimer_left(ITIMER_PROF, NULL) == 0, "a one-shot itimer is disarmed once it fires");

    /* Periodic. */
    prof = 0;
    CHECK(set_itimer(ITIMER_PROF, 10000, 10000, NULL) == 0, "periodic ITIMER_PROF");
    long long iv;
    l = itimer_left(ITIMER_PROF, &iv);
    CHECK(l > 9 * MS && iv == 10 * MS, "periodic ITIMER_PROF reads %lld/%lld", l, iv);
    burn(100 * MS);
    struct itimerval old;
    CHECK(set_itimer(ITIMER_PROF, 0, 0, &old) == 0, "disarm");
    CHECK(prof >= 3, "%d SIGPROFs in 100 ms of CPU time at 10 ms", prof);
    CHECK(old.it_interval.tv_usec == 10000 && old.it_value.tv_sec == 0 && old.it_value.tv_usec > 0,
          "old ITIMER_PROF %ld/%ld", (long)old.it_interval.tv_usec, (long)old.it_value.tv_usec);

    /* A null setting disarms; bad ones are EINVAL. */
    CHECK(set_itimer(ITIMER_VIRTUAL, 1000000, 0, NULL) == 0, "arm ITIMER_VIRTUAL");
    errno = 0;
    CHECK(syscall(SYS_setitimer, ITIMER_VIRTUAL, NULL, NULL) == 0, "setitimer(NULL)");
    CHECK(itimer_left(ITIMER_VIRTUAL, NULL) == 0, "a null setting disarms");
    struct itimerval bad = {{0, 0}, {0, 1000000}};
    CHECK(setitimer(ITIMER_PROF, &bad, NULL) == -1 && errno == EINVAL, "tv_usec of a second");
    struct itimerval ok = {{0, 0}, {1, 0}};
    CHECK(setitimer(3, &ok, NULL) == -1 && errno == EINVAL, "setitimer(3)");
    CHECK(getitimer(3, &old) == -1 && errno == EINVAL, "getitimer(3)");
    signal(SIGPROF, SIG_DFL);
    signal(SIGVTALRM, SIG_DFL);
}

/* exec deletes the POSIX timers and keeps the itimers; fork has neither. */
static int exec_check(void)
{
    struct itimerspec it;
    int bad = 0;
    if (itimer_left(ITIMER_VIRTUAL, NULL) <= 0) {
        printf("FAIL exec_check: ITIMER_VIRTUAL didn't survive exec\n");
        bad = 1;
    }
    if (syscall(SYS_timer_gettime, 0, &it) != -1 || errno != EINVAL) {
        printf("FAIL exec_check: a POSIX timer survived exec\n");
        bad = 1;
    }
    return bad;
}

static void fork_and_exec(void)
{
    CHECK(set_itimer(ITIMER_PROF, 100000000, 0, NULL) == 0, "arm ITIMER_PROF");
    pid_t pid = fork();
    if (pid == 0) {
        if (itimer_left(ITIMER_PROF, NULL) != 0) {
            printf("FAIL fork_and_exec: a fork child inherited ITIMER_PROF\n");
            _exit(1);
        }
        make(CLOCK_PROCESS_CPUTIME_ID, SIGEV_NONE, 0, 0);
        set_itimer(ITIMER_VIRTUAL, 100000000, 0, NULL);
        execl("/proc/self/exe", "cpu_timers", "exec-check", (char *)NULL);
        printf("FAIL fork_and_exec: exec: %s\n", strerror(errno));
        _exit(1);
    }
    int st;
    waitpid(pid, &st, 0);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "child %#x", st);
    set_itimer(ITIMER_PROF, 0, 0, NULL);
}

static volatile sig_atomic_t xcpu;
static void on_xcpu(int sig) { xcpu += sig == SIGXCPU; }

/* RLIMIT_CPU's soft limit moves on a second when it sends SIGXCPU. */
static void rlimit_cpu_moves_on(void)
{
    pid_t pid = fork();
    if (pid == 0) {
        struct rlimit r = {1, RLIM_INFINITY};
        signal(SIGXCPU, on_xcpu);
        setrlimit(RLIMIT_CPU, &r);
        burn_until(&xcpu, 5000 * MS);
        getrlimit(RLIMIT_CPU, &r);
        _exit(xcpu == 1 && r.rlim_cur == 2 ? 0 : 10 + (int)r.rlim_cur);
    }
    int st;
    waitpid(pid, &st, 0);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "SIGXCPU and the soft limit after it: %#x", st);
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "exec-check") == 0)
        return exec_check();
    setvbuf(stdout, NULL, _IONBF, 0);
    process_timer_counts_cpu_time();
    gettime_counts_down_with_cpu_time();
    periodic_timer_overruns();
    process_timer_sums_threads();
    thread_timer_counts_its_thread();
    timer_on_another_process();
    cpu_itimers();
    fork_and_exec();
    rlimit_cpu_moves_on();
    if (failures)
        return 1;
    printf("PASS\n");
    return 0;
}
