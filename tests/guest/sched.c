// Real-time scheduling: SCHED_FIFO and SCHED_RR tasks run ahead of fair ones
// and of lower priorities, FIFO in turn among equals, RR by time slice, and
// the fair tasks a real-time hog starves still get Linux's fair-server share.
// Wants root (it makes real-time tasks); every task is pinned to CPU 0 so
// they compete for it.
#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHECK(c)                                                               \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno);     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

struct shared {
    volatile double cpu; // the child's CPU time, kept up to date as it spins
    volatile int flag;
};

static struct shared *sh;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static double own_cpu(void) {
    struct timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static void set_policy(pid_t pid, int policy, int prio) {
    struct sched_param sp = {.sched_priority = prio};
    CHECK(sched_setscheduler(pid, policy, &sp) == 0);
}

static void spin_for(double secs) {
    double t0 = now();
    while (now() - t0 < secs)
        ;
}

// A child that spins forever, publishing its CPU time.
static pid_t spinner(void) {
    sh->cpu = 0;
    pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        for (;;)
            sh->cpu = own_cpu();
    }
    return pid;
}

static void reap(pid_t pid) {
    kill(pid, SIGKILL);
    CHECK(waitpid(pid, NULL, 0) == pid);
}

int main(void) {
    sh = mmap(NULL, sizeof *sh, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    CHECK(sh != MAP_FAILED);
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(0, &one);
    CHECK(sched_setaffinity(0, sizeof one, &one) == 0);

    struct timespec ts;
    CHECK(sched_rr_get_interval(0, &ts) == 0);
    struct sched_param sp = {.sched_priority = 1};
    CHECK(sched_setscheduler(0, SCHED_RR, &sp) == 0);
    CHECK(sched_rr_get_interval(0, &ts) == 0);
    CHECK(ts.tv_sec == 0 && ts.tv_nsec == 100000000);
    set_policy(0, SCHED_OTHER, 0);

    // A fair task gets nothing while a FIFO task wants the CPU; over two
    // seconds the fair server still gives it about 50 ms in each.
    pid_t fair = spinner();
    usleep(100000);
    CHECK(sh->cpu > 0.02);
    set_policy(0, SCHED_FIFO, 10);
    double c0 = sh->cpu, t0 = now(), early = -1;
    while (now() - t0 < 2.2) {
        if (early < 0 && now() - t0 >= 0.3)
            early = sh->cpu - c0;
    }
    double total = sh->cpu - c0;
    set_policy(0, SCHED_OTHER, 0);
    reap(fair);
    printf("a fair task beside a FIFO hog: %.3f s of CPU in its first 0.3 s, %.3f s in 2.2 s\n",
           early, total);
    CHECK(early < 0.03);
    CHECK(total > 0.04 && total < 0.4);

    // Among FIFO tasks of one priority, the first runs until it yields.
    set_policy(0, SCHED_FIFO, 10);
    sh->flag = 0;
    pid_t kid = fork();
    CHECK(kid >= 0);
    if (kid == 0) {
        sh->flag = 1;
        _exit(0);
    }
    spin_for(0.1);
    CHECK(sh->flag == 0);
    sched_yield();
    CHECK(sh->flag == 1);
    CHECK(waitpid(kid, NULL, 0) == kid);

    // Raising a task's priority above ours hands it the CPU at once.
    sh->flag = 0;
    kid = fork();
    CHECK(kid >= 0);
    if (kid == 0) {
        sh->flag = 1;
        _exit(0);
    }
    spin_for(0.05);
    CHECK(sh->flag == 0);
    set_policy(kid, SCHED_FIFO, 20);
    CHECK(sh->flag == 1);
    CHECK(waitpid(kid, NULL, 0) == kid);

    // SCHED_RR tasks of one priority take turns, a time slice each.
    set_policy(0, SCHED_RR, 10);
    pid_t rr = spinner();
    c0 = sh->cpu;
    spin_for(0.6);
    double got = sh->cpu - c0;
    reap(rr);
    set_policy(0, SCHED_OTHER, 0);
    printf("two SCHED_RR tasks: the other got %.3f s of 0.6 s\n", got);
    CHECK(got > 0.15 && got < 0.45);

    // The same pair under SCHED_FIFO: the other never runs.
    set_policy(0, SCHED_FIFO, 10);
    pid_t ff = spinner();
    c0 = sh->cpu;
    spin_for(0.3);
    got = sh->cpu - c0;
    reap(ff);
    set_policy(0, SCHED_OTHER, 0);
    CHECK(got < 0.03);

    printf("PASS\n");
    return 0;
}
