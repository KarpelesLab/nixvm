/* An i386 program: what a 32-bit Linux binary asks of an x86-64 kernel.
 *
 * Built with `gcc -m32 -static` and exec'd as an ELFCLASS32 process, it runs
 * in compatibility mode below 4 GiB and makes every call through the i386
 * table (glibc goes through the vDSO's __kernel_vsyscall, so the fast
 * entry, syscall or sysenter, is used too). It checks the calls whose
 * structures or arguments the compat layer translates: stat64, statfs64,
 * the time structures, rlimits, rusage, 64-bit offsets split across two
 * registers, fcntl64 locks, getdents, select, msghdr with SCM_RIGHTS,
 * sendmmsg/recvmmsg, socketcall, the SysV ipc calls, mmap2 offsets, ia32
 * signal frames (rt and not), sigaltstack, sigprocmask, fork/wait4/waitid,
 * execve, threads with GDT TLS (clone's user_desc), and the vDSO.
 *
 * It passes on Linux too: `gcc -m32 -static -O1 -pthread -o ia32 ia32.c`.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/msg.h>
#include <sys/poll.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/uio.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <time.h>
#include <ucontext.h>
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

static long raw_int80(long nr, long a, long b, long c)
{
    long ret;
    __asm__ volatile("int $0x80" : "=a"(ret) : "a"(nr), "b"(a), "c"(b), "d"(c) : "memory");
    return ret;
}

static void test_mode(void)
{
    unsigned short cs, ds, ss;
    __asm__("mov %%cs, %0; mov %%ds, %1; mov %%ss, %2" : "=r"(cs), "=r"(ds), "=r"(ss));
    CHECK(cs == 0x23, "cs %#x", cs);
    CHECK(ds == 0x2b && ss == 0x2b, "ds %#x ss %#x", ds, ss);
    CHECK((uintptr_t)&cs < 0xffffe000u, "stack %p", (void *)&cs);
    CHECK(sizeof(long) == 4, "long");
    CHECK(raw_int80(SYS_getpid, 0, 0, 0) == getpid(), "int80 getpid");
    /* 16-bit and 32-bit ID calls. */
    CHECK(syscall(SYS_getuid) == (long)(getuid() & 0xffff), "getuid16");
    CHECK(syscall(SYS_getuid32) == (long)getuid(), "getuid32");
}

unsigned long vsys;
long vargs[7];

static void test_auxv_vdso(void)
{
    vsys = getauxval(AT_SYSINFO);
    unsigned long ehdr = getauxval(AT_SYSINFO_EHDR);
    CHECK(vsys != 0 && ehdr != 0, "AT_SYSINFO %#lx AT_SYSINFO_EHDR %#lx", vsys, ehdr);
    if (ehdr) {
        const unsigned char *e = (const unsigned char *)ehdr;
        CHECK(memcmp(e, "\177ELF", 4) == 0 && e[4] == 1, "vDSO is ELF32");
        CHECK(vsys > ehdr && vsys < ehdr + 0x2000, "vsyscall in the vDSO");
    }
    char plat[16] = "";
    const char *p = (const char *)getauxval(AT_PLATFORM);
    if (p)
        snprintf(plat, sizeof plat, "%s", p);
    CHECK(strcmp(plat, "i686") == 0, "AT_PLATFORM %s", plat);
    /* Through the vDSO's __kernel_vsyscall by hand: getpid. */
    long r;
    __asm__ volatile("call *%1" : "=a"(r) : "r"(vsys), "a"(SYS_getpid) : "memory", "ecx", "edx");
    CHECK(r == getpid(), "__kernel_vsyscall getpid %ld", r);
    /* With six arguments (the sixth goes through the stack): mmap2. */
    vargs[0] = SYS_mmap2;
    vargs[1] = 0;
    vargs[2] = 8192;
    vargs[3] = PROT_READ | PROT_WRITE;
    vargs[4] = MAP_PRIVATE | MAP_ANONYMOUS;
    vargs[5] = -1;
    vargs[6] = 0;
    long addr;
    __asm__ volatile("push %%ebx; push %%esi; push %%edi; push %%ebp\n\t"
                     "mov vargs+4, %%ebx; mov vargs+8, %%ecx; mov vargs+12, %%edx\n\t"
                     "mov vargs+16, %%esi; mov vargs+20, %%edi; mov vargs+24, %%ebp\n\t"
                     "mov vargs, %%eax; call *vsys\n\t"
                     "pop %%ebp; pop %%edi; pop %%esi; pop %%ebx"
                     : "=a"(addr)
                     :
                     : "ecx", "edx", "memory");
    CHECK((unsigned long)addr < 0xffffe000ul && addr != 0 && (unsigned long)addr % 4096 == 0, "vsyscall mmap2 %#lx", addr);
    if ((unsigned long)addr < 0xfffff000ul) {
        *(volatile int *)addr = 1;
        munmap((void *)addr, 8192);
    }
}

static void test_files(void)
{
    char path[] = "/tmp/ia32XXXXXX";
    int fd = mkstemp64(path); /* O_LARGEFILE */
    CHECK(fd >= 0, "mkstemp");
    /* A file past 4 GiB: ftruncate64 and pread64/pwrite64 with split offsets. */
    off64_t big = 0x100001000LL;
    CHECK(ftruncate64(fd, big + 4096) == 0, "ftruncate64");
    CHECK(pwrite64(fd, "abcd", 4, big) == 4, "pwrite64");
    char buf[8] = "";
    CHECK(pread64(fd, buf, 4, big) == 4 && memcmp(buf, "abcd", 4) == 0, "pread64");
    CHECK(lseek64(fd, big, SEEK_SET) == big, "_llseek");
    struct stat64 st;
    CHECK(fstat64(fd, &st) == 0 && st.st_size == big + 4096, "fstat64 size %lld", (long long)st.st_size);
    CHECK(S_ISREG(st.st_mode) && st.st_nlink == 1, "fstat64 mode %o nlink %u", st.st_mode, (unsigned)st.st_nlink);
    struct stat64 st2;
    CHECK(stat64(path, &st2) == 0 && st2.st_ino == st.st_ino, "stat64 ino");
    CHECK(fstatat64(AT_FDCWD, path, &st2, 0) == 0 && st2.st_size == st.st_size, "fstatat64");
    /* The old stat can't hold that size. */
    struct { char b[64]; } oldst;
    CHECK(syscall(SYS_fstat, fd, &oldst) == -1 && errno == EOVERFLOW, "fstat EOVERFLOW");
    CHECK(ftruncate(fd, 100) == 0, "ftruncate");
    CHECK(syscall(SYS_fstat, fd, &oldst) == 0, "fstat");
    /* readv/writev, preadv/pwritev with 32-bit iovecs. */
    struct iovec iov[2] = {{"hel", 3}, {"lo", 2}};
    CHECK(pwritev(fd, iov, 2, 10) == 5, "pwritev");
    char a[3], b[2];
    struct iovec riov[2] = {{a, 3}, {b, 2}};
    CHECK(preadv(fd, riov, 2, 10) == 5 && memcmp(a, "hel", 3) == 0 && memcmp(b, "lo", 2) == 0, "preadv");
    lseek(fd, 0, SEEK_SET);
    CHECK(writev(fd, iov, 2) == 5, "writev");
    lseek(fd, 0, SEEK_SET);
    CHECK(readv(fd, riov, 2) == 5 && memcmp(a, "hel", 3) == 0, "readv");
    /* fcntl64 record locks. */
    struct flock64 fl = {.l_type = F_WRLCK, .l_whence = SEEK_SET, .l_start = 0x100000000LL, .l_len = 10};
    CHECK(fcntl(fd, F_SETLK64, &fl) == 0, "F_SETLK64");
    pid_t child = fork();
    if (child == 0) {
        int f2 = open(path, O_RDWR);
        struct flock64 q = {.l_type = F_WRLCK, .l_whence = SEEK_SET, .l_start = 0x100000002LL, .l_len = 1};
        int r = fcntl(f2, F_GETLK64, &q);
        _exit(r == 0 && q.l_type == F_WRLCK && q.l_start == 0x100000000LL && q.l_len == 10 &&
                      q.l_pid == getppid()
                  ? 0
                  : 1);
    }
    int status;
    CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "F_GETLK64 in a child: %#x", status);
    struct flock ofl = {.l_type = F_UNLCK, .l_whence = SEEK_SET, .l_start = 0, .l_len = 0};
    CHECK(syscall(SYS_fcntl, fd, F_SETLK, &ofl) == 0, "F_SETLK (compat flock)");
    /* statfs64. */
    struct statfs64 sfs;
    CHECK(fstatfs64(fd, &sfs) == 0 && sfs.f_bsize > 0, "fstatfs64");
    CHECK(syscall(SYS_fstatfs64, fd, 88, &sfs) == -1 && errno == EINVAL, "fstatfs64 size check");
    /* utimes/futimens with 32-bit time structures. */
    struct timeval tv[2] = {{1000000000, 5}, {1100000000, 7}};
    CHECK(utimes(path, tv) == 0, "utimes");
    CHECK(stat64(path, &st) == 0 && st.st_mtime == 1100000000, "utimes mtime %ld", (long)st.st_mtime);
    struct timespec ts[2] = {{0, UTIME_OMIT}, {1200000000, 9}};
    CHECK(syscall(SYS_utimensat, AT_FDCWD, path, ts, 0) == 0, "utimensat");
    CHECK(stat64(path, &st) == 0 && st.st_mtime == 1200000000 && st.st_atime == 1000000000,
          "utimensat times %ld %ld", (long)st.st_mtime, (long)st.st_atime);
    close(fd);
    /* getdents through readdir. */
    char dir[] = "/tmp/ia32dXXXXXX";
    CHECK(mkdtemp(dir) != NULL, "mkdtemp");
    char name[64];
    for (int i = 0; i < 40; i++) {
        snprintf(name, sizeof name, "%s/file-with-a-longish-name-%02d", dir, i);
        close(open(name, O_CREAT | O_WRONLY, 0644));
    }
    DIR *d = opendir(dir);
    int seen = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL)
        if (strncmp(de->d_name, "file-with", 9) == 0)
            seen++;
    closedir(d);
    CHECK(seen == 40, "readdir saw %d", seen);
    /* And the old getdents (32-bit d_ino/d_off). */
    int dfd = open(dir, O_RDONLY | O_DIRECTORY);
    char dbuf[512];
    int total = 0, n;
    while ((n = syscall(SYS_getdents, dfd, dbuf, sizeof dbuf)) > 0) {
        for (int off = 0; off < n;) {
            unsigned short reclen = *(unsigned short *)(dbuf + off + 8);
            if (strncmp(dbuf + off + 10, "file-with", 9) == 0)
                total++;
            CHECK(reclen % 4 == 0 && reclen >= 12, "reclen %u", reclen);
            off += reclen;
        }
    }
    CHECK(n == 0 && total == 40, "getdents saw %d (%d)", total, n);
    close(dfd);
    unlink(path);
}

static void test_time(void)
{
    struct timeval tv;
    CHECK(gettimeofday(&tv, NULL) == 0 && tv.tv_sec > 1500000000, "gettimeofday %ld", (long)tv.tv_sec);
    struct timespec ts;
    CHECK(syscall(SYS_clock_gettime, CLOCK_MONOTONIC, &ts) == 0 && ts.tv_nsec < 1000000000, "clock_gettime");
    struct { long long s; long long ns; } ts64;
    CHECK(syscall(403 /* clock_gettime64 */, CLOCK_REALTIME, &ts64) == 0 && ts64.s > 1500000000,
          "clock_gettime64");
    long t32 = 0;
    CHECK(syscall(SYS_time, &t32) > 1500000000 && t32 > 1500000000, "time");
    struct timespec req = {0, 20000000}, rem;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    CHECK(syscall(SYS_nanosleep, &req, &rem) == 0, "nanosleep");
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long ms = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000;
    CHECK(ms >= 19, "slept %ld ms", ms);
    struct itimerval it = {{0, 0}, {5, 0}}, old;
    CHECK(setitimer(ITIMER_REAL, &it, NULL) == 0, "setitimer");
    CHECK(getitimer(ITIMER_REAL, &old) == 0 && old.it_value.tv_sec >= 4 && old.it_value.tv_sec <= 5,
          "getitimer %ld", (long)old.it_value.tv_sec);
    it.it_value.tv_sec = 0;
    setitimer(ITIMER_REAL, &it, NULL);
    struct tms tm;
    CHECK(times(&tm) != (clock_t)-1, "times");
    struct rusage ru;
    memset(&ru, 0xff, sizeof ru);
    CHECK(getrusage(RUSAGE_SELF, &ru) == 0 && ru.ru_utime.tv_usec < 1000000 && ru.ru_stime.tv_usec < 1000000 &&
              ru.ru_maxrss != -1 && ru.ru_nivcsw != -1,
          "getrusage");
}

static void test_limits_info(void)
{
    struct rlimit rl;
    CHECK(getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur > 0, "getrlimit");
    struct rlimit want = {64, rl.rlim_max};
    CHECK(setrlimit(RLIMIT_NOFILE, &want) == 0, "setrlimit");
    CHECK(getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur == 64, "getrlimit after %lu", (unsigned long)rl.rlim_cur);
    want.rlim_cur = 1024;
    setrlimit(RLIMIT_NOFILE, &want);
    CHECK(getrlimit(RLIMIT_AS, &rl) == 0 && rl.rlim_cur == RLIM_INFINITY, "RLIMIT_AS infinite %#lx",
          (unsigned long)rl.rlim_cur);
    struct sysinfo si;
    CHECK(sysinfo(&si) == 0 && si.totalram > 0 && si.mem_unit > 0 && si.procs > 0, "sysinfo");
    struct utsname u;
    CHECK(uname(&u) == 0 && strcmp(u.machine, "x86_64") == 0, "uname machine %s", u.machine);
    cpu_set_t set;
    CHECK(sched_getaffinity(0, sizeof set, &set) == 0 && CPU_COUNT(&set) >= 1, "sched_getaffinity");
}

static void test_select_poll(void)
{
    int p[2];
    CHECK(pipe(p) == 0, "pipe");
    fd_set r;
    FD_ZERO(&r);
    FD_SET(p[0], &r);
    struct timeval tv = {0, 10000};
    CHECK(select(p[0] + 1, &r, NULL, NULL, &tv) == 0, "select timeout");
    CHECK(write(p[1], "x", 1) == 1, "write");
    FD_SET(p[0], &r);
    tv.tv_sec = 1;
    CHECK(select(p[0] + 1, &r, NULL, NULL, &tv) == 1 && FD_ISSET(p[0], &r), "select ready");
    struct timespec ts = {1, 0};
    sigset_t mask;
    sigemptyset(&mask);
    FD_SET(p[0], &r);
    CHECK(pselect(p[0] + 1, &r, NULL, NULL, &ts, &mask) == 1, "pselect");
    struct pollfd pf = {p[0], POLLIN, 0};
    CHECK(ppoll(&pf, 1, &ts, &mask) == 1 && pf.revents == POLLIN, "ppoll");
    close(p[0]);
    close(p[1]);
}

static void test_sockets(void)
{
    int sv[2], dg[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "socketpair");
    CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, dg) == 0, "socketpair dgram");
    /* SCM_RIGHTS with a 32-bit cmsghdr. */
    int p[2];
    pipe(p);
    char data[] = "fd";
    struct iovec io = {data, 2};
    union { char buf[CMSG_SPACE(sizeof(int))]; struct cmsghdr align; } u;
    struct msghdr m = {0};
    m.msg_iov = &io;
    m.msg_iovlen = 1;
    m.msg_control = u.buf;
    m.msg_controllen = sizeof u.buf;
    struct cmsghdr *c = CMSG_FIRSTHDR(&m);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(c), &p[1], sizeof(int));
    CHECK(sendmsg(sv[0], &m, 0) == 2, "sendmsg");
    char got[4] = "";
    struct iovec rio = {got, sizeof got};
    union { char buf[CMSG_SPACE(sizeof(int))]; struct cmsghdr align; } ru;
    struct msghdr rm = {0};
    rm.msg_iov = &rio;
    rm.msg_iovlen = 1;
    rm.msg_control = ru.buf;
    rm.msg_controllen = sizeof ru.buf;
    CHECK(recvmsg(sv[1], &rm, 0) == 2 && memcmp(got, "fd", 2) == 0, "recvmsg");
    struct cmsghdr *rc = CMSG_FIRSTHDR(&rm);
    CHECK(rc && rc->cmsg_type == SCM_RIGHTS && rc->cmsg_len == CMSG_LEN(sizeof(int)), "cmsg %p", (void *)rc);
    CHECK(rm.msg_controllen == CMSG_SPACE(sizeof(int)), "controllen %u", (unsigned)rm.msg_controllen);
    if (rc) {
        int nfd;
        memcpy(&nfd, CMSG_DATA(rc), sizeof nfd);
        CHECK(write(nfd, "z", 1) == 1, "passed fd writes");
        char z;
        CHECK(read(p[0], &z, 1) == 1 && z == 'z', "passed fd is the pipe");
        close(nfd);
    }
    /* sendmmsg/recvmmsg: 32-byte mmsghdrs. */
    struct mmsghdr mm[3];
    struct iovec mio[3];
    char *msgs[3] = {"one", "two", "three"};
    memset(mm, 0, sizeof mm);
    for (int i = 0; i < 3; i++) {
        mio[i].iov_base = msgs[i];
        mio[i].iov_len = strlen(msgs[i]);
        mm[i].msg_hdr.msg_iov = &mio[i];
        mm[i].msg_hdr.msg_iovlen = 1;
    }
    CHECK(sendmmsg(dg[0], mm, 3, 0) == 3 && mm[2].msg_len == 5, "sendmmsg");
    char rb[3][8];
    struct iovec rv[3];
    struct mmsghdr rmm[3];
    memset(rmm, 0, sizeof rmm);
    for (int i = 0; i < 3; i++) {
        rv[i].iov_base = rb[i];
        rv[i].iov_len = 8;
        rmm[i].msg_hdr.msg_iov = &rv[i];
        rmm[i].msg_hdr.msg_iovlen = 1;
    }
    struct timespec to = {1, 0};
    int n = recvmmsg(dg[1], rmm, 3, 0, &to);
    CHECK(n == 3 && rmm[1].msg_len == 3 && memcmp(rb[2], "three", 5) == 0, "recvmmsg %d", n);
    /* socketcall, as old i386 code makes every socket call. */
    unsigned long args[4] = {AF_UNIX, SOCK_STREAM, 0, (unsigned long)sv};
    int sp2[2];
    args[3] = (unsigned long)sp2;
    CHECK(syscall(SYS_socketcall, 8 /* SYS_SOCKETPAIR */, args) == 0, "socketcall socketpair");
    unsigned long sargs[4] = {sp2[0], (unsigned long)"hi", 2, 0};
    CHECK(syscall(SYS_socketcall, 9 /* SYS_SEND */, sargs) == 2, "socketcall send");
    char hb[2];
    unsigned long rargs[4] = {sp2[1], (unsigned long)hb, 2, 0};
    CHECK(syscall(SYS_socketcall, 10 /* SYS_RECV */, rargs) == 2 && hb[0] == 'h', "socketcall recv");
    /* SO_RCVTIMEO with a 32-bit timeval. */
    struct timeval tv = {2, 500000}, tg;
    socklen_t len = sizeof tg;
    CHECK(setsockopt(sp2[0], SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv) == 0, "SO_RCVTIMEO set");
    CHECK(getsockopt(sp2[0], SOL_SOCKET, SO_RCVTIMEO, &tg, &len) == 0 && len == 8 && tg.tv_sec == 2 &&
              tg.tv_usec >= 400000 && tg.tv_usec < 600000,
          "SO_RCVTIMEO get %ld.%06ld len %u", (long)tg.tv_sec, (long)tg.tv_usec, (unsigned)len);
    close(sp2[0]);
    close(sp2[1]);
    close(sv[0]);
    close(sv[1]);
    close(dg[0]);
    close(dg[1]);
    close(p[0]);
    close(p[1]);
}

static void test_ipc(void)
{
    int shm = shmget(IPC_PRIVATE, 8192, IPC_CREAT | 0600);
    CHECK(shm >= 0, "shmget");
    char *at = shmat(shm, NULL, 0);
    CHECK(at != (void *)-1 && (uintptr_t)at < 0xffffe000u, "shmat %p", at);
    struct shmid_ds ds;
    CHECK(shmctl(shm, IPC_STAT, &ds) == 0 && ds.shm_segsz == 8192 && ds.shm_nattch == 1 &&
              ds.shm_cpid == getpid(),
          "shmctl IPC_STAT segsz %u nattch %lu cpid %d", (unsigned)ds.shm_segsz,
          (unsigned long)ds.shm_nattch, ds.shm_cpid);
    shmdt(at);
    shmctl(shm, IPC_RMID, NULL);
    int q = msgget(IPC_PRIVATE, IPC_CREAT | 0600);
    CHECK(q >= 0, "msgget");
    struct { long type; char text[16]; } msg = {7, "hello"}, in = {0, ""};
    CHECK(msgsnd(q, &msg, 6, 0) == 0, "msgsnd");
    struct msqid_ds mq;
    CHECK(msgctl(q, IPC_STAT, &mq) == 0 && mq.msg_qnum == 1, "msgctl IPC_STAT qnum %lu", (unsigned long)mq.msg_qnum);
    CHECK(msgrcv(q, &in, sizeof in.text, 7, 0) == 6 && in.type == 7 && strcmp(in.text, "hello") == 0,
          "msgrcv type %ld %s", in.type, in.text);
    msgctl(q, IPC_RMID, NULL);
    int sem = semget(IPC_PRIVATE, 2, IPC_CREAT | 0600);
    CHECK(sem >= 0, "semget");
    CHECK(semctl(sem, 1, SETVAL, 5) == 0 && semctl(sem, 1, GETVAL) == 5, "semctl SETVAL/GETVAL");
    struct sembuf op = {1, -2, 0};
    CHECK(semop(sem, &op, 1) == 0 && semctl(sem, 1, GETVAL) == 3, "semop");
    struct semid_ds sds;
    CHECK(semctl(sem, 0, IPC_STAT, &sds) == 0 && sds.sem_nsems == 2, "semctl IPC_STAT nsems %lu",
          (unsigned long)sds.sem_nsems);
    semctl(sem, 0, IPC_RMID);
    /* The ipc multiplexer, as old code calls it: SHMGET then SHMCTL. */
    long id = syscall(SYS_ipc, 23 /* SHMGET */, IPC_PRIVATE, 4096, IPC_CREAT | 0600, 0, 0);
    CHECK(id >= 0, "ipc SHMGET");
    CHECK(syscall(SYS_ipc, 24 /* SHMCTL */, id, IPC_RMID | 0x100 /* IPC_64 */, 0, NULL, 0) == 0, "ipc SHMCTL");
}

static void test_mmap(void)
{
    char path[] = "/tmp/ia32mXXXXXX";
    int fd = mkstemp(path);
    CHECK(ftruncate(fd, 3 * 4096) == 0, "ftruncate");
    CHECK(pwrite(fd, "page2", 5, 2 * 4096) == 5, "pwrite");
    /* mmap2: the offset is in pages. */
    char *p = (char *)syscall(SYS_mmap2, 0, 4096, PROT_READ, MAP_PRIVATE, fd, 2);
    CHECK(p != MAP_FAILED && memcmp(p, "page2", 5) == 0, "mmap2 offset");
    CHECK((uintptr_t)p < 0xffffe000u, "below 4 GiB");
    munmap(p, 4096);
    /* old_mmap: its arguments in memory, the offset in bytes. */
    unsigned long args[6] = {0, 4096, PROT_READ, MAP_PRIVATE, (unsigned long)fd, 2 * 4096};
    p = (char *)syscall(SYS_mmap, args);
    CHECK(p != MAP_FAILED && memcmp(p, "page2", 5) == 0, "old_mmap");
    munmap(p, 4096);
    args[5] = 100;
    CHECK(syscall(SYS_mmap, args) == -1 && errno == EINVAL, "old_mmap unaligned");
    close(fd);
    unlink(path);
    /* A big anonymous mapping, all below 4 GiB. */
    char *big = mmap(NULL, 256 << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(big != MAP_FAILED, "mmap 256 MiB");
    if (big != MAP_FAILED) {
        big[0] = big[(256 << 20) - 1] = 1;
        munmap(big, 256 << 20);
    }
}

static volatile int got_sig, got_code;
static volatile unsigned long got_eip, got_addr, got_value;
static sigjmp_buf jb;

static void on_info(int sig, siginfo_t *si, void *ctx)
{
    ucontext_t *uc = ctx;
    got_sig = sig;
    got_code = si->si_code;
    got_addr = (unsigned long)si->si_addr;
    got_value = si->si_value.sival_int;
    got_eip = uc->uc_mcontext.gregs[REG_EIP];
    unsigned short cs;
    __asm__("mov %%cs, %0" : "=r"(cs));
    if (cs != 0x23)
        got_sig = -1;
    if (sig == SIGSEGV)
        siglongjmp(jb, 1);
}

static void on_plain(int sig)
{
    got_sig = sig;
}

static void on_alt(int sig)
{
    int here;
    got_addr = (unsigned long)&here;
    got_sig = sig;
}

static void test_signals(void)
{
    struct sigaction sa = {0};
    sa.sa_sigaction = on_info;
    sa.sa_flags = SA_SIGINFO;
    CHECK(sigaction(SIGUSR1, &sa, NULL) == 0, "sigaction");
    union sigval v = {.sival_int = 1234};
    CHECK(sigqueue(getpid(), SIGUSR1, v) == 0, "sigqueue");
    CHECK(got_sig == SIGUSR1 && got_code == SI_QUEUE && got_value == 1234, "rt frame sig %d code %d value %lu",
          got_sig, got_code, got_value);
    /* A fault, recovered with siglongjmp: si_addr and the saved %eip. */
    sigaction(SIGSEGV, &sa, NULL);
    got_sig = 0;
    if (sigsetjmp(jb, 1) == 0) {
        *(volatile int *)0x10 = 1;
    }
    CHECK(got_sig == SIGSEGV && got_addr == 0x10 && got_eip != 0, "SIGSEGV addr %#lx eip %#lx", got_addr,
          got_eip);
    /* A non-rt (sigframe_ia32) handler, through the old sigaction. */
    struct { unsigned long handler, mask, flags, restorer; } old = {(unsigned long)on_plain, 0, 0, 0};
    CHECK(syscall(SYS_sigaction, SIGUSR2, &old, NULL) == 0, "old sigaction");
    got_sig = 0;
    kill(getpid(), SIGUSR2);
    CHECK(got_sig == SIGUSR2, "sigframe handler %d", got_sig);
    struct { unsigned long handler, mask, flags, restorer; } back;
    CHECK(syscall(SYS_sigaction, SIGUSR2, NULL, &back) == 0 && back.handler == (unsigned long)on_plain,
          "old sigaction get");
    /* signal(): one-shot. */
    CHECK(syscall(SYS_signal, SIGUSR2, (unsigned long)on_plain) == (long)on_plain, "signal old");
    got_sig = 0;
    kill(getpid(), SIGUSR2);
    CHECK(got_sig == SIGUSR2, "signal handler");
    /* sigprocmask on the first 32 signals; the high ones stay. */
    sigset_t s, o;
    sigemptyset(&s);
    sigaddset(&s, SIGRTMIN + 5);
    sigprocmask(SIG_BLOCK, &s, NULL);
    unsigned long m = 1u << (SIGUSR1 - 1), om = 0;
    CHECK(syscall(SYS_sigprocmask, SIG_SETMASK, &m, &om) == 0, "old sigprocmask");
    sigprocmask(SIG_BLOCK, NULL, &o);
    CHECK(sigismember(&o, SIGUSR1) && sigismember(&o, SIGRTMIN + 5), "old SETMASK keeps high signals");
    kill(getpid(), SIGUSR1);
    unsigned long pend = 0;
    CHECK(syscall(SYS_sigpending, &pend) == 0 && (pend & m), "sigpending %#lx", pend);
    got_sig = 0;
    sigemptyset(&s);
    sigprocmask(SIG_SETMASK, &s, NULL);
    CHECK(got_sig == SIGUSR1, "unblocked delivery");
    /* An alternate signal stack (32-bit stack_t). */
    static char altbuf[65536];
    stack_t ss = {.ss_sp = altbuf, .ss_size = sizeof altbuf, .ss_flags = 0}, oss;
    CHECK(sigaltstack(&ss, NULL) == 0, "sigaltstack");
    CHECK(sigaltstack(NULL, &oss) == 0 && oss.ss_sp == altbuf && oss.ss_size == sizeof altbuf, "sigaltstack get");
    struct sigaction alt = {0};
    alt.sa_handler = on_alt;
    alt.sa_flags = SA_ONSTACK;
    sigaction(SIGUSR2, &alt, NULL);
    got_sig = 0;
    kill(getpid(), SIGUSR2);
    CHECK(got_sig == SIGUSR2 && got_addr >= (unsigned long)altbuf && got_addr < (unsigned long)altbuf + sizeof altbuf,
          "ran on the alternate stack %#lx", got_addr);
    /* rt_sigtimedwait's 32-bit siginfo. */
    sigemptyset(&s);
    sigaddset(&s, SIGRTMIN);
    sigprocmask(SIG_BLOCK, &s, NULL);
    v.sival_int = 99;
    sigqueue(getpid(), SIGRTMIN, v);
    siginfo_t info;
    struct timespec to = {1, 0};
    CHECK(sigtimedwait(&s, &info, &to) == SIGRTMIN && info.si_value.sival_int == 99 && info.si_pid == getpid(),
          "sigtimedwait value %d pid %d", info.si_value.sival_int, info.si_pid);
}

static void test_processes(char *self)
{
    pid_t c = fork();
    if (c == 0)
        _exit(42);
    siginfo_t si = {0};
    CHECK(waitid(P_PID, c, &si, WEXITED) == 0 && si.si_pid == c && si.si_code == CLD_EXITED && si.si_status == 42,
          "waitid pid %d code %d status %d", si.si_pid, si.si_code, si.si_status);
    c = fork();
    if (c == 0)
        _exit(3);
    int status;
    struct rusage ru;
    CHECK(wait4(c, &status, 0, &ru) == c && WEXITSTATUS(status) == 3, "wait4");
    /* execve of this program, with arguments and environment (32-bit
     * pointer arrays). */
    c = fork();
    if (c == 0) {
        char *argv[] = {self, "exec-check", "second", NULL};
        char *envp[] = {"IA32_ENV=yes", NULL};
        execve(self, argv, envp);
        _exit(99);
    }
    CHECK(waitpid(c, &status, 0) == c && WIFEXITED(status) && WEXITSTATUS(status) == 77,
          "exec'd child status %#x", status);
}

static __thread int tls_value = 5;
static void *thread_main(void *arg)
{
    tls_value = (int)(intptr_t)arg;
    unsigned short gs;
    __asm__("mov %%gs, %0" : "=r"(gs));
    sched_yield();
    return (void *)(intptr_t)(tls_value * 10 + ((gs & 3) == 3));
}

static void test_threads(void)
{
    pthread_t t[4];
    for (int i = 0; i < 4; i++)
        CHECK(pthread_create(&t[i], NULL, thread_main, (void *)(intptr_t)(i + 1)) == 0, "pthread_create");
    for (int i = 0; i < 4; i++) {
        void *r;
        pthread_join(t[i], &r);
        CHECK((intptr_t)r == (i + 1) * 10 + 1, "thread %d returned %ld", i, (long)(intptr_t)r);
    }
    CHECK(tls_value == 5, "main TLS %d", tls_value);
    /* A futex wait with a 32-bit timeout. */
    int word = 0;
    struct timespec to = {0, 10000000};
    CHECK(syscall(SYS_futex, &word, FUTEX_WAIT, 0, &to, NULL, 0) == -1 && errno == ETIMEDOUT, "futex timeout");
}

int main(int argc, char **argv)
{
    if (argc >= 3 && strcmp(argv[1], "exec-check") == 0) {
        const char *e = getenv("IA32_ENV");
        return strcmp(argv[2], "second") == 0 && e && strcmp(e, "yes") == 0 ? 77 : 1;
    }
    setvbuf(stdout, NULL, _IONBF, 0);
    /* `ia32 only NAME`: just that test (for debugging). */
    if (argc >= 3 && strcmp(argv[1], "only") == 0) {
        if (strcmp(argv[2], "sockets") == 0)
            test_sockets();
        if (strcmp(argv[2], "signals") == 0)
            test_signals();
        if (strcmp(argv[2], "files") == 0)
            test_files();
        printf(failures ? "FAILED\n" : "PASS\n");
        return failures != 0;
    }
    test_mode();
    test_auxv_vdso();
    test_files();
    test_time();
    test_limits_info();
    test_select_poll();
    test_sockets();
    test_ipc();
    test_mmap();
    test_signals();
    test_processes(argv[0]);
    test_threads();
    if (failures == 0)
        printf("PASS\n");
    return failures != 0;
}
