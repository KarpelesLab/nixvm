/*
 * x86-64 differential-testing oracle for nixvm's software interpreter
 * (src/vcpu/interp_x86.rs). Built by tests/x86_diff.rs with
 * `clang -arch x86_64` and run under Rosetta 2 (`arch -x86_64`) on Apple
 * silicon, or natively on an x86-64 macOS host.
 *
 * Protocol (all little-endian, over stdin/stdout):
 *   startup  -> u64 magic, u64 insn_addr, u64 data_addr, u64 data_len
 *   per case <- u32 code_len (bit 31: MMX mode), u8 code[code_len], u64 gpr[16], u64 rflags,
 *               u8 fxsave[512], u8 ymm_hi[256], u8 data[data_len]
 *            -> i32 signo, i32 si_code, u64 si_addr, u64 gpr[16], u64 rflags,
 *               u8 fxsave[512], u8 ymm_hi[256], u8 data[data_len]
 *
 * The case's instruction bytes are copied to a fixed address (insn_addr),
 * between a hand-assembled prologue (FXRSTOR64 the x87/SSE image, VINSERTF128
 * the YMM upper halves, POPFQ the flags, load all sixteen GPRs including RSP)
 * and epilogue (store all GPRs RIP-relative, PUSHFQ on a private stack,
 * FXSAVE64, VEXTRACTF128 the upper halves), so the instruction runs
 * with exactly the requested architectural state and its RIP-relative
 * operands resolve identically to the interpreter's. A fault (SIGSEGV,
 * SIGILL, SIGFPE, SIGTRAP, SIGBUS) is caught on an alternate stack and
 * reported instead of the register state.
 */
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define BASE 0x600000000ULL
#define CODE (BASE)
#define PRIV (BASE + 0x4000)
#define DATA (BASE + 0x8000)
#define DATA_LEN 0x2000
#define MAP_LEN 0x10000

/* Offsets inside PRIV. */
#define P_HOST_RSP 0x000
#define P_IN_RFLAGS 0x008
#define P_OUT_RFLAGS 0x010
#define P_IN_GPR 0x100
#define P_OUT_GPR 0x200
#define P_IN_FX 0x400
#define P_OUT_FX 0x600
#define P_OUT_MM 0x800
#define P_IN_YMM 0xA00
#define P_OUT_YMM 0xB00
#define P_STACK_TOP 0x1000

static sigjmp_buf env;
static volatile int got_sig, got_code;
static volatile uint64_t got_addr;

static void on_signal(int sig, siginfo_t *si, void *uc) {
  (void)uc;
  got_sig = sig;
  got_code = si->si_code;
  got_addr = (uint64_t)si->si_addr;
  siglongjmp(env, 1);
}

static uint8_t *emit_rip(uint8_t *p, const uint8_t *op, int oplen, uint64_t target) {
  memcpy(p, op, oplen);
  p += oplen;
  int32_t d = (int32_t)(int64_t)(target - ((uint64_t)p + 4));
  memcpy(p, &d, 4);
  return p + 4;
}

/* `mov [rip+d], r` (store=1) or `mov r, [rip+d]` (store=0). */
static uint8_t *mov_rip(uint8_t *p, int r, int store, uint64_t target) {
  uint8_t op[3] = {(uint8_t)(0x48 | (r >= 8 ? 4 : 0)), store ? 0x89 : 0x8B,
                   (uint8_t)(((r & 7) << 3) | 5)};
  return emit_rip(p, op, 3, target);
}

/* VINSERTF128 ymmN, ymmN, [rip+d], 1 (load=1) or VEXTRACTF128 [rip+d], ymmN, 1. */
static uint8_t *ymm_hi_rip(uint8_t *p, int n, int load, uint64_t target) {
  uint8_t op[5] = {0xC4, (uint8_t)((n >= 8 ? 0 : 0x80) | 0x63),
                   (uint8_t)((load ? ((~n & 15) << 3) : 0x78) | 0x05), load ? 0x18 : 0x19,
                   (uint8_t)(((n & 7) << 3) | 5)};
  /* The displacement is relative to the end of the instruction, after imm8. */
  p = emit_rip(p, op, 5, target - 1);
  *p++ = 1;
  return p;
}

static size_t prologue_len;
static size_t mmx_loads_at;

/* MMX mode: Rosetta keeps the MMX registers apart from the x87 register file
 * FXSAVE/FXRSTOR see, so they are loaded (from the FXSAVE slots) and stored
 * explicitly with MOVQ; otherwise those 8 MOVQs are same-length NOPs. */
static void set_mmx_loads(int mmx) {
  uint8_t *p = (uint8_t *)CODE + mmx_loads_at;
  for (int r = 0; r < 8; r++) {
    if (mmx) {
      const uint8_t movq[] = {0x0F, 0x6F, (uint8_t)((r << 3) | 5)};
      p = emit_rip(p, movq, 3, PRIV + P_IN_FX + 32 + 16 * r);
    } else {
      const uint8_t nop7[] = {0x0F, 0x1F, 0x80, 0, 0, 0, 0};
      memcpy(p, nop7, 7);
      p += 7;
    }
  }
}

static void build_prologue(void) {
  uint8_t *p = (uint8_t *)CODE;
  const uint8_t saves[] = {0x53, 0x55, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57};
  memcpy(p, saves, sizeof saves);
  p += sizeof saves;
  p = mov_rip(p, 4, 1, PRIV + P_HOST_RSP);
  const uint8_t fxrstor[] = {0x48, 0x0F, 0xAE, 0x0D};
  p = emit_rip(p, fxrstor, 4, PRIV + P_IN_FX);
  for (int r = 0; r < 16; r++) p = ymm_hi_rip(p, r, 1, PRIV + P_IN_YMM + 16 * r);
  mmx_loads_at = (size_t)(p - (uint8_t *)CODE);
  p += 8 * 7;
  const uint8_t pushm[] = {0xFF, 0x35};
  p = emit_rip(p, pushm, 2, PRIV + P_IN_RFLAGS);
  *p++ = 0x9D; /* popfq */
  for (int r = 0; r < 16; r++)
    if (r != 4) p = mov_rip(p, r, 0, PRIV + P_IN_GPR + 8 * r);
  p = mov_rip(p, 4, 0, PRIV + P_IN_GPR + 8 * 4);
  prologue_len = (size_t)(p - (uint8_t *)CODE);
}

static void build_epilogue(uint8_t *p, int mmx) {
  for (int r = 0; r < 16; r++) p = mov_rip(p, r, 1, PRIV + P_OUT_GPR + 8 * r);
  const uint8_t lea[] = {0x48, 0x8D, 0x25};
  p = emit_rip(p, lea, 3, PRIV + P_STACK_TOP);
  *p++ = 0x9C; /* pushfq */
  const uint8_t popm[] = {0x8F, 0x05};
  p = emit_rip(p, popm, 2, PRIV + P_OUT_RFLAGS);
  const uint8_t fxsave[] = {0x48, 0x0F, 0xAE, 0x05};
  p = emit_rip(p, fxsave, 4, PRIV + P_OUT_FX);
  for (int r = 0; r < 16; r++) p = ymm_hi_rip(p, r, 0, PRIV + P_OUT_YMM + 16 * r);
  if (mmx)
    for (int r = 0; r < 8; r++) {
      const uint8_t movq[] = {0x0F, 0x7F, (uint8_t)((r << 3) | 5)};
      p = emit_rip(p, movq, 3, PRIV + P_OUT_MM + 8 * r);
    }
  p = mov_rip(p, 4, 0, PRIV + P_HOST_RSP);
  const uint8_t restores[] = {0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5D, 0x5B, 0xC3};
  memcpy(p, restores, sizeof restores);
}

static int read_full(void *buf, size_t n) {
  uint8_t *b = buf;
  while (n) {
    size_t k = fread(b, 1, n, stdin);
    if (k == 0) return 0;
    b += k;
    n -= k;
  }
  return 1;
}

static void write_full(const void *buf, size_t n) { fwrite(buf, 1, n, stdout); }

int main(void) {
  void *m = mmap((void *)BASE, MAP_LEN, PROT_READ | PROT_WRITE | PROT_EXEC,
                 MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
  if (m == MAP_FAILED) {
    perror("mmap");
    return 1;
  }
  static uint8_t altstack[1 << 16];
  stack_t ss = {.ss_sp = altstack, .ss_size = sizeof altstack, .ss_flags = 0};
  sigaltstack(&ss, NULL);
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = on_signal;
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
  sigemptyset(&sa.sa_mask);
  int sigs[] = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP, SIGALRM};
  for (unsigned i = 0; i < sizeof sigs / sizeof sigs[0]; i++) sigaction(sigs[i], &sa, NULL);

  build_prologue();
  uint64_t hello[4] = {0x6E69786F72616C65ULL, CODE + prologue_len, DATA, DATA_LEN};
  write_full(hello, sizeof hello);
  fflush(stdout);

  uint8_t *priv = (uint8_t *)PRIV;
  for (;;) {
    uint32_t len;
    uint8_t code[64];
    if (!read_full(&len, 4)) break;
    int mmx = (len >> 31) & 1;
    len &= 0x7fffffff;
    if (len > sizeof code || !read_full(code, len)) break;
    if (!read_full(priv + P_IN_GPR, 128) || !read_full(priv + P_IN_RFLAGS, 8) ||
        !read_full(priv + P_IN_FX, 512) || !read_full(priv + P_IN_YMM, 256) ||
        !read_full((void *)DATA, DATA_LEN))
      break;
    uint8_t *insn = (uint8_t *)CODE + prologue_len;
    memcpy(insn, code, len);
    build_epilogue(insn + len, mmx);
    set_mmx_loads(mmx);
    /* Poison the outputs so a skipped store is visible. */
    memset(priv + P_OUT_GPR, 0xA5, 128);
    memset(priv + P_OUT_FX, 0xA5, 512);
    memset(priv + P_OUT_YMM, 0xA5, 256);
    got_sig = 0;
    got_code = 0;
    got_addr = 0;
    if (sigsetjmp(env, 1) == 0) {
      /* A watchdog: an instruction that never completes reports SIGALRM. */
      alarm(2);
      ((void (*)(void))CODE)();
      alarm(0);
    } else {
      alarm(0);
      __asm__ volatile("cld; fninit");
      uint32_t mx = 0x1f80;
      __asm__ volatile("ldmxcsr %0" ::"m"(mx));
    }
    if (mmx && !got_sig) {
      /* Report the MMX registers in their FXSAVE slots (TOP = 0). */
      for (int r = 0; r < 8; r++) {
        memcpy(priv + P_OUT_FX + 32 + 16 * r, priv + P_OUT_MM + 8 * r, 8);
        priv[P_OUT_FX + 32 + 16 * r + 8] = 0xff;
        priv[P_OUT_FX + 32 + 16 * r + 9] = 0xff;
      }
    }
    int32_t hdr[2] = {got_sig, got_code};
    uint64_t addr = got_addr;
    write_full(hdr, 8);
    write_full(&addr, 8);
    write_full(priv + P_OUT_GPR, 128);
    write_full(priv + P_OUT_RFLAGS, 8);
    write_full(priv + P_OUT_FX, 512);
    write_full(priv + P_OUT_YMM, 256);
    write_full((void *)DATA, DATA_LEN);
    fflush(stdout);
  }
  return 0;
}
