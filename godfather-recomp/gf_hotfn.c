#define _CRT_SECURE_NO_WARNINGS
#include "gf_hotfn.h"

/* v1129N: no FP contraction anywhere in this file. The helpers below must give the bits of the lifted code, which is
 * compiled with -ffp-contract=off; with the AVX2/FMA target attribute (further down) the compiler could otherwise fuse
 * a*b+c into one vfmadd and change results. */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern uint8_t* vm_base;
extern volatile int g_v243a_sel;
extern unsigned char g_v243a_stflag[];
extern PPU_THREAD_LOCAL void (*g_trampoline_fn)(void*);
extern void ps3_indirect_call(ppu_context* ctx);

#define DRAIN_TRAMPOLINE(ctx) do { \
    while (g_trampoline_fn) { \
        void (*fn)(void*) = g_trampoline_fn; \
        g_trampoline_fn = 0; \
        fn((void*)(ctx)); \
    } \
} while (0)

int gf_hotfn_on = 0;

enum {
    GF_HOTFN_DISABLE_00696704 = 1u << 0,
    GF_HOTFN_DISABLE_0070AF14 = 1u << 1,
    GF_HOTFN_DISABLE_00691358 = 1u << 2,
    GF_HOTFN_DISABLE_00696784 = 1u << 3,
    GF_HOTFN_DISABLE_006A9400 = 1u << 4
};

static uint64_t s_gf_hotfn_mask;

/* v1129N GF_HOTFN2: ENABLE mask (code default 0 = old behaviour) for the register-promoted lifted bodies at the end. */
enum {
    GF_HOTFN2_0054665C = 1u << 0,
    GF_HOTFN2_0070AF14 = 1u << 1,
    GF_HOTFN2_LEANNAN  = 1u << 2,   /* v1129K: bits 0/1 bodies with one NaN test per op (gf_hot3_*) */
};
static uint64_t s_gf_hotfn2;
static int gf_hot3_0054665C(ppu_context* ctx);   /* v1129K */
static int gf_hot3_0070AF14(ppu_context* ctx);   /* v1129K */
static void gf_hotfn2_resolve(void);

/* v1129N: gf_common is compiled for baseline x86-64 (only the lifted gf_ppu objects get PORT_PPU_ISA's
 * -mavx2 -mfma -mbmi -mbmi2 -mlzcnt -mmovbe). Built that way every fma() of the helpers below was a CALL into the
 * C runtime and every guest load a load + bswap, so the helpers ran slower than the lifted code they replace
 * (0070AF14: 24 fma calls per 4x4 product, 006A9400: 40). The helpers are now compiled for the same CPU floor as the
 * lifted code (fma/movbe inline, results bit-identical: fma() is the exact IEEE operation either way, contraction is
 * off). gf_common is shared with the -compat exe, so the switch only turns on where the CPU has all of it. */
#if defined(__clang__) && (defined(__x86_64__) || defined(_M_X64)) && !defined(GF_HOTFN_NO_ISA)
#define GF_HOTFN_ISA 1
#if defined(_MSC_VER)
#include <intrin.h>
__attribute__((target("xsave"))) static unsigned long long gf_hotfn_xgetbv0(void) { return _xgetbv(0); }
static void gf_hotfn_cpuid(int r[4], int leaf, int sub) { __cpuidex(r, leaf, sub); }
#else
#include <cpuid.h>
static unsigned long long gf_hotfn_xgetbv0(void)
{
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
}
static void gf_hotfn_cpuid(int r[4], int leaf, int sub)
{
    unsigned a, b, c, d;
    __cpuid_count((unsigned)leaf, (unsigned)sub, a, b, c, d);
    r[0] = (int)a; r[1] = (int)b; r[2] = (int)c; r[3] = (int)d;
}
#endif
static int gf_hotfn_cpu_ok(void)
{
    int r[4];
    gf_hotfn_cpuid(r, 0, 0);
    if (r[0] < 7) return 0;
    gf_hotfn_cpuid(r, 1, 0);
    if (!((r[2] >> 12) & 1) || !((r[2] >> 22) & 1) || !((r[2] >> 27) & 1) || !((r[2] >> 28) & 1)) return 0; /* FMA MOVBE OSXSAVE AVX */
    if ((gf_hotfn_xgetbv0() & 6) != 6) return 0;
    gf_hotfn_cpuid(r, 7, 0);
    if (!((r[1] >> 5) & 1) || !((r[1] >> 3) & 1) || !((r[1] >> 8) & 1)) return 0;                           /* AVX2 BMI1 BMI2 */
    gf_hotfn_cpuid(r, (int)0x80000000, 0);
    if ((unsigned)r[0] < 0x80000001u) return 0;
    gf_hotfn_cpuid(r, (int)0x80000001, 0);
    return (r[2] >> 5) & 1;                                                                                   /* LZCNT */
}
#else
#define GF_HOTFN_ISA 0
static int gf_hotfn_cpu_ok(void) { return 1; }
#endif

static void gf_hotfn_init(void)
{
    const char* enabled = getenv("GF_HOTFN");
    const char* mask = getenv("GF_HOTFN_MASK");
    const char* en2 = getenv("GF_HOTFN2");
    if (mask && *mask) s_gf_hotfn_mask = strtoull(mask, NULL, 0);
    if (en2 && *en2) s_gf_hotfn2 = strtoull(en2, NULL, 0);
    if (s_gf_hotfn2 && gf_hotfn_cpu_ok()) gf_hotfn2_resolve();   /* callee pointers, before any helper can run */
    if (s_gf_hotfn2 && !(enabled && *enabled && strtol(enabled, NULL, 0) != 0)) {
        /* GF_HOTFN2 alone: the hooks must run, but none of the GF_HOTFN helpers. */
        if (!gf_hotfn_cpu_ok()) {
            fprintf(stderr, "[v1129N] GF_HOTFN2 ignored: CPU lacks AVX2/FMA/BMI/LZCNT/MOVBE (helpers need them)\n");
            fflush(stderr);
            s_gf_hotfn2 = 0;
            return;
        }
        s_gf_hotfn_mask = ~0ull;
        gf_hotfn_on = 1;
        fprintf(stderr, "[v1129N] GF_HOTFN2=0x%llX register-promoted lifted bodies ON (GF_HOTFN helpers off)\n",
                (unsigned long long)s_gf_hotfn2);
        fflush(stderr);
        return;
    }
    if (enabled && *enabled && strtol(enabled, NULL, 0) != 0) {
        if (!gf_hotfn_cpu_ok()) {
            fprintf(stderr, "[v1129N] GF_HOTFN ignored: CPU lacks AVX2/FMA/BMI/LZCNT/MOVBE (helpers need them)\n");
            fflush(stderr);
            return;
        }
        gf_hotfn_on = 1;
        fprintf(stderr, "[v1129N] GF_HOTFN helpers built for AVX2/FMA/MOVBE (inline fma, same as the lifted code): %s\n",
                GF_HOTFN_ISA ? "yes" : "no");
        fprintf(stderr, "[v1121F] native hot functions ON (GF_HOTFN_MASK=0x%llX)\n",
                (unsigned long long)s_gf_hotfn_mask);
        fprintf(stderr, "[v1129M] four-function native hot batch ON (bits 1-4; mask disables)\n");
        if (s_gf_hotfn2)
            fprintf(stderr, "[v1129N] GF_HOTFN2=0x%llX register-promoted lifted bodies ON\n", (unsigned long long)s_gf_hotfn2);
        fflush(stderr);
    }
}

#if defined(_MSC_VER)
#pragma section(".CRT$XCU", read)
__declspec(allocate(".CRT$XCU"))
__attribute__((used)) static void (__cdecl* s_gf_hotfn_init)(void) = gf_hotfn_init;
#else
__attribute__((constructor))
static void gf_hotfn_ctor(void)
{
    gf_hotfn_init();
}
#endif

/* v1129N: everything from here to the end of the file (the helpers and the gf_hot_* bodies) is compiled for the CPU
 * floor of the lifted code; gf_hotfn_init above stays baseline and keeps gf_hotfn_on 0 on other CPUs. */
#if GF_HOTFN_ISA
#pragma clang attribute push (__attribute__((target("avx2,fma,bmi,bmi2,lzcnt,movbe"))), apply_to = function)
#endif

/* v1129N: the guest accessors are always inlined, like the lifted code's static-inline ppu__ld*L / ppu__st*L (out of
 * line they cost a call per guest store in the copy / conversion helpers). */
#if defined(__clang__) || defined(__GNUC__)
#define GF_HOT_INL static inline __attribute__((always_inline))
#else
#define GF_HOT_INL static __forceinline
#endif

static void gf_cr_field(ppu_context* ctx, unsigned shift, uint32_t value)
{
    ctx->cr = (ctx->cr & ~(0xFu << shift)) | (value << shift);
}

static uint32_t gf_cr_s64(int64_t a, int64_t b)
{
    return a < b ? 8u : a > b ? 4u : 2u;
}

static uint32_t gf_cr_u32(uint32_t a, uint32_t b)
{
    return a < b ? 8u : a > b ? 4u : 2u;
}

GF_HOT_INL uint8_t gf_ld8(uint8_t* b, uint64_t a)
{
    return b[(uint32_t)a];
}

GF_HOT_INL uint16_t gf_ld16(uint8_t* b, uint64_t a)
{
    uint16_t v;
    memcpy(&v, b + (uint32_t)a, sizeof(v));
    return __builtin_bswap16(v);
}

GF_HOT_INL uint32_t gf_ld32(uint8_t* b, uint64_t a)
{
    const uint32_t x = (uint32_t)a;
    if (__builtin_expect(x <= 0xDFFFFFFCu && x != PPU_INJECT_BASE_C + 0x2008u, 1)) {
        uint32_t v;
        memcpy(&v, b + x, sizeof(v));
        return __builtin_bswap32(v);
    }
    return vm_read32(a);
}

GF_HOT_INL uint64_t gf_ld64(uint8_t* b, uint64_t a)
{
    uint64_t v;
    memcpy(&v, b + (uint32_t)a, sizeof(v));
    return __builtin_bswap64(v);
}

GF_HOT_INL void gf_st8(uint8_t* b, uint64_t a, uint8_t v)
{
    const uint32_t x = (uint32_t)a;
    if (__builtin_expect(!(g_v243a_stflag[x >> 7] & 3u), 1)) { b[x] = v; return; }
    vm_write8(a, v);
}

GF_HOT_INL void gf_st16(uint8_t* b, uint64_t a, uint16_t v)
{
    const uint32_t x = (uint32_t)a;
    if (__builtin_expect(!(g_v243a_stflag[x >> 7] & 3u), 1)) {
        const uint16_t w = __builtin_bswap16(v);
        memcpy(b + x, &w, sizeof(w));
        return;
    }
    vm_write16(a, v);
}

GF_HOT_INL void gf_st32(uint8_t* b, uint64_t a, uint32_t v)
{
    const uint32_t x = (uint32_t)a;
    if (__builtin_expect(!(g_v243a_stflag[x >> 7] & 7u), 1)) {
        const uint32_t w = __builtin_bswap32(v);
        memcpy(b + x, &w, sizeof(w));
        return;
    }
    vm_write32(a, v);
}

GF_HOT_INL void gf_st64(uint8_t* b, uint64_t a, uint64_t v)
{
    const uint32_t x = (uint32_t)a;
    if (__builtin_expect(!(g_v243a_stflag[x >> 7] & 3u), 1)) {
        const uint64_t w = __builtin_bswap64(v);
        memcpy(b + x, &w, sizeof(w));
        return;
    }
    vm_write64(a, v);
}

static double ppu_fp_quiet(double x)
{
    uint64_t v;
    memcpy(&v, &x, sizeof(v));
    v |= 0x0008000000000000ULL;
    memcpy(&x, &v, sizeof(v));
    return x;
}

static double ppu_fp_default_qnan(void)
{
    uint64_t v = 0x7FF8000000000000ULL;
    double d;
    memcpy(&d, &v, sizeof(v));
    return d;
}

static double ppu_fp_bin(double a, double b, double r)
{
    if (a != a) return ppu_fp_quiet(a);
    if (b != b) return ppu_fp_quiet(b);
    return r != r ? ppu_fp_default_qnan() : r;
}

static double ppu_fadd(double a, double b) { return ppu_fp_bin(a, b, a + b); }
static double ppu_fmul(double a, double b) { return ppu_fp_bin(a, b, a * b); }
static double ppu_fsub(double a, double b) { return ppu_fp_bin(a, b, a - b); }

static double ppu_fmadd_core(double a, double c, double b, int neg_b, int neg_res)
{
    double r;
    if (a != a) return ppu_fp_quiet(a);
    if (b != b) return ppu_fp_quiet(b);
    if (c != c) return ppu_fp_quiet(c);
    r = fma(a, c, neg_b ? -b : b);
    if (r != r) return ppu_fp_default_qnan();
    return neg_res ? -r : r;
}

static double ppu_fp_single(double r)
{
    return r != r ? r : (double)(float)r;
}

static uint32_t ppu_vbswap32(uint32_t x)
{
    return (x >> 24) | ((x >> 8) & 0xFF00u) | ((x << 8) & 0xFF0000u) | (x << 24);
}

static void ppu_vldf4(const void* v, float o[4])
{
    const uint8_t* p = (const uint8_t*)v;
    int i;
    for (i = 0; i < 4; i++) {
        uint32_t x;
        memcpy(&x, p + i * 4, 4);
        x = ppu_vbswap32(x);
        memcpy(&o[i], &x, 4);
    }
}

static void ppu_vstf4(void* v, const float in[4])
{
    uint8_t* p = (uint8_t*)v;
    int i;
    for (i = 0; i < 4; i++) {
        uint32_t x;
        memcpy(&x, &in[i], 4);
        x = ppu_vbswap32(x);
        memcpy(p + i * 4, &x, 4);
    }
}

static uint32_t ppc_rlwinm(uint32_t rs, int sh, int mb, int me)
{
    uint32_t mask;
    uint32_t rotated;
    sh &= 31;
    rotated = sh ? ((rs << sh) | (rs >> (32 - sh))) : rs;
    if (mb <= me) mask = ((uint32_t)-1 >> mb) & ((uint32_t)-1 << (31 - me));
    else mask = ((uint32_t)-1 >> mb) | ((uint32_t)-1 << (31 - me));
    return rotated & mask;
}

static uint64_t ppc_rotl64(uint64_t v, int n)
{
    n &= 63;
    return n ? ((v << n) | (v >> (64 - n))) : v;
}

static uint64_t ppc_mask64(int mb, int me)
{
    uint64_t mask;
    if (mb <= me) {
        mask = mb == 0 ? ~0ULL : (~0ULL >> mb);
        mask &= me == 63 ? ~0ULL : (~0ULL << (63 - me));
    } else mask = (~0ULL >> mb) | (~0ULL << (63 - me));
    return mask;
}

static uint64_t ppc_rldicl(uint64_t rs, int sh, int mb)
{
    return ppc_rotl64(rs, sh) & ppc_mask64(mb, 63);
}

static uint64_t ppc_rldicr(uint64_t rs, int sh, int me)
{
    return ppc_rotl64(rs, sh) & ppc_mask64(0, me);
}

static uint64_t ppc_rldic(uint64_t rs, int sh, int mb)
{
    return ppc_rotl64(rs, sh) & ppc_mask64(mb, 63 - sh);
}

static uint64_t ppc_rldimi(uint64_t ra, uint64_t rs, int sh, int mb)
{
    const uint64_t mask = ppc_mask64(mb, 63 - sh);
    return (ppc_rotl64(rs, sh) & mask) | (ra & ~mask);
}

static int64_t ppc_sraw(uint32_t* xer, int32_t rs, int n)
{
    int sh = n & 0x3F;
    uint32_t so = (sh == 0) ? 0u : ((sh >= 32) ? (uint32_t)rs : ((uint32_t)rs & (((uint32_t)1u << sh) - 1u)));
    uint32_t ca = (rs < 0 && so != 0u) ? 1u : 0u;
    *xer = (*xer & ~(1u << 29)) | (ca << 29);
    return (int64_t)((sh >= 32) ? (rs >> 31) : (rs >> sh));
}

int gf_hot_00696704(ppu_context* ctx)
{
    const uint64_t in3 = ctx->gpr[3];
    const uint64_t in4 = ctx->gpr[4];
    const uint64_t n64 = ctx->gpr[5];
    const uint32_t a0 = (uint32_t)in3;
    const uint32_t b0 = (uint32_t)in4;
    uint32_t n;

    if (s_gf_hotfn_mask & GF_HOTFN_DISABLE_00696704) return 0;

    /* Direct byte loads are identical to the validated lean accessor only in
     * its standard memory mode.  Fail back for diagnostics, wrapped ranges,
     * and non-ABI sizes so their full lifted behavior remains observable. */
    if (g_v243a_sel != 3 || n64 > UINT32_MAX) return 0;
    n = (uint32_t)n64;
    if ((uint64_t)a0 + n > 0x100000000ull ||
        (uint64_t)b0 + n > 0x100000000ull) return 0;

    gf_cr_field(ctx, 28, gf_cr_s64((int64_t)n64, 0));
    ctx->gpr[10] = in3;
    ctx->gpr[11] = in4;
    if (n == 0) {
        ctx->gpr[3] = 0;
        return 1;
    }

    ctx->gpr[3] = vm_base[a0];
    ctx->gpr[4] = vm_base[b0];
    gf_cr_field(ctx, 0, gf_cr_s64((int32_t)ctx->gpr[3],
                                  (int32_t)ctx->gpr[4]));
    if (ctx->gpr[3] != ctx->gpr[4]) {
        gf_cr_field(ctx, 0, gf_cr_u32((uint32_t)ctx->gpr[4],
                                      (uint32_t)ctx->gpr[3]));
        ctx->gpr[3] = ctx->gpr[4] > ctx->gpr[3] ? (uint64_t)(int64_t)-1 : 1;
        return 1;
    }

    ctx->gpr[9] = (uint32_t)(n64 - 1);
    ctx->gpr[9] += 1;
    ctx->ctr = (uint32_t)ctx->gpr[9];

    for (;;) {
        ctx->gpr[0] = ctx->gpr[10] + 1;
        ctx->gpr[9] = ctx->gpr[11] + 1;
        ctx->gpr[10] = (uint32_t)ctx->gpr[0];
        ctx->gpr[11] = (uint32_t)ctx->gpr[9];
        ctx->ctr = (uint32_t)(ctx->ctr - 1);
        if (ctx->ctr == 0) {
            ctx->gpr[3] = 0;
            return 1;
        }

        ctx->gpr[3] = vm_base[(uint32_t)ctx->gpr[10]];
        ctx->gpr[4] = vm_base[(uint32_t)ctx->gpr[11]];
        gf_cr_field(ctx, 0, gf_cr_s64((int32_t)ctx->gpr[4],
                                      (int32_t)ctx->gpr[3]));
        if (ctx->gpr[3] != ctx->gpr[4]) {
            gf_cr_field(ctx, 0, gf_cr_u32((uint32_t)ctx->gpr[4],
                                          (uint32_t)ctx->gpr[3]));
            ctx->gpr[3] = ctx->gpr[4] > ctx->gpr[3] ? (uint64_t)(int64_t)-1 : 1;
            return 1;
        }
    }
}


int gf_hot_00691358(ppu_context* ctx) {
        uint8_t* const b = vm_base;
        if ((s_gf_hotfn_mask & GF_HOTFN_DISABLE_00691358) || g_v243a_sel != 3) return 0;
        { uint64_t ea = (ctx->gpr[3]) & ~0xFULL; memcpy(&ctx->vr[12], b + (uint32_t)ea, 16); }
        { uint64_t* d = (uint64_t*)&ctx->vr[9]; uint64_t* a = (uint64_t*)&ctx->vr[9]; uint64_t* b = (uint64_t*)&ctx->vr[9]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
        ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x3F00 << 16);
        { float a[4],b[4],c[4],d[4]; ppu_vldf4(&ctx->vr[12],a); ppu_vldf4(&ctx->vr[9],b); ppu_vldf4(&ctx->vr[12],c); for(int i=0;i<4;i++) d[i]=a[i]*c[i]+b[i]; ppu_vstf4(&ctx->vr[13],d); }
        { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[12], 16); memcpy(tmp+16, &ctx->vr[12], 16); memcpy(&ctx->vr[1], tmp + 4, 16); }
        { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[12], 16); memcpy(tmp+16, &ctx->vr[12], 16); memcpy(&ctx->vr[0], tmp + 8, 16); }
        ctx->gpr[9] = (int64_t)(int32_t)((uint32_t)0x3F80 << 16);
        ctx->gpr[11] = (int64_t)(int32_t)(0);
        ctx->gpr[7] = (int64_t)(int32_t)(0);
        ctx->gpr[11] = ppc_rldimi(ctx->gpr[11], ctx->gpr[0], 32, 0);
        ctx->gpr[7] = ppc_rldimi(ctx->gpr[7], ctx->gpr[9], 32, 0);
        ctx->gpr[12] = (int64_t)(int32_t)(0);
        ctx->gpr[9] = ctx->gpr[1] + (int64_t)(-16);
        ctx->gpr[8] = (int64_t)(int32_t)(0);
        gf_st64(b, ctx->gpr[9] + 0x0, ctx->gpr[11]);
        gf_st64(b, ctx->gpr[9] + 0x8, ctx->gpr[12]);
        { float a[4],b[4],c[4],d[4]; ppu_vldf4(&ctx->vr[1],a); ppu_vldf4(&ctx->vr[13],b); ppu_vldf4(&ctx->vr[1],c); for(int i=0;i<4;i++) d[i]=a[i]*c[i]+b[i]; ppu_vstf4(&ctx->vr[1],d); }
        { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; memcpy(&ctx->vr[10], b + (uint32_t)ea, 16); }
        gf_st64(b, ctx->gpr[9] + 0x0, ctx->gpr[7]);
        gf_st64(b, ctx->gpr[9] + 0x8, ctx->gpr[8]);
        { uint64_t* d = (uint64_t*)&ctx->vr[11]; uint64_t* a = (uint64_t*)&ctx->vr[10]; uint64_t* b = (uint64_t*)&ctx->vr[10]; d[0] = a[0] | b[0]; d[1] = a[1] | b[1]; }
        { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; memcpy(&ctx->vr[13], b + (uint32_t)ea, 16); }
        ctx->gpr[9] = ctx->gpr[4] + (int64_t)(0xC);
        { uint32_t* d = (uint32_t*)&ctx->vr[10]; uint32_t val = ((uint32_t*)&ctx->vr[13])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        { uint32_t* d = (uint32_t*)&ctx->vr[11]; uint32_t val = ((uint32_t*)&ctx->vr[11])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[9] + 0x0);
        { float a[4],b[4],c[4],d[4]; ppu_vldf4(&ctx->vr[0],a); ppu_vldf4(&ctx->vr[1],b); ppu_vldf4(&ctx->vr[0],c); for(int i=0;i<4;i++) d[i]=a[i]*c[i]+b[i]; ppu_vstf4(&ctx->vr[0],d); }
        { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[0])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { float b[4],d[4]; ppu_vldf4(&ctx->vr[0],b); for(int i=0;i<4;i++) d[i]=1.0f/sqrtf(b[i]); ppu_vstf4(&ctx->vr[1],d); }
        { float a[4],b[4]; ppu_vldf4(&ctx->vr[0],a); ppu_vldf4(&ctx->vr[9],b); uint32_t d[4]; int t=0; for(int i=0;i<4;i++){ int r=(a[i]==b[i]); d[i]=r?~0u:0u; t+=r; } memcpy(&ctx->vr[8], d, 16); }
        { float a[4],b[4],c[4],d[4]; ppu_vldf4(&ctx->vr[1],a); ppu_vldf4(&ctx->vr[9],b); ppu_vldf4(&ctx->vr[1],c); for(int i=0;i<4;i++) d[i]=a[i]*c[i]+b[i]; ppu_vstf4(&ctx->vr[13],d); }
        { float a[4],b[4],c[4],d[4]; ppu_vldf4(&ctx->vr[1],a); ppu_vldf4(&ctx->vr[9],b); ppu_vldf4(&ctx->vr[11],c); for(int i=0;i<4;i++) d[i]=a[i]*c[i]+b[i]; ppu_vstf4(&ctx->vr[11],d); }
        { float a[4],b[4],c[4],d[4]; ppu_vldf4(&ctx->vr[0],a); ppu_vldf4(&ctx->vr[10],b); ppu_vldf4(&ctx->vr[13],c); for(int i=0;i<4;i++) d[i]=b[i]-a[i]*c[i]; ppu_vstf4(&ctx->vr[0],d); }
        { float a[4],b[4],c[4],d[4]; ppu_vldf4(&ctx->vr[0],a); ppu_vldf4(&ctx->vr[1],b); ppu_vldf4(&ctx->vr[11],c); for(int i=0;i<4;i++) d[i]=a[i]*c[i]+b[i]; ppu_vstf4(&ctx->vr[0],d); }
        { float a[4],b[4],c[4],d[4]; ppu_vldf4(&ctx->vr[12],a); ppu_vldf4(&ctx->vr[9],b); ppu_vldf4(&ctx->vr[0],c); for(int i=0;i<4;i++) d[i]=a[i]*c[i]+b[i]; ppu_vstf4(&ctx->vr[0],d); }
        { uint64_t* d=(uint64_t*)&ctx->vr[0]; uint64_t* a=(uint64_t*)&ctx->vr[0]; uint64_t* b=(uint64_t*)&ctx->vr[12]; uint64_t* c=(uint64_t*)&ctx->vr[8]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
        { uint64_t ea = (ctx->gpr[4]) & ~0xFULL; memcpy(b + (uint32_t)ea, &ctx->vr[0], 16); }
        gf_st32(b, ctx->gpr[9] + 0x0, ctx->gpr[0]);
        return 1;
}


int gf_hot_00696784(ppu_context* ctx) {
        uint8_t* const b = vm_base;
        if ((s_gf_hotfn_mask & GF_HOTFN_DISABLE_00696784) || g_v243a_sel != 3) return 0;
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x7; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[11] = ctx->gpr[3] | ctx->gpr[3];
        if (((ctx->cr >> 0) & 4)) goto loc_006967C0;
        { int64_t a = (int64_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        ctx->gpr[9] = ctx->gpr[3] | ctx->gpr[3];
        if (((ctx->cr >> 28) & 2)) return 1;
        ctx->ctr = (uint32_t)ctx->gpr[5];
loc_006967A8:
        ctx->gpr[0] = gf_ld8(b, ctx->gpr[4] + 0x0);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(1);
        gf_st8(b, ctx->gpr[9] + 0x0, ctx->gpr[0]);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_006967A8;
        return 1;
loc_006967C0:
        ctx->gpr[6] = ppc_rldicl(ctx->gpr[4], 0, 52);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 52);
        { uint64_t a = (uint64_t)ctx->gpr[6]; uint64_t b = (uint64_t)ctx->gpr[9]; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
        if ((!((ctx->cr >> 24) & 8))) goto loc_00696958;
        ctx->gpr[4] = ctx->gpr[4] + ctx->gpr[5];
        ctx->gpr[11] = ctx->gpr[3] + ctx->gpr[5];
        ctx->gpr[7] = ppc_rldicl(ctx->gpr[4], 0, 63);
        { int64_t a = (int64_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        if ((!((ctx->cr >> 4) & 2))) goto loc_00696AE8;
loc_006967E4:
        ctx->gpr[12] = (int64_t)(int32_t)(2);
        ctx->gpr[10] = ctx->gpr[4] & ctx->gpr[12];
        { int64_t a = (int64_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if (((ctx->cr >> 0) & 2)) goto loc_00696810;
        ctx->gpr[11] = ctx->gpr[11] + (int64_t)(-2);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(-2);
        ctx->gpr[9] = gf_ld8(b, ctx->gpr[4] + 0x1);
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-2);
        gf_st8(b, ctx->gpr[11] + 0x1, ctx->gpr[9]);
        ctx->gpr[0] = gf_ld8(b, ctx->gpr[4] + 0x0);
        gf_st8(b, ctx->gpr[11] + 0x0, ctx->gpr[0]);
loc_00696810:
        ctx->gpr[7] = (int64_t)(int32_t)(4);
        ctx->gpr[6] = ctx->gpr[4] & ctx->gpr[7];
        { int64_t a = (int64_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if (((ctx->cr >> 28) & 2)) goto loc_0069684C;
        ctx->gpr[11] = ctx->gpr[11] + (int64_t)(-4);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(-4);
        ctx->gpr[0] = gf_ld8(b, ctx->gpr[4] + 0x3);
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-4);
        gf_st8(b, ctx->gpr[11] + 0x3, ctx->gpr[0]);
        ctx->gpr[12] = gf_ld8(b, ctx->gpr[4] + 0x2);
        gf_st8(b, ctx->gpr[11] + 0x2, ctx->gpr[12]);
        ctx->gpr[10] = gf_ld8(b, ctx->gpr[4] + 0x1);
        gf_st8(b, ctx->gpr[11] + 0x1, ctx->gpr[10]);
        ctx->gpr[8] = gf_ld8(b, ctx->gpr[4] + 0x0);
        gf_st8(b, ctx->gpr[11] + 0x0, ctx->gpr[8]);
loc_0069684C:
        { int64_t a = (int64_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
        if (((ctx->cr >> 24) & 2)) return 1;
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[11], 0, 62);
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        if (((ctx->cr >> 4) & 2)) goto loc_00696CAC;
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
        if (((ctx->cr >> 24) & 2)) goto loc_00696DB4;
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if (((ctx->cr >> 28) & 2)) goto loc_00696B9C;
        ctx->gpr[12] = ppc_rldicl(ctx->gpr[11], 0, 61);
        { int64_t a = (int64_t)ctx->gpr[12]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((!((ctx->cr >> 0) & 2))) goto loc_00696E30;
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x2F; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        ctx->gpr[12] = ctx->gpr[11] | ctx->gpr[11];
        if ((!((ctx->cr >> 28) & 4))) goto loc_00696918;
loc_00696888:
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(-48);
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-48);
        ctx->gpr[12] = ctx->gpr[12] + (int64_t)(-48);
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x2F; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
        ctx->gpr[0] = gf_ld64(b, ctx->gpr[4] + 0x28);
        ctx->gpr[9] = gf_ld64(b, ctx->gpr[4] + 0x20);
        ctx->gpr[11] = gf_ld64(b, ctx->gpr[4] + 0x18);
        ctx->gpr[10] = gf_ld64(b, ctx->gpr[4] + 0x10);
        ctx->gpr[8] = gf_ld64(b, ctx->gpr[4] + 0x8);
        ctx->gpr[7] = gf_ld64(b, ctx->gpr[4] + 0x0);
        gf_st64(b, ctx->gpr[12] + 0x28, ctx->gpr[0]);
        gf_st64(b, ctx->gpr[12] + 0x20, ctx->gpr[9]);
        gf_st64(b, ctx->gpr[12] + 0x18, ctx->gpr[11]);
        gf_st64(b, ctx->gpr[12] + 0x10, ctx->gpr[10]);
        gf_st64(b, ctx->gpr[12] + 0x8, ctx->gpr[8]);
        gf_st64(b, ctx->gpr[12] + 0x0, ctx->gpr[7]);
        if ((!((ctx->cr >> 24) & 4))) goto loc_00696918;
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(-48);
        ctx->gpr[12] = ctx->gpr[12] + (int64_t)(-48);
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-48);
        ctx->gpr[6] = ctx->gpr[4] + (int64_t)(-256);
        ctx->gpr[0] = gf_ld64(b, ctx->gpr[4] + 0x28);
        ctx->gpr[9] = gf_ld64(b, ctx->gpr[4] + 0x20);
        ctx->gpr[11] = gf_ld64(b, ctx->gpr[4] + 0x18);
        ctx->gpr[10] = gf_ld64(b, ctx->gpr[4] + 0x10);
        ctx->gpr[8] = gf_ld64(b, ctx->gpr[4] + 0x8);
        ctx->gpr[7] = gf_ld64(b, ctx->gpr[4] + 0x0);
        gf_st64(b, ctx->gpr[12] + 0x28, ctx->gpr[0]);
        gf_st64(b, ctx->gpr[12] + 0x20, ctx->gpr[9]);
        gf_st64(b, ctx->gpr[12] + 0x18, ctx->gpr[11]);
        gf_st64(b, ctx->gpr[12] + 0x10, ctx->gpr[10]);
        gf_st64(b, ctx->gpr[12] + 0x8, ctx->gpr[8]);
        gf_st64(b, ctx->gpr[12] + 0x0, ctx->gpr[7]);
        /* dcbt: cache hint, no-op */;
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x2F; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if (((ctx->cr >> 28) & 4)) goto loc_00696888;
loc_00696918:
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x7; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        if ((!((ctx->cr >> 4) & 4))) goto loc_00696938;
        ctx->gpr[6] = ppc_rldicl(ctx->gpr[5], 61, 3);
        ctx->ctr = (uint32_t)ctx->gpr[6];
loc_00696928:
        ctx->gpr[7] = gf_ld64(b, ctx->gpr[4] + -0x8); ctx->gpr[4] += -0x8;
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-8);
        gf_st64(b, ctx->gpr[12] + -0x8, ctx->gpr[7]); ctx->gpr[12] += -0x8;
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696928;
loc_00696938:
        { int64_t a = (int64_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ctx->gpr[12] | ctx->gpr[12];
        if (((ctx->cr >> 0) & 2)) return 1;
        ctx->ctr = (uint32_t)ctx->gpr[5];
loc_00696948:
        ctx->gpr[5] = gf_ld8(b, ctx->gpr[4] + -0x1); ctx->gpr[4] += -0x1;
        gf_st8(b, ctx->gpr[9] + -0x1, ctx->gpr[5]); ctx->gpr[9] += -0x1;
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696948;
        return 1;
loc_00696958:
        ctx->gpr[7] = ppc_rldicl(ctx->gpr[4], 0, 63);
        { int64_t a = (int64_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
        if ((!((ctx->cr >> 24) & 2))) goto loc_00696AF8;
loc_00696964:
        ctx->gpr[10] = (int64_t)(int32_t)(2);
        ctx->gpr[8] = ctx->gpr[4] & ctx->gpr[10];
        { int64_t a = (int64_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        if (((ctx->cr >> 4) & 2)) goto loc_00696990;
        ctx->gpr[0] = gf_ld8(b, ctx->gpr[4] + 0x0);
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-2);
        gf_st8(b, ctx->gpr[11] + 0x0, ctx->gpr[0]);
        ctx->gpr[9] = gf_ld8(b, ctx->gpr[4] + 0x1);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(2);
        gf_st8(b, ctx->gpr[11] + 0x1, ctx->gpr[9]);
        ctx->gpr[11] = ctx->gpr[11] + (int64_t)(2);
loc_00696990:
        ctx->gpr[7] = (int64_t)(int32_t)(4);
        ctx->gpr[12] = ctx->gpr[4] & ctx->gpr[7];
        { int64_t a = (int64_t)ctx->gpr[12]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if (((ctx->cr >> 0) & 2)) goto loc_006969CC;
        ctx->gpr[9] = gf_ld8(b, ctx->gpr[4] + 0x0);
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-4);
        gf_st8(b, ctx->gpr[11] + 0x0, ctx->gpr[9]);
        ctx->gpr[10] = gf_ld8(b, ctx->gpr[4] + 0x1);
        gf_st8(b, ctx->gpr[11] + 0x1, ctx->gpr[10]);
        ctx->gpr[8] = gf_ld8(b, ctx->gpr[4] + 0x2);
        gf_st8(b, ctx->gpr[11] + 0x2, ctx->gpr[8]);
        ctx->gpr[6] = gf_ld8(b, ctx->gpr[4] + 0x3);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(4);
        gf_st8(b, ctx->gpr[11] + 0x3, ctx->gpr[6]);
        ctx->gpr[11] = ctx->gpr[11] + (int64_t)(4);
loc_006969CC:
        { int64_t a = (int64_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if (((ctx->cr >> 28) & 2)) return 1;
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[11], 0, 62);
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
        if (((ctx->cr >> 24) & 2)) goto loc_00696C20;
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if (((ctx->cr >> 28) & 2)) goto loc_00696D30;
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if (((ctx->cr >> 0) & 2)) goto loc_00696B10;
        ctx->gpr[6] = ppc_rldicl(ctx->gpr[11], 0, 61);
        { int64_t a = (int64_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        if ((!((ctx->cr >> 4) & 2))) goto loc_00696E80;
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x2F; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[6] = ctx->gpr[11] | ctx->gpr[11];
        if ((!((ctx->cr >> 0) & 4))) goto loc_00696A98;
loc_00696A08:
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-48);
        ctx->gpr[7] = gf_ld64(b, ctx->gpr[4] + 0x0);
        ctx->gpr[0] = gf_ld64(b, ctx->gpr[4] + 0x8);
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x2F; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        ctx->gpr[9] = gf_ld64(b, ctx->gpr[4] + 0x10);
        ctx->gpr[11] = gf_ld64(b, ctx->gpr[4] + 0x18);
        ctx->gpr[10] = gf_ld64(b, ctx->gpr[4] + 0x20);
        ctx->gpr[8] = gf_ld64(b, ctx->gpr[4] + 0x28);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x30);
        gf_st64(b, ctx->gpr[6] + 0x0, ctx->gpr[7]);
        gf_st64(b, ctx->gpr[6] + 0x8, ctx->gpr[0]);
        gf_st64(b, ctx->gpr[6] + 0x10, ctx->gpr[9]);
        gf_st64(b, ctx->gpr[6] + 0x18, ctx->gpr[11]);
        gf_st64(b, ctx->gpr[6] + 0x20, ctx->gpr[10]);
        gf_st64(b, ctx->gpr[6] + 0x28, ctx->gpr[8]);
        ctx->gpr[6] = ctx->gpr[6] + (int64_t)(0x30);
        if ((!((ctx->cr >> 28) & 4))) goto loc_00696A98;
        ctx->gpr[0] = gf_ld64(b, ctx->gpr[4] + 0x0);
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-48);
        ctx->gpr[9] = gf_ld64(b, ctx->gpr[4] + 0x8);
        ctx->gpr[11] = gf_ld64(b, ctx->gpr[4] + 0x10);
        ctx->gpr[10] = gf_ld64(b, ctx->gpr[4] + 0x18);
        ctx->gpr[8] = gf_ld64(b, ctx->gpr[4] + 0x20);
        ctx->gpr[7] = gf_ld64(b, ctx->gpr[4] + 0x28);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x30);
        gf_st64(b, ctx->gpr[6] + 0x0, ctx->gpr[0]);
        gf_st64(b, ctx->gpr[6] + 0x8, ctx->gpr[9]);
        ctx->gpr[12] = ctx->gpr[4] + (int64_t)(0x100);
        gf_st64(b, ctx->gpr[6] + 0x10, ctx->gpr[11]);
        gf_st64(b, ctx->gpr[6] + 0x18, ctx->gpr[10]);
        gf_st64(b, ctx->gpr[6] + 0x20, ctx->gpr[8]);
        gf_st64(b, ctx->gpr[6] + 0x28, ctx->gpr[7]);
        ctx->gpr[6] = ctx->gpr[6] + (int64_t)(0x30);
        /* dcbt: cache hint, no-op */;
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x2F; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if (((ctx->cr >> 0) & 4)) goto loc_00696A08;
loc_00696A98:
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x7; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
        if ((!((ctx->cr >> 24) & 4))) goto loc_00696AC0;
        ctx->gpr[12] = ppc_rldicl(ctx->gpr[5], 61, 3);
        ctx->ctr = (uint32_t)ctx->gpr[12];
loc_00696AA8:
        ctx->gpr[7] = gf_ld64(b, ctx->gpr[4] + 0x0);
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-8);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(8);
        gf_st64(b, ctx->gpr[6] + 0x0, ctx->gpr[7]);
        ctx->gpr[6] = ctx->gpr[6] + (int64_t)(8);
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696AA8;
loc_00696AC0:
        { int64_t a = (int64_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        ctx->gpr[9] = ctx->gpr[6] | ctx->gpr[6];
        if (((ctx->cr >> 4) & 2)) return 1;
        ctx->ctr = (uint32_t)ctx->gpr[5];
loc_00696AD0:
        ctx->gpr[5] = gf_ld8(b, ctx->gpr[4] + 0x0);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(1);
        gf_st8(b, ctx->gpr[9] + 0x0, ctx->gpr[5]);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696AD0;
        return 1;
loc_00696AE8:
        ctx->gpr[8] = gf_ld8(b, ctx->gpr[4] + -0x1); ctx->gpr[4] += -0x1;
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-1);
        gf_st8(b, ctx->gpr[11] + -0x1, ctx->gpr[8]); ctx->gpr[11] += -0x1;
        goto loc_006967E4;
loc_00696AF8:
        ctx->gpr[6] = gf_ld8(b, ctx->gpr[4] + 0x0);
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-1);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(1);
        ctx->gpr[11] = ctx->gpr[3] + (int64_t)(1);
        gf_st8(b, ctx->gpr[3] + 0x0, ctx->gpr[6]);
        goto loc_00696964;
loc_00696B10:
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xF; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        ctx->gpr[6] = ctx->gpr[11] | ctx->gpr[11];
        ctx->gpr[12] = ctx->gpr[5] | ctx->gpr[5];
        if ((!((ctx->cr >> 28) & 4))) goto loc_00696B74;
        ctx->gpr[7] = ppc_rldicl(ctx->gpr[5], 60, 4);
        ctx->ctr = (uint32_t)ctx->gpr[7];
loc_00696B28:
        ctx->gpr[7] = gf_ld64(b, ctx->gpr[4] + 0x0);
        ctx->gpr[12] = ctx->gpr[12] + (int64_t)(-16);
        ctx->gpr[5] = gf_ld64(b, ctx->gpr[4] + 0x8);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x10);
        ctx->gpr[10] = ppc_rldicr(ctx->gpr[7], 8, 55);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[5], 8, 56);
        gf_st8(b, ctx->gpr[6] + 0xF, ctx->gpr[5]);
        ctx->gpr[8] = ppc_rldicl(ctx->gpr[7], 8, 56);
        ctx->gpr[9] = ctx->gpr[11] | ctx->gpr[10];
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[7], 40, 24);
        gf_st8(b, ctx->gpr[6] + 0x0, ctx->gpr[8]);
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[5], 40, 24);
        gf_st32(b, ctx->gpr[6] + 0x5, ctx->gpr[9]);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[5], 56, 8);
        gf_st32(b, ctx->gpr[6] + 0x1, ctx->gpr[0]);
        gf_st32(b, ctx->gpr[6] + 0x9, ctx->gpr[10]);
        gf_st16(b, ctx->gpr[6] + 0xD, ctx->gpr[11]);
        ctx->gpr[6] = ctx->gpr[6] + (int64_t)(0x10);
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696B28;
loc_00696B74:
        { int64_t a = (int64_t)ctx->gpr[12]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
        ctx->ctr = (uint32_t)ctx->gpr[12];
        ctx->gpr[9] = ctx->gpr[6] | ctx->gpr[6];
        if (((ctx->cr >> 24) & 2)) return 1;
loc_00696B84:
        ctx->gpr[12] = gf_ld8(b, ctx->gpr[4] + 0x0);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(1);
        gf_st8(b, ctx->gpr[9] + 0x0, ctx->gpr[12]);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696B84;
        return 1;
loc_00696B9C:
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xF; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
        ctx->gpr[6] = ctx->gpr[11] | ctx->gpr[11];
        ctx->gpr[12] = ctx->gpr[5] | ctx->gpr[5];
        if ((!((ctx->cr >> 24) & 4))) goto loc_00696C00;
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 60, 4);
        ctx->ctr = (uint32_t)ctx->gpr[5];
loc_00696BB4:
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(-16);
        ctx->gpr[6] = ctx->gpr[6] + (int64_t)(-16);
        ctx->gpr[12] = ctx->gpr[12] + (int64_t)(-16);
        ctx->gpr[10] = gf_ld64(b, ctx->gpr[4] + 0x8);
        ctx->gpr[5] = gf_ld64(b, ctx->gpr[4] + 0x0);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[10], 8, 56);
        gf_st8(b, ctx->gpr[6] + 0xF, ctx->gpr[10]);
        ctx->gpr[9] = ppc_rldicr(ctx->gpr[5], 8, 55);
        ctx->gpr[7] = ppc_rldicl(ctx->gpr[5], 8, 56);
        ctx->gpr[11] = ctx->gpr[0] | ctx->gpr[9];
        ctx->gpr[8] = ppc_rldicl(ctx->gpr[10], 56, 8);
        gf_st8(b, ctx->gpr[6] + 0x0, ctx->gpr[7]);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 40, 24);
        gf_st32(b, ctx->gpr[6] + 0x5, ctx->gpr[11]);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[5], 40, 24);
        gf_st16(b, ctx->gpr[6] + 0xD, ctx->gpr[8]);
        gf_st32(b, ctx->gpr[6] + 0x9, ctx->gpr[9]);
        gf_st32(b, ctx->gpr[6] + 0x1, ctx->gpr[0]);
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696BB4;
loc_00696C00:
        { int64_t a = (int64_t)ctx->gpr[12]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        ctx->ctr = (uint32_t)ctx->gpr[12];
        ctx->gpr[9] = ctx->gpr[6] | ctx->gpr[6];
        if (((ctx->cr >> 4) & 2)) return 1;
loc_00696C10:
        ctx->gpr[6] = gf_ld8(b, ctx->gpr[4] + -0x1); ctx->gpr[4] += -0x1;
        gf_st8(b, ctx->gpr[9] + -0x1, ctx->gpr[6]); ctx->gpr[9] += -0x1;
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696C10;
        return 1;
loc_00696C20:
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xF; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        ctx->gpr[6] = ctx->gpr[11] | ctx->gpr[11];
        ctx->gpr[12] = ctx->gpr[5] | ctx->gpr[5];
        if ((!((ctx->cr >> 4) & 4))) goto loc_00696C84;
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 60, 4);
        ctx->ctr = (uint32_t)ctx->gpr[5];
loc_00696C38:
        ctx->gpr[5] = gf_ld64(b, ctx->gpr[4] + 0x0);
        ctx->gpr[12] = ctx->gpr[12] + (int64_t)(-16);
        ctx->gpr[10] = gf_ld64(b, ctx->gpr[4] + 0x8);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x10);
        ctx->gpr[11] = ppc_rldicr(ctx->gpr[5], 24, 39);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 24, 40);
        gf_st8(b, ctx->gpr[6] + 0xF, ctx->gpr[10]);
        ctx->gpr[7] = ppc_rldicl(ctx->gpr[5], 8, 56);
        ctx->gpr[9] = ctx->gpr[9] | ctx->gpr[11];
        ctx->gpr[8] = ppc_rldicl(ctx->gpr[5], 24, 40);
        gf_st8(b, ctx->gpr[6] + 0x0, ctx->gpr[7]);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[5], 56, 8);
        gf_st32(b, ctx->gpr[6] + 0x7, ctx->gpr[9]);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 56, 8);
        gf_st16(b, ctx->gpr[6] + 0x1, ctx->gpr[8]);
        gf_st32(b, ctx->gpr[6] + 0x3, ctx->gpr[0]);
        gf_st32(b, ctx->gpr[6] + 0xB, ctx->gpr[11]);
        ctx->gpr[6] = ctx->gpr[6] + (int64_t)(0x10);
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696C38;
loc_00696C84:
        { int64_t a = (int64_t)ctx->gpr[12]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->ctr = (uint32_t)ctx->gpr[12];
        ctx->gpr[9] = ctx->gpr[6] | ctx->gpr[6];
        if (((ctx->cr >> 0) & 2)) return 1;
loc_00696C94:
        ctx->gpr[12] = gf_ld8(b, ctx->gpr[4] + 0x0);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(1);
        gf_st8(b, ctx->gpr[9] + 0x0, ctx->gpr[12]);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696C94;
        return 1;
loc_00696CAC:
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xF; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[12] = ctx->gpr[11] | ctx->gpr[11];
        ctx->gpr[6] = ctx->gpr[5] | ctx->gpr[5];
        if ((!((ctx->cr >> 0) & 4))) goto loc_00696D10;
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 60, 4);
        ctx->ctr = (uint32_t)ctx->gpr[5];
loc_00696CC4:
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(-16);
        ctx->gpr[12] = ctx->gpr[12] + (int64_t)(-16);
        ctx->gpr[6] = ctx->gpr[6] + (int64_t)(-16);
        ctx->gpr[5] = gf_ld64(b, ctx->gpr[4] + 0x0);
        ctx->gpr[8] = gf_ld64(b, ctx->gpr[4] + 0x8);
        ctx->gpr[11] = ppc_rldicr(ctx->gpr[5], 24, 39);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[8], 24, 40);
        gf_st8(b, ctx->gpr[12] + 0xF, ctx->gpr[8]);
        ctx->gpr[7] = ppc_rldicl(ctx->gpr[5], 8, 56);
        ctx->gpr[9] = ctx->gpr[0] | ctx->gpr[11];
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[8], 56, 8);
        gf_st8(b, ctx->gpr[12] + 0x0, ctx->gpr[7]);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[5], 56, 8);
        gf_st32(b, ctx->gpr[12] + 0x7, ctx->gpr[9]);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[5], 24, 40);
        gf_st32(b, ctx->gpr[12] + 0xB, ctx->gpr[10]);
        gf_st32(b, ctx->gpr[12] + 0x3, ctx->gpr[11]);
        gf_st16(b, ctx->gpr[12] + 0x1, ctx->gpr[0]);
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696CC4;
loc_00696D10:
        { int64_t a = (int64_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        ctx->ctr = (uint32_t)ctx->gpr[6];
        ctx->gpr[9] = ctx->gpr[12] | ctx->gpr[12];
        if (((ctx->cr >> 28) & 2)) return 1;
loc_00696D20:
        ctx->gpr[6] = gf_ld8(b, ctx->gpr[4] + -0x1); ctx->gpr[4] += -0x1;
        gf_st8(b, ctx->gpr[9] + -0x1, ctx->gpr[6]); ctx->gpr[9] += -0x1;
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696D20;
        return 1;
loc_00696D30:
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xF; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
        ctx->gpr[7] = ctx->gpr[11] | ctx->gpr[11];
        ctx->gpr[6] = ctx->gpr[5] | ctx->gpr[5];
        if ((!((ctx->cr >> 24) & 4))) goto loc_00696D8C;
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[5], 60, 4);
        ctx->ctr = (uint32_t)ctx->gpr[10];
loc_00696D48:
        ctx->gpr[12] = gf_ld64(b, ctx->gpr[4] + 0x0);
        ctx->gpr[6] = ctx->gpr[6] + (int64_t)(-16);
        ctx->gpr[5] = gf_ld64(b, ctx->gpr[4] + 0x8);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x10);
        ctx->gpr[10] = ppc_rldicr(ctx->gpr[12], 16, 47);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[5], 16, 48);
        gf_st16(b, ctx->gpr[7] + 0xE, ctx->gpr[5]);
        ctx->gpr[8] = ppc_rldicl(ctx->gpr[12], 16, 48);
        ctx->gpr[9] = ctx->gpr[11] | ctx->gpr[10];
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[12], 48, 16);
        gf_st16(b, ctx->gpr[7] + 0x0, ctx->gpr[8]);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[5], 48, 16);
        gf_st32(b, ctx->gpr[7] + 0x6, ctx->gpr[9]);
        gf_st32(b, ctx->gpr[7] + 0x2, ctx->gpr[0]);
        gf_st32(b, ctx->gpr[7] + 0xA, ctx->gpr[11]);
        ctx->gpr[7] = ctx->gpr[7] + (int64_t)(0x10);
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696D48;
loc_00696D8C:
        { int64_t a = (int64_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        ctx->gpr[9] = ctx->gpr[7] | ctx->gpr[7];
        if (((ctx->cr >> 4) & 2)) return 1;
        ctx->ctr = (uint32_t)ctx->gpr[6];
loc_00696D9C:
        ctx->gpr[6] = gf_ld8(b, ctx->gpr[4] + 0x0);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(1);
        gf_st8(b, ctx->gpr[9] + 0x0, ctx->gpr[6]);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696D9C;
        return 1;
loc_00696DB4:
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xF; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        ctx->gpr[6] = ctx->gpr[11] | ctx->gpr[11];
        ctx->gpr[12] = ctx->gpr[5] | ctx->gpr[5];
        if ((!((ctx->cr >> 4) & 4))) goto loc_00696E10;
        ctx->gpr[8] = ppc_rldicl(ctx->gpr[5], 60, 4);
        ctx->ctr = (uint32_t)ctx->gpr[8];
loc_00696DCC:
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(-16);
        ctx->gpr[6] = ctx->gpr[6] + (int64_t)(-16);
        ctx->gpr[12] = ctx->gpr[12] + (int64_t)(-16);
        ctx->gpr[5] = gf_ld64(b, ctx->gpr[4] + 0x8);
        ctx->gpr[8] = gf_ld64(b, ctx->gpr[4] + 0x0);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[5], 16, 48);
        gf_st16(b, ctx->gpr[6] + 0xE, ctx->gpr[5]);
        ctx->gpr[11] = ppc_rldicr(ctx->gpr[8], 16, 47);
        ctx->gpr[7] = ppc_rldicl(ctx->gpr[8], 16, 48);
        ctx->gpr[9] = ctx->gpr[0] | ctx->gpr[11];
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[5], 48, 16);
        gf_st16(b, ctx->gpr[6] + 0x0, ctx->gpr[7]);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[8], 48, 16);
        gf_st32(b, ctx->gpr[6] + 0x6, ctx->gpr[9]);
        gf_st32(b, ctx->gpr[6] + 0xA, ctx->gpr[10]);
        gf_st32(b, ctx->gpr[6] + 0x2, ctx->gpr[0]);
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696DCC;
loc_00696E10:
        { int64_t a = (int64_t)ctx->gpr[12]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ctx->gpr[6] | ctx->gpr[6];
        if (((ctx->cr >> 0) & 2)) return 1;
        ctx->ctr = (uint32_t)ctx->gpr[12];
loc_00696E20:
        ctx->gpr[12] = gf_ld8(b, ctx->gpr[4] + -0x1); ctx->gpr[4] += -0x1;
        gf_st8(b, ctx->gpr[9] + -0x1, ctx->gpr[12]); ctx->gpr[9] += -0x1;
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696E20;
        return 1;
loc_00696E30:
        ctx->gpr[12] = ctx->gpr[11] | ctx->gpr[11];
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x17; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
loc_00696E38:
        if ((!((ctx->cr >> 0) & 4))) goto loc_00696ED0;
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(-24);
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-24);
        ctx->gpr[12] = ctx->gpr[12] + (int64_t)(-24);
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x17; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[10] = gf_ld64(b, ctx->gpr[4] + 0x10);
        ctx->gpr[8] = gf_ld64(b, ctx->gpr[4] + 0x8);
        ctx->gpr[6] = gf_ld64(b, ctx->gpr[4] + 0x0);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[10], 32, 32);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 32, 32);
        gf_st32(b, ctx->gpr[12] + 0x14, ctx->gpr[10]);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[6], 32, 32);
        gf_st32(b, ctx->gpr[12] + 0x10, ctx->gpr[0]);
        gf_st32(b, ctx->gpr[12] + 0x8, ctx->gpr[9]);
        gf_st32(b, ctx->gpr[12] + 0x0, ctx->gpr[11]);
        gf_st32(b, ctx->gpr[12] + 0xC, ctx->gpr[8]);
        gf_st32(b, ctx->gpr[12] + 0x4, ctx->gpr[6]);
        goto loc_00696E38;
loc_00696E80:
        ctx->gpr[6] = ctx->gpr[11] | ctx->gpr[11];
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x17; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
loc_00696E88:
        if ((!((ctx->cr >> 0) & 4))) goto loc_00696EF0;
        ctx->gpr[10] = gf_ld64(b, ctx->gpr[4] + 0x0);
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-24);
        ctx->gpr[8] = gf_ld64(b, ctx->gpr[4] + 0x8);
        ctx->gpr[12] = gf_ld64(b, ctx->gpr[4] + 0x10);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[10], 32, 32);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 32, 32);
        gf_st32(b, ctx->gpr[6] + 0x4, ctx->gpr[10]);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[12], 32, 32);
        gf_st32(b, ctx->gpr[6] + 0x0, ctx->gpr[0]);
        { uint64_t a = (uint64_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x17; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        gf_st32(b, ctx->gpr[6] + 0x8, ctx->gpr[9]);
        gf_st32(b, ctx->gpr[6] + 0xC, ctx->gpr[8]);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x18);
        gf_st32(b, ctx->gpr[6] + 0x10, ctx->gpr[11]);
        gf_st32(b, ctx->gpr[6] + 0x14, ctx->gpr[12]);
        ctx->gpr[6] = ctx->gpr[6] + (int64_t)(0x18);
        goto loc_00696E88;
loc_00696ED0:
        { int64_t a = (int64_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        ctx->gpr[9] = ctx->gpr[12] | ctx->gpr[12];
        if (((ctx->cr >> 28) & 2)) return 1;
        ctx->ctr = (uint32_t)ctx->gpr[5];
loc_00696EE0:
        ctx->gpr[12] = gf_ld8(b, ctx->gpr[4] + -0x1); ctx->gpr[4] += -0x1;
        gf_st8(b, ctx->gpr[9] + -0x1, ctx->gpr[12]); ctx->gpr[9] += -0x1;
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696EE0;
        return 1;
loc_00696EF0:
        { int64_t a = (int64_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ctx->gpr[6] | ctx->gpr[6];
        if (((ctx->cr >> 0) & 2)) return 1;
        ctx->ctr = (uint32_t)ctx->gpr[5];
loc_00696F00:
        ctx->gpr[6] = gf_ld8(b, ctx->gpr[4] + 0x0);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(1);
        gf_st8(b, ctx->gpr[9] + 0x0, ctx->gpr[6]);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        if (((ctx->ctr = (uint32_t)(ctx->ctr - 1)) != 0)) goto loc_00696F00;
        return 1;
        /* TODO: .word 0x00000000 */;
        /* TODO: .word 0x00000000 */;
        /* TODO: .word 0x00000000 */;
        { g_trampoline_fn = (void(*)(void*))func_00696F24; return 1; }
}


int gf_hot_006A9400(ppu_context* ctx) {
        uint8_t* const b = vm_base;
        if ((s_gf_hotfn_mask & GF_HOTFN_DISABLE_006A9400) || g_v243a_sel != 3) return 0;
        uint64_t _cs_26 = ctx->gpr[26];
        uint64_t _cs_27 = ctx->gpr[27];
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        gf_st64(b, ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
        ctx->gpr[0] = ctx->lr;
        gf_st64(b, ctx->gpr[1] + 0xA8, ctx->gpr[29]);
        ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
        gf_st64(b, ctx->gpr[1] + 0xD0, ctx->gpr[0]);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[4] + 0x4);
        gf_st64(b, ctx->gpr[1] + 0xB0, ctx->gpr[30]);
        ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xCB8; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        gf_st64(b, ctx->gpr[1] + 0xB8, ctx->gpr[31]);
        gf_st64(b, ctx->gpr[1] + 0x90, ctx->gpr[26]);
        ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
        gf_st64(b, ctx->gpr[1] + 0x98, ctx->gpr[27]);
        gf_st64(b, ctx->gpr[1] + 0xA0, ctx->gpr[28]);
        if (((ctx->cr >> 0) & 2)) goto loc_006A944C;
        ctx->gpr[9] = gf_ld32(b, ctx->gpr[4] + 0x0);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1045);
        { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x2E; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((!((ctx->cr >> 0) & 4))) goto loc_006A9474;
loc_006A944C:
        ctx->gpr[0] = gf_ld64(b, ctx->gpr[1] + 0xD0);
        ctx->gpr[26] = _cs_26;
        ctx->gpr[27] = _cs_27;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
        return 1;
loc_006A9474:
        ctx->gpr[11] = gf_ld32(b, ctx->gpr[2] + -0x4254);
        ctx->gpr[9] = ppc_rldic(ctx->gpr[9], 2, 30);
        ctx->gpr[0] = gf_ld32(b, (ctx->gpr[9] + ctx->gpr[11]));
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->ctr = (uint32_t)ctx->gpr[0];
        switch ((uint32_t)ctx->ctr) { case 0x006A944Cu: goto loc_006A944C;case 0x006A954Cu: goto loc_006A954C;case 0x006A9618u: goto loc_006A9618;case 0x006A9740u: goto loc_006A9740; default: ps3_indirect_call(ctx); return 1; } return 1;
        /* TODO: .word 0x000002B0 */;
        /* TODO: .word 0x000002B0 */;
        /* TODO: .word 0x000002B0 */;
        /* TODO: .word 0x000002B0 */;
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        /* TODO: .word 0x000000BC */;
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        /* TODO: .word 0x00000188 */;
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        ctx->fpr[31] = ppu_fmadd_core(ctx->fpr[31], ctx->fpr[30], ctx->fpr[31], 1, 1);
        /* TODO: .word 0x000002B0 */;
loc_006A954C:
        ctx->gpr[0] = ctx->gpr[4] + (int64_t)(0x34);
        ctx->gpr[27] = (int64_t)(int32_t)((uint32_t)0x14 << 16);
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[27] = ctx->gpr[27] | 0x1EFC;
        ctx->gpr[28] = (int64_t)(int32_t)(3);
        ctx->gpr[26] = (int64_t)(int32_t)(0);
loc_006A9564:
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[30] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xCB8; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if (((ctx->cr >> 0) & 2)) goto loc_006A95D0;
        { uint32_t tmp = gf_ld32(b, ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->gpr[8] = gf_ld32(b, ctx->gpr[29] + 0x8);
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[1] + 0x70, tmp); }
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[29] + 0x4);
        ctx->gpr[9] = ctx->gpr[8] + (int64_t)(0x18);
        { uint32_t tmp = gf_ld32(b, ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[1] + 0x74, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[1] + 0x78, tmp); }
        if ((!((ctx->cr >> 0) & 8))) goto loc_006A9868;
loc_006A959C:
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[8], 0, 32);
        ctx->gpr[0] = ctx->gpr[8] + (int64_t)(0x18);
        gf_st32(b, ctx->gpr[11] + 0x0, ctx->gpr[27]);
        ctx->gpr[9] = gf_ld32(b, ctx->gpr[30] + 0x8);
        gf_st32(b, ctx->gpr[29] + 0x8, ctx->gpr[0]);
        gf_st32(b, ctx->gpr[11] + 0x4, ctx->gpr[9]);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[1] + 0x70);
        gf_st32(b, ctx->gpr[11] + 0x8, ctx->gpr[0]);
        ctx->gpr[9] = gf_ld32(b, ctx->gpr[1] + 0x74);
        gf_st32(b, ctx->gpr[11] + 0xC, ctx->gpr[9]);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[1] + 0x78);
        gf_st32(b, ctx->gpr[11] + 0x14, ctx->gpr[26]);
        gf_st32(b, ctx->gpr[11] + 0x10, ctx->gpr[0]);
loc_006A95D0:
        ctx->gpr[0] = ctx->gpr[28] + (int64_t)(-1);
        ctx->gpr[11] = ctx->gpr[31] + (int64_t)(0xC);
        ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[9] = ctx->gpr[30] + (int64_t)(0x30);
        { int64_t a = (int64_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[9], 0, 32);
        if ((!((ctx->cr >> 0) & 2))) goto loc_006A9564;
        ctx->gpr[0] = gf_ld64(b, ctx->gpr[1] + 0xD0);
        ctx->gpr[26] = _cs_26;
        ctx->gpr[27] = _cs_27;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
        return 1;
loc_006A9618:
        ctx->gpr[0] = ctx->gpr[4] + (int64_t)(0x30);
        ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[9] = gf_ld32(b, ctx->gpr[28] + 0x4);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0xCB8; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if (((ctx->cr >> 0) & 2)) goto loc_006A97B8;
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[4] + 0x64);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xCB8; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if (((ctx->cr >> 0) & 2)) goto loc_006A97B8;
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[4] + 0x94);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xCB8; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if (((ctx->cr >> 0) & 2)) goto loc_006A97B8;
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[4] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xCB8; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if (((ctx->cr >> 0) & 2)) goto loc_006A97B8;
        ctx->gpr[8] = gf_ld32(b, ctx->gpr[3] + 0x8);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[3] + 0x4);
        ctx->gpr[9] = ctx->gpr[8] + (int64_t)(0x48);
        { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if (((ctx->cr >> 0) & 8)) goto loc_006A9690;
        ctx->gpr[9] = gf_ld32(b, ctx->gpr[3] + 0xC);
        ctx->gpr[4] = (int64_t)(int32_t)(0x12);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[9] + 0x0);
        gf_st64(b, ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = (uint32_t)ctx->gpr[0];
        ctx->gpr[2] = gf_ld32(b, ctx->gpr[9] + 0x4);
        ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = gf_ld64(b, ctx->gpr[1] + 0x28);
        { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((!((ctx->cr >> 0) & 2))) goto loc_006A944C;
        ctx->gpr[8] = gf_ld32(b, ctx->gpr[29] + 0x8);
loc_006A9690:
        ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x44 << 16);
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[8], 0, 32);
        ctx->gpr[0] = ctx->gpr[0] | 0x1EFC;
        ctx->gpr[9] = ctx->gpr[8] + (int64_t)(8);
        gf_st32(b, ctx->gpr[10] + 0x0, ctx->gpr[0]);
        ctx->gpr[11] = gf_ld32(b, ctx->gpr[28] + 0xC);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        gf_st32(b, ctx->gpr[10] + 0x4, ctx->gpr[11]);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[31] + 0x0);
        ctx->gpr[11] = gf_ld32(b, ctx->gpr[31] + 0x4);
        ctx->gpr[10] = gf_ld32(b, ctx->gpr[31] + 0x8);
        ctx->gpr[8] = gf_ld32(b, ctx->gpr[31] + 0xC);
        gf_st32(b, ctx->gpr[9] + 0x0, ctx->gpr[0]);
        gf_st32(b, ctx->gpr[9] + 0x4, ctx->gpr[11]);
        gf_st32(b, ctx->gpr[9] + 0x8, ctx->gpr[10]);
        gf_st32(b, ctx->gpr[9] + 0xC, ctx->gpr[8]);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[31] + 0x10);
        ctx->gpr[11] = gf_ld32(b, ctx->gpr[31] + 0x14);
        ctx->gpr[10] = gf_ld32(b, ctx->gpr[31] + 0x18);
        ctx->gpr[8] = gf_ld32(b, ctx->gpr[31] + 0x1C);
        gf_st32(b, ctx->gpr[9] + 0x10, ctx->gpr[0]);
        gf_st32(b, ctx->gpr[9] + 0x14, ctx->gpr[11]);
        gf_st32(b, ctx->gpr[9] + 0x18, ctx->gpr[10]);
        gf_st32(b, ctx->gpr[9] + 0x1C, ctx->gpr[8]);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[31] + 0x20);
        ctx->gpr[11] = gf_ld32(b, ctx->gpr[31] + 0x24);
        ctx->gpr[10] = gf_ld32(b, ctx->gpr[31] + 0x28);
        ctx->gpr[8] = gf_ld32(b, ctx->gpr[31] + 0x2C);
        gf_st32(b, ctx->gpr[9] + 0x20, ctx->gpr[0]);
        gf_st32(b, ctx->gpr[9] + 0x2C, ctx->gpr[8]);
        gf_st32(b, ctx->gpr[9] + 0x24, ctx->gpr[11]);
        gf_st32(b, ctx->gpr[9] + 0x28, ctx->gpr[10]);
        ctx->gpr[7] = gf_ld32(b, ctx->gpr[31] + 0x3C);
        ctx->gpr[11] = gf_ld32(b, ctx->gpr[31] + 0x34);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[31] + 0x30);
        ctx->gpr[10] = gf_ld32(b, ctx->gpr[31] + 0x38);
        gf_st32(b, ctx->gpr[9] + 0x34, ctx->gpr[11]);
        gf_st32(b, ctx->gpr[9] + 0x30, ctx->gpr[0]);
        gf_st32(b, ctx->gpr[9] + 0x38, ctx->gpr[10]);
        gf_st32(b, ctx->gpr[9] + 0x3C, ctx->gpr[7]);
        ctx->gpr[11] = gf_ld32(b, ctx->gpr[29] + 0x8);
        ctx->gpr[8] = ctx->gpr[11] + (int64_t)(0x48);
        gf_st32(b, ctx->gpr[29] + 0x8, ctx->gpr[8]);
        goto loc_006A944C;
loc_006A9740:
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->gpr[8] = gf_ld32(b, ctx->gpr[3] + 0x8);
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[1] + 0x7C, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        ctx->gpr[9] = ctx->gpr[8] + (int64_t)(0x18);
        { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[1] + 0x80, tmp); }
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[3] + 0x4);
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[1] + 0x84, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[1] + 0x88, tmp); }
        if ((!((ctx->cr >> 0) & 8))) goto loc_006A98D0;
loc_006A9774:
        ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x14 << 16);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[8], 0, 32);
        ctx->gpr[0] = ctx->gpr[0] | 0x1EFC;
        ctx->gpr[8] = ctx->gpr[8] + (int64_t)(0x18);
        gf_st32(b, ctx->gpr[11] + 0x0, ctx->gpr[0]);
        ctx->gpr[9] = gf_ld32(b, ctx->gpr[30] + 0xC);
        gf_st32(b, ctx->gpr[29] + 0x8, ctx->gpr[8]);
        gf_st32(b, ctx->gpr[11] + 0x4, ctx->gpr[9]);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[1] + 0x7C);
        gf_st32(b, ctx->gpr[11] + 0x8, ctx->gpr[0]);
        ctx->gpr[9] = gf_ld32(b, ctx->gpr[1] + 0x80);
        gf_st32(b, ctx->gpr[11] + 0xC, ctx->gpr[9]);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[1] + 0x84);
        gf_st32(b, ctx->gpr[11] + 0x10, ctx->gpr[0]);
        ctx->gpr[9] = gf_ld32(b, ctx->gpr[1] + 0x88);
        gf_st32(b, ctx->gpr[11] + 0x14, ctx->gpr[9]);
        goto loc_006A944C;
loc_006A97B8:
        ctx->gpr[0] = ctx->gpr[28] + (int64_t)(4);
        ctx->gpr[27] = (int64_t)(int32_t)((uint32_t)0x14 << 16);
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[27] = ctx->gpr[27] | 0x1EFC;
        ctx->gpr[28] = (int64_t)(int32_t)(0);
loc_006A97CC:
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[30] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xCB8; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if (((ctx->cr >> 0) & 2)) goto loc_006A9844;
        { uint32_t tmp = gf_ld32(b, ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->gpr[8] = gf_ld32(b, ctx->gpr[29] + 0x8);
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[1] + 0x7C, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        ctx->gpr[9] = ctx->gpr[8] + (int64_t)(0x18);
        { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[1] + 0x80, tmp); }
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[29] + 0x4);
        { uint32_t tmp = gf_ld32(b, ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[1] + 0x84, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[1] + 0x88, tmp); }
        if ((!((ctx->cr >> 0) & 8))) goto loc_006A989C;
loc_006A980C:
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[8], 0, 32);
        ctx->gpr[8] = ctx->gpr[8] + (int64_t)(0x18);
        gf_st32(b, ctx->gpr[11] + 0x0, ctx->gpr[27]);
        ctx->gpr[9] = gf_ld32(b, ctx->gpr[30] + 0x8);
        gf_st32(b, ctx->gpr[29] + 0x8, ctx->gpr[8]);
        gf_st32(b, ctx->gpr[11] + 0x4, ctx->gpr[9]);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[1] + 0x7C);
        gf_st32(b, ctx->gpr[11] + 0x8, ctx->gpr[0]);
        ctx->gpr[9] = gf_ld32(b, ctx->gpr[1] + 0x80);
        gf_st32(b, ctx->gpr[11] + 0xC, ctx->gpr[9]);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[1] + 0x84);
        gf_st32(b, ctx->gpr[11] + 0x10, ctx->gpr[0]);
        ctx->gpr[9] = gf_ld32(b, ctx->gpr[1] + 0x88);
        gf_st32(b, ctx->gpr[11] + 0x14, ctx->gpr[9]);
loc_006A9844:
        ctx->gpr[0] = ctx->gpr[28] + (int64_t)(1);
        ctx->gpr[11] = ctx->gpr[31] + (int64_t)(0x10);
        ctx->gpr[28] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[30] + (int64_t)(0x30);
        { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)4; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[9], 0, 32);
        if ((!((ctx->cr >> 0) & 2))) goto loc_006A97CC;
        goto loc_006A944C;
loc_006A9868:
        ctx->gpr[9] = gf_ld32(b, ctx->gpr[29] + 0xC);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[4] = (int64_t)(int32_t)(6);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[9] + 0x0);
        gf_st64(b, ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = (uint32_t)ctx->gpr[0];
        ctx->gpr[2] = gf_ld32(b, ctx->gpr[9] + 0x4);
        ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = gf_ld64(b, ctx->gpr[1] + 0x28);
        { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((!((ctx->cr >> 0) & 2))) goto loc_006A944C;
        ctx->gpr[8] = gf_ld32(b, ctx->gpr[29] + 0x8);
        goto loc_006A959C;
loc_006A989C:
        ctx->gpr[9] = gf_ld32(b, ctx->gpr[29] + 0xC);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[4] = (int64_t)(int32_t)(6);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[9] + 0x0);
        gf_st64(b, ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = (uint32_t)ctx->gpr[0];
        ctx->gpr[2] = gf_ld32(b, ctx->gpr[9] + 0x4);
        ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = gf_ld64(b, ctx->gpr[1] + 0x28);
        { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((!((ctx->cr >> 0) & 2))) goto loc_006A944C;
        ctx->gpr[8] = gf_ld32(b, ctx->gpr[29] + 0x8);
        goto loc_006A980C;
loc_006A98D0:
        ctx->gpr[9] = gf_ld32(b, ctx->gpr[3] + 0xC);
        ctx->gpr[4] = (int64_t)(int32_t)(6);
        ctx->gpr[0] = gf_ld32(b, ctx->gpr[9] + 0x0);
        gf_st64(b, ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = (uint32_t)ctx->gpr[0];
        ctx->gpr[2] = gf_ld32(b, ctx->gpr[9] + 0x4);
        ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = gf_ld64(b, ctx->gpr[1] + 0x28);
        { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((!((ctx->cr >> 0) & 2))) goto loc_006A944C;
        ctx->gpr[8] = gf_ld32(b, ctx->gpr[29] + 0x8);
        goto loc_006A9774;
}


int gf_hot2_0070AF14(ppu_context* ctx);
int gf_hot_0070AF14(ppu_context* ctx) {
        if (s_gf_hotfn2 & GF_HOTFN2_0070AF14) return gf_hot2_0070AF14(ctx);   /* v1129N */
        uint8_t* const b = vm_base;
        if ((s_gf_hotfn_mask & GF_HOTFN_DISABLE_0070AF14) || g_v243a_sel != 3) return 0;
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
        ctx->fpr[5] = ppu_fp_single(ppu_fmul(ctx->fpr[9], ctx->fpr[10]));
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        ctx->fpr[9] = ppu_fp_single(ppu_fmul(ctx->fpr[9], ctx->fpr[8]));
        ctx->fpr[10] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[10]));
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[8]));
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); gf_st64(b, ctx->gpr[1] + -0x10, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[6] = ftmp; }
        ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[11], ctx->fpr[7], ctx->fpr[5], 0, 0));
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); gf_st64(b, ctx->gpr[1] + -0x28, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); gf_st64(b, ctx->gpr[1] + -0x18, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); gf_st64(b, ctx->gpr[1] + -0x20, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); gf_st64(b, ctx->gpr[1] + -0x8, tmp); }
        ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[3], ctx->fpr[2], ctx->fpr[11], 0, 0));
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
        { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[3] + 0x0, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
        ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[11], ctx->fpr[12], ctx->fpr[9], 0, 0));
        ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[30], ctx->fpr[10], 0, 0));
        ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[3], ctx->fpr[5], ctx->fpr[12], 0, 0));
        ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[6], ctx->fpr[2], ctx->fpr[0], 0, 0));
        { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[3] + 0x4, tmp); }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[3] + 0x10, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[27]));
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
        ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[10], ctx->fpr[1], ctx->fpr[13], 0, 0));
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
        ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[7], ctx->fpr[4]));
        ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[11], ctx->fpr[29], ctx->fpr[0], 0, 0));
        ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[6], ctx->fpr[5], ctx->fpr[13], 0, 0));
        ctx->fpr[8] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[8], ctx->fpr[30], ctx->fpr[12], 0, 0));
        ctx->fpr[3] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[3], ctx->fpr[28], ctx->fpr[11], 0, 0));
        { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[3] + 0x14, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
        ctx->fpr[2] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[31], ctx->fpr[2], ctx->fpr[8], 0, 0));
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->fpr[7] = ppu_fp_single(ppu_fmul(ctx->fpr[7], ctx->fpr[29]));
        ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[27]));
        { float ftmp = (float)ctx->fpr[3]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[3] + 0x8, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
        ctx->fpr[4] = ppu_fp_single(ppu_fmul(ctx->fpr[9], ctx->fpr[4]));
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        { float ftmp = (float)ctx->fpr[2]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[3] + 0x20, tmp); }
        ctx->fpr[9] = ppu_fp_single(ppu_fmul(ctx->fpr[9], ctx->fpr[29]));
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        ctx->fpr[7] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[8], ctx->fpr[1], ctx->fpr[7], 0, 0));
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
        ctx->fpr[10] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[10], ctx->fpr[3], ctx->fpr[0], 0, 0));
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
        { uint64_t tmp = gf_ld64(b, ctx->gpr[1] + -0x28); memcpy(&ctx->fpr[27], &tmp, 8); }
        { uint64_t tmp = gf_ld64(b, ctx->gpr[1] + -0x18); memcpy(&ctx->fpr[29], &tmp, 8); }
        ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[30], ctx->fpr[4], 0, 0));
        { uint64_t tmp = gf_ld64(b, ctx->gpr[1] + -0x10); memcpy(&ctx->fpr[30], &tmp, 8); }
        ctx->fpr[5] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[31], ctx->fpr[5], ctx->fpr[7], 0, 0));
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        ctx->fpr[6] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[6], ctx->fpr[28], ctx->fpr[10], 0, 0));
        ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[11], ctx->fpr[13], 0, 0));
        { float ftmp = (float)ctx->fpr[5]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[3] + 0x24, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
        { float ftmp = (float)ctx->fpr[6]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[3] + 0x18, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[6] = ftmp; }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[6]));
        ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[7]));
        ctx->fpr[8] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[8], ctx->fpr[3], ctx->fpr[0], 0, 0));
        { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[3] + 0x30, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        ctx->fpr[1] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[1], ctx->fpr[9], 0, 0));
        ctx->fpr[31] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[31], ctx->fpr[28], ctx->fpr[8], 0, 0));
        { uint64_t tmp = gf_ld64(b, ctx->gpr[1] + -0x20); memcpy(&ctx->fpr[28], &tmp, 8); }
        ctx->fpr[10] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[10], ctx->fpr[1], 0, 0));
        { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[3] + 0x28, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[5] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
        { uint64_t tmp = gf_ld64(b, ctx->gpr[1] + -0x8); memcpy(&ctx->fpr[31], &tmp, 8); }
        ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[5]));
        { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[3] + 0x34, tmp); }
        { uint32_t tmp = gf_ld32(b, ctx->gpr[4] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[6]));
        ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[3], ctx->fpr[0], 0, 0));
        ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[11], ctx->fpr[13], 0, 0));
        ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[2]));
        { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(b, ctx->gpr[3] + 0x38, tmp); }
        return 1;
}

/* ---- v1129N GF_HOTFN2: register-promoted copies of lifted bodies. Generated mechanically from recompiled/ppu_recomp_00N.cpp
 * (guest registers in locals, vm_read/vm_write -> the lean accessors, ctx spilled before / reloaded after every call,
 * written back at exit). Regenerate if the lift of one of these functions changes. ---- */
static void gf_hotfn2_resolve(void) { }

/* v1129N: func_0054665C with the guest registers it touches held in locals (lift body otherwise unchanged:
 * same operations in the same order, same guest loads/stores through the lean accessors); ctx written once at exit. */
int gf_hot_0054665C(ppu_context* ctx) {
        uint8_t* const gf_vmb = vm_base;
        if (!(s_gf_hotfn2 & GF_HOTFN2_0054665C) || g_v243a_sel != 3) return 0;
        if (s_gf_hotfn2 & GF_HOTFN2_LEANNAN) return gf_hot3_0054665C(ctx);   /* v1129K */
        uint64_t r0 = ctx->gpr[0];
        uint64_t r1 = ctx->gpr[1];
        uint64_t r2 = ctx->gpr[2];
        uint64_t r3 = ctx->gpr[3];
        uint64_t r4 = ctx->gpr[4];
        uint64_t r9 = ctx->gpr[9];
        uint64_t r11 = ctx->gpr[11];
        double f0 = ctx->fpr[0];
        double f2 = ctx->fpr[2];
        double f3 = ctx->fpr[3];
        double f4 = ctx->fpr[4];
        double f5 = ctx->fpr[5];
        double f6 = ctx->fpr[6];
        double f7 = ctx->fpr[7];
        double f8 = ctx->fpr[8];
        double f9 = ctx->fpr[9];
        double f10 = ctx->fpr[10];
        double f11 = ctx->fpr[11];
        double f12 = ctx->fpr[12];
        double f13 = ctx->fpr[13];
        r9 = r3 + (int64_t)(0x10);
        r11 = gf_ld32(gf_vmb, r2 + -0x7394);
        { uint32_t tmp = gf_ld32(gf_vmb, r3 + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); f0 = ftmp; }
        r9 = ppc_rldicl(r9, 0, 32);
        { uint32_t tmp = gf_ld32(gf_vmb, r11 + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); f11 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); f2 = ftmp; }
        f11 = ppu_fp_single(ppu_fadd(f11, f11));
        { uint32_t tmp = gf_ld32(gf_vmb, r9 + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); f10 = ftmp; }
        f5 = fabs(f2);
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); f9 = ftmp; }
        f6 = ppu_fp_single(ppu_fsub(f10, f0));
        f9 = ppu_fp_single(ppu_fadd(f9, f9));
        { uint32_t tmp = gf_ld32(gf_vmb, r3 + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); f13 = ftmp; }
        f10 = ppu_fp_single(ppu_fadd(f10, f0));
        { uint32_t tmp = gf_ld32(gf_vmb, r9 + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); f12 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); f4 = ftmp; }
        f8 = ppu_fp_single(ppu_fsub(f12, f13));
        f3 = fabs(f4);
        { uint32_t tmp = gf_ld32(gf_vmb, r3 + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); f7 = ftmp; }
        f12 = ppu_fp_single(ppu_fadd(f12, f13));
        { uint32_t tmp = gf_ld32(gf_vmb, r9 + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); f0 = ftmp; }
        f13 = ppu_fp_single(ppu_fsub(f0, f7));
        f0 = ppu_fp_single(ppu_fadd(f0, f7));
        f6 = ppu_fp_single(ppu_fmadd_core(f6, f5, f11, 0, 0));
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); f5 = ftmp; }
        f11 = fabs(f5);
        f10 = ppu_fp_single(ppu_fmadd_core(f10, f2, f9, 0, 0));
        f8 = ppu_fp_single(ppu_fmadd_core(f8, f3, f6, 0, 0));
        f12 = ppu_fp_single(ppu_fmadd_core(f12, f4, f10, 0, 0));
        f13 = ppu_fp_single(ppu_fmadd_core(f13, f11, f8, 0, 0));
        f0 = ppu_fp_single(ppu_fmadd_core(f0, f5, f12, 0, 0));
        f12 = ppu_fp_single(ppu_fadd(f0, f13));
        f13 = ppu_fp_single(ppu_fsub(f13, f0));
        { float ftmp = (float)f12; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r1 + -0x10, tmp); }
        r0 = gf_ld32(gf_vmb, r1 + -0x10);
        { float ftmp = (float)f13; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r1 + -0x10, tmp); }
        r3 = gf_ld32(gf_vmb, r1 + -0x10);
        r0 = (uint64_t)ppc_rlwinm((uint32_t)r0, 1, 31, 31);
        r3 = (uint64_t)ppc_rlwinm((uint32_t)r3, 1, 31, 31);
        r3 = r3 - r0;
        r3 = r3 + (int64_t)(2);
        r3 = ppc_rldicl(r3, 0, 32);
        goto gf_wb;
gf_wb:
        ctx->gpr[0] = r0;
        ctx->gpr[1] = r1;
        ctx->gpr[2] = r2;
        ctx->gpr[3] = r3;
        ctx->gpr[4] = r4;
        ctx->gpr[9] = r9;
        ctx->gpr[11] = r11;
        ctx->fpr[0] = f0;
        ctx->fpr[2] = f2;
        ctx->fpr[3] = f3;
        ctx->fpr[4] = f4;
        ctx->fpr[5] = f5;
        ctx->fpr[6] = f6;
        ctx->fpr[7] = f7;
        ctx->fpr[8] = f8;
        ctx->fpr[9] = f9;
        ctx->fpr[10] = f10;
        ctx->fpr[11] = f11;
        ctx->fpr[12] = f12;
        ctx->fpr[13] = f13;
        return 1;
}

/* v1129N: func_0070AF14 with the guest registers it touches held in locals (lift body otherwise unchanged:
 * same operations in the same order, same guest loads/stores through the lean accessors); ctx written once at exit. */
int gf_hot2_0070AF14(ppu_context* ctx) {
        uint8_t* const gf_vmb = vm_base;
        if (!(s_gf_hotfn2 & GF_HOTFN2_0070AF14) || g_v243a_sel != 3) return 0;
        if (s_gf_hotfn2 & GF_HOTFN2_LEANNAN) return gf_hot3_0070AF14(ctx);   /* v1129K */
        uint64_t r1 = ctx->gpr[1];
        uint64_t r3 = ctx->gpr[3];
        uint64_t r4 = ctx->gpr[4];
        uint64_t r5 = ctx->gpr[5];
        double f0 = ctx->fpr[0];
        double f1 = ctx->fpr[1];
        double f2 = ctx->fpr[2];
        double f3 = ctx->fpr[3];
        double f4 = ctx->fpr[4];
        double f5 = ctx->fpr[5];
        double f6 = ctx->fpr[6];
        double f7 = ctx->fpr[7];
        double f8 = ctx->fpr[8];
        double f9 = ctx->fpr[9];
        double f10 = ctx->fpr[10];
        double f11 = ctx->fpr[11];
        double f12 = ctx->fpr[12];
        double f13 = ctx->fpr[13];
        double f27 = ctx->fpr[27];
        double f28 = ctx->fpr[28];
        double f29 = ctx->fpr[29];
        double f30 = ctx->fpr[30];
        double f31 = ctx->fpr[31];
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); f10 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); f9 = ftmp; }
        f5 = ppu_fp_single(ppu_fmul(f9, f10));
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); f7 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); f11 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); f3 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); f2 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); f8 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); f13 = ftmp; }
        f9 = ppu_fp_single(ppu_fmul(f9, f8));
        f10 = ppu_fp_single(ppu_fmul(f13, f10));
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); f12 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); f0 = ftmp; }
        f13 = ppu_fp_single(ppu_fmul(f13, f8));
        { uint64_t tmp; memcpy(&tmp, &f30, 8); gf_st64(gf_vmb, r1 + -0x10, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); f6 = ftmp; }
        f11 = ppu_fp_single(ppu_fmadd_core(f11, f7, f5, 0, 0));
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); f5 = ftmp; }
        { uint64_t tmp; memcpy(&tmp, &f27, 8); gf_st64(gf_vmb, r1 + -0x28, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); f27 = ftmp; }
        { uint64_t tmp; memcpy(&tmp, &f29, 8); gf_st64(gf_vmb, r1 + -0x18, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); f29 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); f7 = ftmp; }
        { uint64_t tmp; memcpy(&tmp, &f28, 8); gf_st64(gf_vmb, r1 + -0x20, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); f28 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); f8 = ftmp; }
        { uint64_t tmp; memcpy(&tmp, &f31, 8); gf_st64(gf_vmb, r1 + -0x8, tmp); }
        f11 = ppu_fp_single(ppu_fmadd_core(f3, f2, f11, 0, 0));
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); f31 = ftmp; }
        { float ftmp = (float)f11; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x0, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); f11 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); f30 = ftmp; }
        f12 = ppu_fp_single(ppu_fmadd_core(f11, f12, f9, 0, 0));
        f0 = ppu_fp_single(ppu_fmadd_core(f0, f30, f10, 0, 0));
        f12 = ppu_fp_single(ppu_fmadd_core(f3, f5, f12, 0, 0));
        f0 = ppu_fp_single(ppu_fmadd_core(f6, f2, f0, 0, 0));
        { float ftmp = (float)f12; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x4, tmp); }
        { float ftmp = (float)f0; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x10, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); f0 = ftmp; }
        f0 = ppu_fp_single(ppu_fmul(f0, f27));
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); f1 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); f10 = ftmp; }
        f13 = ppu_fp_single(ppu_fmadd_core(f10, f1, f13, 0, 0));
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); f4 = ftmp; }
        f12 = ppu_fp_single(ppu_fmul(f7, f4));
        f11 = ppu_fp_single(ppu_fmadd_core(f11, f29, f0, 0, 0));
        f13 = ppu_fp_single(ppu_fmadd_core(f6, f5, f13, 0, 0));
        f8 = ppu_fp_single(ppu_fmadd_core(f8, f30, f12, 0, 0));
        f3 = ppu_fp_single(ppu_fmadd_core(f3, f28, f11, 0, 0));
        { float ftmp = (float)f13; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x14, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); f29 = ftmp; }
        f2 = ppu_fp_single(ppu_fmadd_core(f31, f2, f8, 0, 0));
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); f0 = ftmp; }
        f7 = ppu_fp_single(ppu_fmul(f7, f29));
        f0 = ppu_fp_single(ppu_fmul(f0, f27));
        { float ftmp = (float)f3; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x8, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); f9 = ftmp; }
        f4 = ppu_fp_single(ppu_fmul(f9, f4));
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); f13 = ftmp; }
        { float ftmp = (float)f2; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x20, tmp); }
        f9 = ppu_fp_single(ppu_fmul(f9, f29));
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); f8 = ftmp; }
        f7 = ppu_fp_single(ppu_fmadd_core(f8, f1, f7, 0, 0));
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); f12 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); f11 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); f3 = ftmp; }
        f10 = ppu_fp_single(ppu_fmadd_core(f10, f3, f0, 0, 0));
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); f2 = ftmp; }
        { uint64_t tmp = gf_ld64(gf_vmb, r1 + -0x28); memcpy(&f27, &tmp, 8); }
        { uint64_t tmp = gf_ld64(gf_vmb, r1 + -0x18); memcpy(&f29, &tmp, 8); }
        f13 = ppu_fp_single(ppu_fmadd_core(f13, f30, f4, 0, 0));
        { uint64_t tmp = gf_ld64(gf_vmb, r1 + -0x10); memcpy(&f30, &tmp, 8); }
        f5 = ppu_fp_single(ppu_fmadd_core(f31, f5, f7, 0, 0));
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); f7 = ftmp; }
        f6 = ppu_fp_single(ppu_fmadd_core(f6, f28, f10, 0, 0));
        f11 = ppu_fp_single(ppu_fmadd_core(f12, f11, f13, 0, 0));
        { float ftmp = (float)f5; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x24, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); f10 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); f5 = ftmp; }
        { float ftmp = (float)f6; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x18, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); f6 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); f0 = ftmp; }
        f0 = ppu_fp_single(ppu_fmul(f0, f6));
        f11 = ppu_fp_single(ppu_fadd(f11, f7));
        f8 = ppu_fp_single(ppu_fmadd_core(f8, f3, f0, 0, 0));
        { float ftmp = (float)f11; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x30, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); f13 = ftmp; }
        f1 = ppu_fp_single(ppu_fmadd_core(f13, f1, f9, 0, 0));
        f31 = ppu_fp_single(ppu_fmadd_core(f31, f28, f8, 0, 0));
        { uint64_t tmp = gf_ld64(gf_vmb, r1 + -0x20); memcpy(&f28, &tmp, 8); }
        f10 = ppu_fp_single(ppu_fmadd_core(f12, f10, f1, 0, 0));
        { float ftmp = (float)f31; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x28, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); f11 = ftmp; }
        { uint64_t tmp = gf_ld64(gf_vmb, r1 + -0x8); memcpy(&f31, &tmp, 8); }
        f10 = ppu_fp_single(ppu_fadd(f10, f5));
        { float ftmp = (float)f10; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x34, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); f0 = ftmp; }
        f0 = ppu_fp_single(ppu_fmul(f0, f6));
        f13 = ppu_fp_single(ppu_fmadd_core(f13, f3, f0, 0, 0));
        f12 = ppu_fp_single(ppu_fmadd_core(f12, f11, f13, 0, 0));
        f12 = ppu_fp_single(ppu_fadd(f12, f2));
        { float ftmp = (float)f12; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x38, tmp); }
        goto gf_wb;
gf_wb:
        ctx->gpr[1] = r1;
        ctx->gpr[3] = r3;
        ctx->gpr[4] = r4;
        ctx->gpr[5] = r5;
        ctx->fpr[0] = f0;
        ctx->fpr[1] = f1;
        ctx->fpr[2] = f2;
        ctx->fpr[3] = f3;
        ctx->fpr[4] = f4;
        ctx->fpr[5] = f5;
        ctx->fpr[6] = f6;
        ctx->fpr[7] = f7;
        ctx->fpr[8] = f8;
        ctx->fpr[9] = f9;
        ctx->fpr[10] = f10;
        ctx->fpr[11] = f11;
        ctx->fpr[12] = f12;
        ctx->fpr[13] = f13;
        ctx->fpr[27] = f27;
        ctx->fpr[28] = f28;
        ctx->fpr[29] = f29;
        ctx->fpr[30] = f30;
        ctx->fpr[31] = f31;
        return 1;
}


/* ---- v1129K GF_HOTFN2 bit 2: lean NaN tests for the register-promoted bodies (bit 0 / bit 1).
 * The lift's ppu_fp_bin / ppu_fmadd_core test every input for NaN, then the result (3-4 compare+branch per op), and
 * ppu_fp_single tests again. On x86 a NaN input always gives a NaN result (addsd/subsd/mulsd/vfmadd*), so ONE test of
 * the raw host result is enough to know that no input and no result was NaN -- and then the lift returns exactly that
 * raw result (rounded to single). If the raw result IS NaN we redo the op with the lift's own helpers, which pick the
 * PPC NaN (quieted first NaN operand / default 0x7FF8.. qNaN). Same ops, same order, same rounding -> bit-identical. */
GF_HOT_INL double gf_fadds(double a, double b)
{
    const double r = a + b;
    if (__builtin_expect(r != r, 0)) return ppu_fp_single(ppu_fadd(a, b));
    return (double)(float)r;
}
GF_HOT_INL double gf_fsubs(double a, double b)
{
    const double r = a - b;
    if (__builtin_expect(r != r, 0)) return ppu_fp_single(ppu_fsub(a, b));
    return (double)(float)r;
}
GF_HOT_INL double gf_fmuls(double a, double b)
{
    const double r = a * b;
    if (__builtin_expect(r != r, 0)) return ppu_fp_single(ppu_fmul(a, b));
    return (double)(float)r;
}
GF_HOT_INL double gf_fmas(double a, double c, double b, int neg_b, int neg_res)
{
    const double r = fma(a, c, neg_b ? -b : b);
    if (__builtin_expect(r != r, 0)) return ppu_fp_single(ppu_fmadd_core(a, c, b, neg_b, neg_res));
    return (double)(float)(neg_res ? -r : r);
}

/* v1129K: the bit-0 / bit-1 bodies of v1129N with the lean NaN tests above (generated: ppu_fp_single(ppu_fX(..)) ->
 * gf_fXs(..); nothing else changed). Called only from gf_hot_0054665C / gf_hot2_0070AF14 after their gates. */
static int gf_hot3_0070AF14(ppu_context* ctx) {
        uint8_t* const gf_vmb = vm_base;
        uint64_t r1 = ctx->gpr[1];
        uint64_t r3 = ctx->gpr[3];
        uint64_t r4 = ctx->gpr[4];
        uint64_t r5 = ctx->gpr[5];
        double f0 = ctx->fpr[0];
        double f1 = ctx->fpr[1];
        double f2 = ctx->fpr[2];
        double f3 = ctx->fpr[3];
        double f4 = ctx->fpr[4];
        double f5 = ctx->fpr[5];
        double f6 = ctx->fpr[6];
        double f7 = ctx->fpr[7];
        double f8 = ctx->fpr[8];
        double f9 = ctx->fpr[9];
        double f10 = ctx->fpr[10];
        double f11 = ctx->fpr[11];
        double f12 = ctx->fpr[12];
        double f13 = ctx->fpr[13];
        double f27 = ctx->fpr[27];
        double f28 = ctx->fpr[28];
        double f29 = ctx->fpr[29];
        double f30 = ctx->fpr[30];
        double f31 = ctx->fpr[31];
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); f10 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); f9 = ftmp; }
        f5 = gf_fmuls(f9, f10);
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); f7 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); f11 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); f3 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); f2 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); f8 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); f13 = ftmp; }
        f9 = gf_fmuls(f9, f8);
        f10 = gf_fmuls(f13, f10);
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); f12 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); f0 = ftmp; }
        f13 = gf_fmuls(f13, f8);
        { uint64_t tmp; memcpy(&tmp, &f30, 8); gf_st64(gf_vmb, r1 + -0x10, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); f6 = ftmp; }
        f11 = gf_fmas(f11, f7, f5, 0, 0);
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); f5 = ftmp; }
        { uint64_t tmp; memcpy(&tmp, &f27, 8); gf_st64(gf_vmb, r1 + -0x28, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); f27 = ftmp; }
        { uint64_t tmp; memcpy(&tmp, &f29, 8); gf_st64(gf_vmb, r1 + -0x18, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); f29 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); f7 = ftmp; }
        { uint64_t tmp; memcpy(&tmp, &f28, 8); gf_st64(gf_vmb, r1 + -0x20, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); f28 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); f8 = ftmp; }
        { uint64_t tmp; memcpy(&tmp, &f31, 8); gf_st64(gf_vmb, r1 + -0x8, tmp); }
        f11 = gf_fmas(f3, f2, f11, 0, 0);
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); f31 = ftmp; }
        { float ftmp = (float)f11; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x0, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); f11 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); f30 = ftmp; }
        f12 = gf_fmas(f11, f12, f9, 0, 0);
        f0 = gf_fmas(f0, f30, f10, 0, 0);
        f12 = gf_fmas(f3, f5, f12, 0, 0);
        f0 = gf_fmas(f6, f2, f0, 0, 0);
        { float ftmp = (float)f12; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x4, tmp); }
        { float ftmp = (float)f0; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x10, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); f0 = ftmp; }
        f0 = gf_fmuls(f0, f27);
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); f1 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); f10 = ftmp; }
        f13 = gf_fmas(f10, f1, f13, 0, 0);
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); f4 = ftmp; }
        f12 = gf_fmuls(f7, f4);
        f11 = gf_fmas(f11, f29, f0, 0, 0);
        f13 = gf_fmas(f6, f5, f13, 0, 0);
        f8 = gf_fmas(f8, f30, f12, 0, 0);
        f3 = gf_fmas(f3, f28, f11, 0, 0);
        { float ftmp = (float)f13; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x14, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); f29 = ftmp; }
        f2 = gf_fmas(f31, f2, f8, 0, 0);
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); f0 = ftmp; }
        f7 = gf_fmuls(f7, f29);
        f0 = gf_fmuls(f0, f27);
        { float ftmp = (float)f3; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x8, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); f9 = ftmp; }
        f4 = gf_fmuls(f9, f4);
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); f13 = ftmp; }
        { float ftmp = (float)f2; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x20, tmp); }
        f9 = gf_fmuls(f9, f29);
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); f8 = ftmp; }
        f7 = gf_fmas(f8, f1, f7, 0, 0);
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); f12 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); f11 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); f3 = ftmp; }
        f10 = gf_fmas(f10, f3, f0, 0, 0);
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); f2 = ftmp; }
        { uint64_t tmp = gf_ld64(gf_vmb, r1 + -0x28); memcpy(&f27, &tmp, 8); }
        { uint64_t tmp = gf_ld64(gf_vmb, r1 + -0x18); memcpy(&f29, &tmp, 8); }
        f13 = gf_fmas(f13, f30, f4, 0, 0);
        { uint64_t tmp = gf_ld64(gf_vmb, r1 + -0x10); memcpy(&f30, &tmp, 8); }
        f5 = gf_fmas(f31, f5, f7, 0, 0);
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); f7 = ftmp; }
        f6 = gf_fmas(f6, f28, f10, 0, 0);
        f11 = gf_fmas(f12, f11, f13, 0, 0);
        { float ftmp = (float)f5; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x24, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); f10 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); f5 = ftmp; }
        { float ftmp = (float)f6; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x18, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); f6 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); f0 = ftmp; }
        f0 = gf_fmuls(f0, f6);
        f11 = gf_fadds(f11, f7);
        f8 = gf_fmas(f8, f3, f0, 0, 0);
        { float ftmp = (float)f11; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x30, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); f13 = ftmp; }
        f1 = gf_fmas(f13, f1, f9, 0, 0);
        f31 = gf_fmas(f31, f28, f8, 0, 0);
        { uint64_t tmp = gf_ld64(gf_vmb, r1 + -0x20); memcpy(&f28, &tmp, 8); }
        f10 = gf_fmas(f12, f10, f1, 0, 0);
        { float ftmp = (float)f31; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x28, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r5 + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); f11 = ftmp; }
        { uint64_t tmp = gf_ld64(gf_vmb, r1 + -0x8); memcpy(&f31, &tmp, 8); }
        f10 = gf_fadds(f10, f5);
        { float ftmp = (float)f10; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x34, tmp); }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); f0 = ftmp; }
        f0 = gf_fmuls(f0, f6);
        f13 = gf_fmas(f13, f3, f0, 0, 0);
        f12 = gf_fmas(f12, f11, f13, 0, 0);
        f12 = gf_fadds(f12, f2);
        { float ftmp = (float)f12; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r3 + 0x38, tmp); }
        goto gf_wb;
gf_wb:
        ctx->gpr[1] = r1;
        ctx->gpr[3] = r3;
        ctx->gpr[4] = r4;
        ctx->gpr[5] = r5;
        ctx->fpr[0] = f0;
        ctx->fpr[1] = f1;
        ctx->fpr[2] = f2;
        ctx->fpr[3] = f3;
        ctx->fpr[4] = f4;
        ctx->fpr[5] = f5;
        ctx->fpr[6] = f6;
        ctx->fpr[7] = f7;
        ctx->fpr[8] = f8;
        ctx->fpr[9] = f9;
        ctx->fpr[10] = f10;
        ctx->fpr[11] = f11;
        ctx->fpr[12] = f12;
        ctx->fpr[13] = f13;
        ctx->fpr[27] = f27;
        ctx->fpr[28] = f28;
        ctx->fpr[29] = f29;
        ctx->fpr[30] = f30;
        ctx->fpr[31] = f31;
        return 1;
}

static int gf_hot3_0054665C(ppu_context* ctx) {
        uint8_t* const gf_vmb = vm_base;
        uint64_t r0 = ctx->gpr[0];
        uint64_t r1 = ctx->gpr[1];
        uint64_t r2 = ctx->gpr[2];
        uint64_t r3 = ctx->gpr[3];
        uint64_t r4 = ctx->gpr[4];
        uint64_t r9 = ctx->gpr[9];
        uint64_t r11 = ctx->gpr[11];
        double f0 = ctx->fpr[0];
        double f2 = ctx->fpr[2];
        double f3 = ctx->fpr[3];
        double f4 = ctx->fpr[4];
        double f5 = ctx->fpr[5];
        double f6 = ctx->fpr[6];
        double f7 = ctx->fpr[7];
        double f8 = ctx->fpr[8];
        double f9 = ctx->fpr[9];
        double f10 = ctx->fpr[10];
        double f11 = ctx->fpr[11];
        double f12 = ctx->fpr[12];
        double f13 = ctx->fpr[13];
        r9 = r3 + (int64_t)(0x10);
        r11 = gf_ld32(gf_vmb, r2 + -0x7394);
        { uint32_t tmp = gf_ld32(gf_vmb, r3 + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); f0 = ftmp; }
        r9 = ppc_rldicl(r9, 0, 32);
        { uint32_t tmp = gf_ld32(gf_vmb, r11 + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); f11 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); f2 = ftmp; }
        f11 = gf_fadds(f11, f11);
        { uint32_t tmp = gf_ld32(gf_vmb, r9 + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); f10 = ftmp; }
        f5 = fabs(f2);
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); f9 = ftmp; }
        f6 = gf_fsubs(f10, f0);
        f9 = gf_fadds(f9, f9);
        { uint32_t tmp = gf_ld32(gf_vmb, r3 + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); f13 = ftmp; }
        f10 = gf_fadds(f10, f0);
        { uint32_t tmp = gf_ld32(gf_vmb, r9 + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); f12 = ftmp; }
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); f4 = ftmp; }
        f8 = gf_fsubs(f12, f13);
        f3 = fabs(f4);
        { uint32_t tmp = gf_ld32(gf_vmb, r3 + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); f7 = ftmp; }
        f12 = gf_fadds(f12, f13);
        { uint32_t tmp = gf_ld32(gf_vmb, r9 + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); f0 = ftmp; }
        f13 = gf_fsubs(f0, f7);
        f0 = gf_fadds(f0, f7);
        f6 = gf_fmas(f6, f5, f11, 0, 0);
        { uint32_t tmp = gf_ld32(gf_vmb, r4 + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); f5 = ftmp; }
        f11 = fabs(f5);
        f10 = gf_fmas(f10, f2, f9, 0, 0);
        f8 = gf_fmas(f8, f3, f6, 0, 0);
        f12 = gf_fmas(f12, f4, f10, 0, 0);
        f13 = gf_fmas(f13, f11, f8, 0, 0);
        f0 = gf_fmas(f0, f5, f12, 0, 0);
        f12 = gf_fadds(f0, f13);
        f13 = gf_fsubs(f13, f0);
        { float ftmp = (float)f12; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r1 + -0x10, tmp); }
        r0 = gf_ld32(gf_vmb, r1 + -0x10);
        { float ftmp = (float)f13; uint32_t tmp; memcpy(&tmp, &ftmp, 4); gf_st32(gf_vmb, r1 + -0x10, tmp); }
        r3 = gf_ld32(gf_vmb, r1 + -0x10);
        r0 = (uint64_t)ppc_rlwinm((uint32_t)r0, 1, 31, 31);
        r3 = (uint64_t)ppc_rlwinm((uint32_t)r3, 1, 31, 31);
        r3 = r3 - r0;
        r3 = r3 + (int64_t)(2);
        r3 = ppc_rldicl(r3, 0, 32);
        goto gf_wb;
gf_wb:
        ctx->gpr[0] = r0;
        ctx->gpr[1] = r1;
        ctx->gpr[2] = r2;
        ctx->gpr[3] = r3;
        ctx->gpr[4] = r4;
        ctx->gpr[9] = r9;
        ctx->gpr[11] = r11;
        ctx->fpr[0] = f0;
        ctx->fpr[2] = f2;
        ctx->fpr[3] = f3;
        ctx->fpr[4] = f4;
        ctx->fpr[5] = f5;
        ctx->fpr[6] = f6;
        ctx->fpr[7] = f7;
        ctx->fpr[8] = f8;
        ctx->fpr[9] = f9;
        ctx->fpr[10] = f10;
        ctx->fpr[11] = f11;
        ctx->fpr[12] = f12;
        ctx->fpr[13] = f13;
        return 1;
}

#if GF_HOTFN_ISA
#pragma clang attribute pop
#endif
