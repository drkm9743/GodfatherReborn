#!/usr/bin/env python3
"""v1039A: lift Godfather's RWAC audio-mixer modules (runtime-loaded SPU code in SPU thread 0x2001's local store) as
FIXED-address, content-verified "plugin" functions served through the runtime's spu_plug_lookup (ABX 0x10 + 0x01, the
inFamous v133/v137 mechanism, tools/plug_lift.py). A lifted function runs only when its own bytes in LS match the
capture exactly; anything else (another module, a partly loaded buffer) still goes to the interpreter.

Usage: gf_mix_plug_lift.py <LS dump (256 KB)> [more dumps ...] <outdir>   (v1039B: relocatable, every module id found)
Modules (header magic a122e6a2 .. a122e6a2 at -0x20; code bounded by control-flow reachability from the hot entries):"""
import sys, os, struct
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', '..', 'ps3recomp', 'tools'))
from spu_disasm import disassemble_spu
import spu_lifter as L

MAGIC = bytes.fromhex('a122e6a2')
# Observed interpreter-fallback pcs (absolute, per capture layout) -- seeds for function starts / reachability.
SEEDS_ABS = [0x9DA0, 0xAD20, 0xCD20, 0x0D1C0, 0x0A380, 0x0CE60, 0x0AD70, 0x0A7D4, 0x0EF8C, 0x0F274, 0x0A508, 0x0D108,
             0x0B0A4, 0x0D7B0, 0x09DA8, 0x0AD28, 0x0CD28, 0x0A7B0, 0x09DB0, 0x0A3A0, 0x0AD30, 0x0CD30, 0x0CE70, 0x0D1D8,
             # world (e10)
             0x0F9D4, 0x123C0, 0x14474, 0x0F580, 0x102A4, 0x1418C, 0x12060, 0x0F708, 0x0FF70, 0x0FF88, 0x129B0, 0x10248,
             0x0F9B0, 0x0AA70, 0x12308, 0x0F6F8, 0x0EFA8, 0x0FF28, 0x11F28, 0x0C630, 0x0FF20, 0x0EFA0, 0x11F20]
# v1077A: seeds that belong to ONE capture's layout only (abs pcs of that dump). Gunfight [spu-miss] 2026-11-06: module 2
# (2102deaa..) entries reached only through a pointer (0x0FB00) and a PC-getter function (0x0FD58, `brsl $r126, .+4`),
# seen at 0x0FB00/0x0FD58 (city layout = ls_0D1C0) and 0x14D00/0x14F58 (fight layout, module 0x5200 higher).
SEEDS_BY_DUMP = {'ls_0D1C0.bin': [0x0FB00, 0x0FD58]}
# v1129U: modules 4..8 (ids below; in ls_0F9D4.bin at 0x28B80 / 0x29500 / 0x2B100 / 0x2E680 / 0x2EC80) were lifted as their
# 0x40-byte relocation stub only: the stub (+0x20) stores header+`ila $r3,X` into a pointer table and returns (`bi $r0`), so
# reachability from +0x20 never saw the body. The body is entered through that pointer -- in the world at LS >= 0x15000,
# where it ran in the interpreter (RR_21 [PROF] SPU 0x2001: ~26 % of the mixer thread's samples in
# spu_interp_run_until_fast). Seeds = the stub's pointer target (+0x58; module 3a7e.. +0x290) and +0x58 (first code after
# the stub). Emitted as EXTRA blobs (gate 2, served only with GF_MIX_HI bit 1); the original stub blobs get gate 1.
SEEDS_REL_1129U = {bytes.fromhex('9b33436e59672e89'): [0x58], bytes.fromhex('3a7e1eee34f4e011'): [0x58, 0x290],
                   bytes.fromhex('c72945674949bcbf'): [0x58], bytes.fromhex('b22533988a50772d'): [0x58],
                   bytes.fromhex('9b8f934962a5de1d'): [0x58]}

def find_modules(ls):
    """All a122e6a2 job-module headers above the low-LS program (0x80 = the static live_01ee0 program, skipped)."""
    hs = [a for a in range(0x100, 0x40000 - 0x20, 16) if ls[a:a+4] == MAGIC and ls[a+12:a+16] == MAGIC]
    return hs

def reach_end(ls, lo, hi, seeds):
    """Highest reachable instruction (+4) from the module entry and the seeds, following direct control flow."""
    ins = [i for i in disassemble_spu(ls[lo:hi], lo)]
    by = {i.addr: i for i in ins}; seen = set(); st = [s for s in seeds if lo <= s < hi]
    while st:
        a = st.pop()
        while a in by and a not in seen:
            seen.add(a); i = by[a]; m = i.mnemonic; t = L.SPULifter._branch_target(i)
            if m.startswith('.') or m in ('stop', 'stopd'): break
            if t is not None and lo <= t < hi: st.append(t)
            if m in ('br', 'bra', 'bi', 'iret'): break
            a += 4
    return ((max(seen) + 4 + 15) & ~15) if seen else lo + 0x10

def fnv32(bs):
    h = 2166136261
    for x in bs: h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return h

def main():
    *dumps, outdir = sys.argv[1:]
    mods = {}; rels = {}   # module id (8 bytes) -> (ls, header, next header, dump name); id -> set of entry offsets
    for dp in dumps:
        ls = open(dp, 'rb').read(); assert len(ls) >= 0x40000
        hs = find_modules(ls)
        for j, h in enumerate(hs):
            mid = ls[h+4:h+12]
            nxt = hs[j+1] if j + 1 < len(hs) else 0x40000
            rels.setdefault(mid, {0x20}).update(x - h for x in SEEDS_ABS if h + 0x20 <= x < nxt)
            rels[mid].update(x - h for x in SEEDS_BY_DUMP.get(os.path.basename(dp), []) if h + 0x20 <= x < nxt)   # v1077A
            if mid not in mods: mods[mid] = (ls, h, nxt, os.path.basename(dp))
    for mid, (ls, h, nxt, dn) in list(mods.items()):
        seeds = sorted(h + r for r in rels[mid])
        dend = min(nxt, reach_end(ls, h + 0x20, nxt, seeds))
        mods[mid] = (ls, h, dend, seeds, dn)
    REGIONS = [(mid.hex(), v) for mid, v in mods.items()]
    GATE = [0] * len(REGIONS)   # v1129U: 0 = always, 1 = only with GF_MIX_HI bit 1 OFF, 2 = only with it ON
    for mid, rel in SEEDS_REL_1129U.items():
        if mid not in mods: continue
        ls, h, _dend, seeds, dn = mods[mid]
        hs = find_modules(ls); j = hs.index(h); nxt = hs[j + 1] if j + 1 < len(hs) else 0x40000
        seeds2 = sorted(set(seeds) | {h + r for r in rel})
        dend2 = min(nxt, reach_end(ls, h + 0x20, nxt, seeds2))
        GATE[[n for n, _ in REGIONS].index(mid.hex())] = 1
        REGIONS.append((mid.hex(), (ls, h, dend2, seeds2, dn))); GATE.append(2)
    os.makedirs(outdir, exist_ok=True)
    src = ['/* Auto-generated by spu/gf_mix_plug_lift.py (v1039A) -- content-verified lift of the RWAC mixer modules. Do not edit. */',
           '#include "spu_recomp.h"', '#include "spu_helpers.h"', '#include <string.h>', '#include <stdio.h>', '']
    hdr = ['/* Auto-generated by spu/gf_mix_plug_lift.py (v1039A). */', '#pragma once', '#include "spu_context.h"',
           '#ifdef __cplusplus', 'extern "C" {', '#endif',
           L.HEADER_PREAMBLE.split('#ifdef __cplusplus\nextern "C" {\n#endif', 1)[1].rsplit('#ifdef __cplusplus', 1)[0],
           '#if defined(_MSC_VER)', '#define SPU_PLUG_TLS __declspec(thread)', '#else', '#define SPU_PLUG_TLS _Thread_local', '#endif',
           'extern SPU_PLUG_TLS uint32_t g_spu_frag_delta;', 'typedef void (*spu_plug_fn)(spu_context*);',
           'spu_plug_fn spu_plug_lookup(spu_context* ctx, uint32_t pc);']
    entries = []; blobdata = []; report = []
    for k, (name, (ls, hdr0, dend, seeds, srcname)) in enumerate(REGIONS):
        lo = hdr0 + 0x20
        org = hdr0; raw = ls[hdr0:dend + 0x20]       # header (verified at runtime) + code; +0x20 hash window
        insns = [i for i in disassemble_spu(raw, org) if lo <= i.addr < dend]
        by = {i.addr: i for i in insns}
        starts = {lo} | {s for s in seeds if lo <= s < dend}
        for i in insns:
            t = L.SPULifter._branch_target(i)
            if i.mnemonic == 'brsl' and t is not None and lo <= t < dend and t not in (i.addr, i.addr + 4):
                starts.add(t)
            if i.mnemonic in ('brsl', 'bisl', 'brasl') and i.addr + 4 < dend and t != i.addr + 4:
                starts.add(i.addr + 4)
            if i.mnemonic in ('bi', 'br') and i.addr + 4 < dend:
                nxt = by.get(i.addr + 4)
                if nxt is not None and nxt.mnemonic not in ('.word', 'stop', 'stopd', 'lnop', 'nop'):
                    starts.add(i.addr + 4)
        for _ in range(20):
            ss = sorted(starts); bounds = list(zip(ss, ss[1:] + [dend])); new = set()
            for s, e in bounds:
                for i in insns:
                    if s <= i.addr < e:
                        t = L.SPULifter._branch_target(i)
                        if t is not None and lo <= t < dend and not (s <= t < e) and t not in starts and t % 4 == 0:
                            new.add(t)
            if not new: break
            starts |= new
        ss = sorted(starts); bounds = list(zip(ss, ss[1:] + [dend]))
        lf = L.SPULifter(prefix=f"gfmix{k:02d}_")
        lf.reloc = True
        allins = [i for i in disassemble_spu(raw, org) if i.addr < dend]
        lf.bi_r0_jump = L.compute_bi_r0_jumps(allins, bounds)
        lf.link_return = L.compute_link_returns(allins, bounds)
        lf.func_starts = set(ss) | {dend}
        for s, e in bounds:
            lf.lift_function(allins, s, e)
        lf.branch_targets.add(dend)
        body = lf.emit_source('spu_recomp.h').split('\n')
        cut = body.index('/* Function table */'); pre_end = 0
        while pre_end < len(body) and not body[pre_end].startswith('void '): pre_end += 1
        src.append(f'/* ---- module {k} (id {name}, from {srcname} at LS 0x{hdr0:X}, code 0x{lo:X}..0x{dend:X}, relocatable): {len(lf.functions)} functions ---- */')
        src.extend(body[pre_end:cut])
        for f in lf.functions: hdr.append(f'void {f.name}(spu_context* ctx);')
        for t in sorted((lf.call_targets | lf.branch_targets) - {f.start_addr for f in lf.functions}):
            hdr.append(f'void gfmix{k:02d}_spu_func_{t:08X}(spu_context* ctx);')
        for s, e in bounds:
            flen = e - s
            entries.append((fnv32(raw[s - org:s - org + 32]), k, s, flen, f"gfmix{k:02d}_spu_func_{s:08X}"))
        blobdata.append((k, org, raw[:dend - org + 0x20]))
        report.append((k, name, hdr0, dend, len(lf.functions), dict(lf.unsupported)))
    src.append('/* ---- module images (verification) ---- */')
    for k, org, raw in blobdata:
        src.append(f'static const uint8_t s_gfmix_img{k:02d}[0x{len(raw):X}] = {{')
        for i in range(0, len(raw), 32): src.append('    ' + ','.join(str(x) for x in raw[i:i+32]) + ',')
        src.append('};')
    src.append('static const uint8_t* const s_gfmix_img[] = { ' + ', '.join(f's_gfmix_img{k:02d}' for k, *_ in blobdata) + ' };')
    src.append('static const uint32_t s_gfmix_org[] = { ' + ', '.join(f'0x{org:X}u' for k, org, _ in blobdata) + ' };')
    src.append('static const uint8_t s_gfmix_gate[] = { ' + ', '.join(str(g) for g in GATE) + ' };   /* v1129U */')
    src.append('#define GFMIX_N_HI_1129U %du' % sum(1 for e in entries if GATE[e[1]] == 2))
    entries.sort()
    pf = [0] * 2048
    for h, k, s, flen, fn in entries:
        org, raw = blobdata[k][1], blobdata[k][2]
        w = struct.unpack('<I', raw[s - org:s - org + 4])[0]
        b = ((w * 0x9E3779B1) & 0xFFFFFFFF) >> 16
        pf[b >> 5] |= 1 << (b & 31)
    src.append('static const uint32_t s_gfmix_pf[2048] = {')
    for i in range(0, 2048, 16): src.append('    ' + ','.join(f'0x{x:08X}u' for x in pf[i:i+16]) + ',')
    src.append('};')
    src.append('typedef struct { uint32_t h, off; uint16_t blob, len; spu_plug_fn fn; } gfmix_entry;')
    src.append(f'static const gfmix_entry s_gfmix_tab[{len(entries)}] = {{')
    for h, k, s, flen, fn in entries: src.append(f'    {{ 0x{h:08X}u, 0x{s:X}u, {k}, 0x{flen:X}, {fn} }},')
    src.append('};')
    src.append(r'''SPU_PLUG_TLS uint32_t g_spu_frag_delta;
#if defined(_WIN32)
#include <windows.h>
#endif
/* [gfmix] every ~10 s: lifted mixer calls served vs resolver misses that fell through (any code). Per-thread counters. */
static volatile long s_ok, s_no, s_f2pos, s_f2neg, s_fast2_init, s_fast2_on;
static volatile long long s_t0;
static SPU_PLUG_TLS unsigned t_ok, t_no, t_f2pos, t_f2neg;
static void gfmix_flush(void)
{
    InterlockedExchangeAdd(&s_ok, (long)t_ok); InterlockedExchangeAdd(&s_no, (long)t_no);
    InterlockedExchangeAdd(&s_f2pos, (long)t_f2pos); InterlockedExchangeAdd(&s_f2neg, (long)t_f2neg);
    t_ok = t_no = t_f2pos = t_f2neg = 0;
    long long now = (long long)GetTickCount64();
    if (!s_t0) s_t0 = now;
    if (now - s_t0 >= 10000 && InterlockedExchange64((volatile long long*)&s_t0, now) != now) {
        extern volatile long g_spu_plug_rejoins;
        fprintf(stderr, "[gfmix] v1039A 10s: %ld lifted mixer calls served, %ld other resolver misses, %ld interpreter hand-backs\n",
                InterlockedExchange(&s_ok, 0), InterlockedExchange(&s_no, 0), InterlockedExchange(&g_spu_plug_rejoins, 0));
        if (InterlockedCompareExchange(&s_fast2_on, 0, 0))
            fprintf(stderr, "[gfmix] v1129S fast2 cache: %ld lifted hits, %ld safe negative hits\n",
                    InterlockedExchange(&s_f2pos, 0), InterlockedExchange(&s_f2neg, 0));
    }
}
static inline void gfmix_stat(int ok) { if (ok) t_ok++; else t_no++; if (((t_ok + t_no) & 4095u) == 0) gfmix_flush(); }
static uint32_t gfmix_fnv32(const uint8_t* p) { uint32_t h = 2166136261u; for (int i = 0; i < 32; i++) h = (h ^ p[i]) * 16777619u; return h; }
#define GFMIX_FAST2_N 1024u
typedef struct {
    uint32_t pc, base;
    uint16_t tab;
    uint8_t kind, code[32];
} gfmix_fast2_ent;
static SPU_PLUG_TLS gfmix_fast2_ent t_gfmix_fast2[GFMIX_FAST2_N];
static volatile long s_fast2_diag_init, s_fast2_diag_on, s_fast2_diag_n;
static int gfmix_fast2_on(void)
{
    if (s_fast2_init == 2) return s_fast2_on != 0;   /* v1129H: decided -> plain loads, no locked op per probe */
    if (InterlockedCompareExchange(&s_fast2_init, 1, 0) == 0) {
        const char* e = getenv("GF_MIX_FAST2");
        const long on = (e && e[0] == '1' && e[1] == 0) ? 1 : 0;
        InterlockedExchange(&s_fast2_on, on);
        if (on) fprintf(stderr, "[v1129S] GF_MIX_FAST2 content-keyed resolver cache: ON\n");
        InterlockedExchange(&s_fast2_init, 2);
    } else {
        while (InterlockedCompareExchange(&s_fast2_init, 2, 2) != 2) SwitchToThread();
    }
    return InterlockedCompareExchange(&s_fast2_on, 0, 0) != 0;
}
static int gfmix_fast2_diag(void)
{
    if (InterlockedCompareExchange(&s_fast2_diag_init, 1, 0) == 0) {
        const char* e = getenv("GF_MIX_FAST2_DIAG");
        InterlockedExchange(&s_fast2_diag_on, (e && e[0] == '1' && e[1] == 0) ? 1 : 0);
        InterlockedExchange(&s_fast2_diag_init, 2);
    } else {
        while (InterlockedCompareExchange(&s_fast2_diag_init, 2, 2) != 2) SwitchToThread();
    }
    return InterlockedCompareExchange(&s_fast2_diag_on, 0, 0) != 0;
}
/* Relocatable: served wherever the module is loaded, only when the module's 32-byte header (magic + id) at the
 * implied base AND the function's full [start,end) instruction bytes match the capture; GF_MIX_LIFT=0 turns the
 * whole table off (interpreter, the old behaviour). Direct lifted transfers are not rechecked because modules load whole. */
static volatile long s_on_init;
static volatile long s_on = 1;
static int gfmix_on(void)
{
    if (InterlockedCompareExchange(&s_on_init, 1, 0) == 0) {
        const char* e = getenv("GF_MIX_LIFT");
        const long on = (e && *e == '0') ? 0 : 1;
        InterlockedExchange(&s_on, on);
        fprintf(stderr, "[gfmix] v1039A content-verified RWAC mixer lift (%u entries): %s (GF_MIX_LIFT)\n",
                (unsigned)(sizeof s_gfmix_tab / sizeof s_gfmix_tab[0]), on ? "ON" : "off");
        InterlockedExchange(&s_on_init, 2);
    } else {
        while (InterlockedCompareExchange(&s_on_init, 2, 2) != 2) SwitchToThread();
    }
    return InterlockedCompareExchange(&s_on, 0, 0) != 0;
}
/* v1129U GF_MIX_HI bit 1 (default 0): serve the extended module-body blobs (s_gfmix_gate 2) instead of the old stub-only
 * blobs of the same modules (gate 1). Off = exactly the old table. Decided once; plain loads afterwards. */
static volatile long s_hi_init, s_hi_on;
static int gfmix_hi_on(void)
{
    if (s_hi_init == 2) return s_hi_on != 0;
    if (InterlockedCompareExchange(&s_hi_init, 1, 0) == 0) {
        const char* e = getenv("GF_MIX_HI");
        const long on = (e && *e) ? ((strtol(e, 0, 0) >> 1) & 1) : 0;
        InterlockedExchange(&s_hi_on, on);
        if (on) fprintf(stderr, "[v1129U] GF_MIX_HI bit1: RWAC mixer module bodies 4..8 lifted (%u extra entries): ON\n",
                        (unsigned)GFMIX_N_HI_1129U);
        InterlockedExchange(&s_hi_init, 2);
    } else {
        while (InterlockedCompareExchange(&s_hi_init, 2, 2) != 2) SwitchToThread();
    }
    return InterlockedCompareExchange(&s_hi_on, 0, 0) != 0;
}
spu_plug_fn spu_plug_lookup_probe(spu_context* ctx, uint32_t pc, int* stable_miss);
spu_plug_fn spu_plug_lookup_probe(spu_context* ctx, uint32_t pc, int* stable_miss)
{
    if (stable_miss) *stable_miss = 0;
    /* v1129H: with GF_MIX_FAST2 on and GF_MIX_LIFT decided, read the decision with plain loads (gfmix_on() does a locked
     * compare-exchange on a line shared by every SPU host thread on each call). Same result as gfmix_on(). */
    if (!((s_fast2_init == 2 && s_fast2_on && s_on_init == 2) ? (s_on != 0) : gfmix_on())) return 0;
    pc &= SPU_LS_MASK;
    if (pc + 32 > SPU_LS_SIZE) return 0;
    { uint32_t w; memcpy(&w, ctx->ls + pc, 4); const uint32_t b = (w * 0x9E3779B1u) >> 16;
      if (!(s_gfmix_pf[b >> 5] & (1u << (b & 31)))) { if (stable_miss) *stable_miss = 1; gfmix_stat(0); return 0; } }
    const int fast2 = gfmix_fast2_on();
    const uint32_t ck = (pc >> 2) & (GFMIX_FAST2_N - 1u);
    gfmix_fast2_ent* const ce = &t_gfmix_fast2[ck];
    if (fast2 && ce->kind && ce->pc == pc && memcmp(ce->code, ctx->ls + pc, sizeof ce->code) == 0) {
        if (ce->kind == 2) {
            t_f2neg++; gfmix_stat(0); return 0;
        }
        const gfmix_entry* e = &s_gfmix_tab[ce->tab];
        const uint32_t rel = e->off - s_gfmix_org[e->blob];
        const uint8_t* img = s_gfmix_img[e->blob];
        if (ce->base + 32 <= SPU_LS_SIZE && pc + e->len <= SPU_LS_SIZE &&
            memcmp(ctx->ls + ce->base, img, 32) == 0 &&
            memcmp(ctx->ls + pc, img + rel, e->len) == 0) {
            g_spu_frag_delta = ce->base - s_gfmix_org[e->blob];
            t_f2pos++; gfmix_stat(1); return e->fn;
        }
        ce->kind = 0;
    }
    const uint32_t h = gfmix_fnv32(ctx->ls + pc);
    int lo = 0, hi = (int)(sizeof s_gfmix_tab / sizeof s_gfmix_tab[0]) - 1;
    while (lo < hi) { int m = (lo + hi) >> 1; if (s_gfmix_tab[m].h < h) lo = m + 1; else hi = m; }
    const gfmix_entry* hit = 0; uint32_t hit_base = 0;
    for (int i = lo; i < (int)(sizeof s_gfmix_tab / sizeof s_gfmix_tab[0]) && s_gfmix_tab[i].h == h; i++) {
        const gfmix_entry* e = &s_gfmix_tab[i];
        { const uint8_t g = s_gfmix_gate[e->blob]; if (g && (g == 2) != gfmix_hi_on()) continue; }   /* v1129U */
        const uint32_t rel = e->off - s_gfmix_org[e->blob];          /* function offset from its module header */
        if (pc < rel) continue;
        const uint32_t base = pc - rel;                                 /* runtime LS address of the module header */
        if ((base & 15u) || pc + e->len > SPU_LS_SIZE) continue;
        const uint8_t* img = s_gfmix_img[e->blob];
        if (memcmp(ctx->ls + base, img, 32) != 0) continue;            /* magic + module id + layout words */
        if (memcmp(ctx->ls + pc, img + rel, e->len) != 0) continue;     /* the function's own bytes */
        if (hit && (hit->blob != e->blob || hit_base != base)) { gfmix_stat(0); return 0; }   /* ambiguous: interpreter */
        hit = e; hit_base = base;
    }
    if (hit) {
        g_spu_frag_delta = hit_base - s_gfmix_org[hit->blob];          /* lifted at the capture address */
        if (fast2) {
            ce->pc = pc; ce->base = hit_base; ce->tab = (uint16_t)(hit - s_gfmix_tab); ce->kind = 1;
            memcpy(ce->code, ctx->ls + pc, sizeof ce->code);
        }
        gfmix_stat(1);
        return hit->fn;
    }
    if (fast2 && s_gfmix_tab[lo].h != h) {
        ce->pc = pc; ce->base = 0; ce->tab = 0; ce->kind = 2;
        memcpy(ce->code, ctx->ls + pc, sizeof ce->code);
    }
    if (fast2 && gfmix_fast2_diag()) {
        const long n = InterlockedIncrement(&s_fast2_diag_n);
        if (n <= 64) {
            uint32_t w[4]; memcpy(w, ctx->ls + pc, sizeof w);
            fprintf(stderr, "[gfmix-fast2-miss] img=%d ovl=%d pc=0x%05X hash-candidate=%d words=%08X %08X %08X %08X\n",
                    ctx->image_id, ctx->resident_ovl, pc, s_gfmix_tab[lo].h == h, w[0], w[1], w[2], w[3]);
        }
    }
    gfmix_stat(0);
    return 0;
}
spu_plug_fn spu_plug_lookup(spu_context* ctx, uint32_t pc)
{
    return spu_plug_lookup_probe(ctx, pc, 0);
}''')
    hdr += ['#ifdef __cplusplus', '}', '#endif', '']
    open(os.path.join(outdir, 'spu_recomp.h'), 'w', newline='\r\n').write('\n'.join(hdr))
    open(os.path.join(outdir, 'spu_recomp.c'), 'w', newline='\r\n').write('\n'.join(src) + '\n')
    for r in report: print('module %d %s 0x%X..0x%X: %d functions, unsupported %s' % r)
    print(len(entries), 'entries')

if __name__ == '__main__':
    main()
