/*
 * ps3recomp game project -- entry point template.
 *
 * This is the runner. It boots a lifted title through the toolkit's PPU
 * scaffold: allocate the flat guest VM, load the PPU ELF named on the command
 * line, register the lifted function table and the HLE NID handlers, start a
 * frame clock, and dispatch the entry OPD.
 *
 * The scaffold does the work; nothing below reimplements it. ppu_load_elf,
 * ppu_recomp_register, ppu_hle_init, ppu_sysprx_register, ppu_fs_register,
 * lv2_init_syscalls and ppu_run all come from the toolkit, compiled into this
 * project from PS3RECOMP_DIR (see CMakeLists.txt). What is left here is the
 * part a port owns: which backend to present through, how the frame clock is
 * paced, and whatever diagnostics the title turns out to need.
 *
 * runtime/ppu/tests/boot_main.cpp in the toolkit is the same boot in its
 * fully-instrumented form -- crash filter, hang watchdog, guest-PC sampling
 * profiler. Read it when a boot goes wrong; copy from it what the title
 * needs. It is deliberately not what a fresh project starts with.
 *
 * The one part that IS duplicated from it is the frame clock below, because a
 * port is expected to change its pacing and a shared one would be the wrong
 * shape for that. It was copied faithfully, comments and all. If a pacing bug
 * is fixed in one of the two, look at the other.
 *
 * This builds on Windows, macOS and Linux. CI builds and runs it against the
 * boot smoke title on the last two, and compile-checks it against clang-cl on
 * the first (tools/check_ppu_scaffold.py).
 */

/* The lifter's generated header, which declares ppu_context and the function
 * table. It comes first: everything below is written against that struct. */
#include "ppu_recomp.h"

/* The Win32 names -- Sleep, GetTickCount64, CreateThread, InterlockedIncrement,
 * VirtualAlloc, the scalar typedefs -- from one place. On Windows this is a
 * passthrough to <windows.h>; off Windows the toolkit's shim supplies the same
 * names over pthreads and mmap, so the call sites below are written once. */
#include "win32_compat.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
/* timeBeginPeriod: <windows.h> arrives with WIN32_LEAN_AND_MEAN set, which
 * excludes the multimedia timer API, so it is asked for by name -- and after
 * windows.h, since timeapi.h uses UINT. No shim exists off Windows because
 * there is nothing to shim: the 15.6 ms default timer granularity this works
 * around is a Windows problem. */
#include <timeapi.h>
#endif

/* ---------------------------------------------------------------------------
 * What the scaffold gives this file
 * -----------------------------------------------------------------------*/
extern "C" {
uint32_t ppu_load_elf(const char* path);      /* ELF -> guest RAM, returns entry OPD */
void     ppu_recomp_register(void);           /* generated: lifted table -> address map */
void     ppu_hle_init(void);                  /* firmware import NID -> HLE handler */
void     ppu_sysprx_register(void);           /* boot-critical CRT (sys_initialize_tls, ...) */
void     ppu_fs_register(void);               /* cellFs over the real game directory */
void     lv2_init_syscalls(void);             /* the lv2 syscall table */
int      ppu_run(uint32_t entry_opd, uint32_t stack_top);

/* This project's own: load whatever real system PRX modules the title needs
 * into guest RAM and register their exports. stubs.cpp has the empty version
 * a game with no lifted PRX wants. */
void     ps3_load_prx_modules(void);

extern const char* ppu_vfs_root;              /* host dir the PS3 mount points map into */

/* The status the guest handed to sys_process_exit. A title that exits that way
 * never returns through ppu_run, so this is only read on the path where the
 * entry function unwound instead. */
extern int     g_sys_process_exit_called;
extern int32_t g_sys_process_exit_code;

/* Host-provided symbols the runtime and the HLE libraries link against. */
uint8_t* vm_base = nullptr;
extern uint32_t ppu_vm_size;                  /* ppu_loader.cpp: the OOB guard */

/* Guest-callback dispatch. g_ps3_guest_caller is the hook the HLE runtime
 * calls back into recompiled code through -- cellSysutil events and the GCM
 * vblank/flip handlers. ppu_guest_call does the OPD -> dispatch. */
typedef void (*ps3_guest_caller_fn)(uint32_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                    uint64_t, uint64_t, uint64_t, uint64_t);
extern ps3_guest_caller_fn g_ps3_guest_caller;
uint64_t ppu_guest_call(uint32_t, uint64_t, uint64_t, uint64_t, uint64_t,
                        uint64_t, uint64_t, uint64_t, uint64_t);

/* The GCM half of the frame clock. */
void     cellGcmTickVBlank(void);
void     cellGcmTickFlip(void);
int      cellGcm_take_flip_pending(void);
void     cellGcm_rsx_process_fifo(void);      /* drain get -> put */
unsigned cellGcm_flip_request_count(void);
#ifdef _WIN32
/* v1031P: cellGcmSys.c -- wait up to ms for a FIFO kick (guest put store / dry ref poll). */
void     cellGcm_fifo_kick_wait(unsigned ms);
void*    cellGcm_fifo_kick_event(void);     /* v1129P: the auto-reset kick event the wait above uses */
#endif
}
/* v1038P PS3_MAIN_THREAD_SETUP: ppu_loader.cpp (C++ linkage). Main guest thread -> HIGHEST priority on the P-cores
 * ([PCORE]), PS3_MAIN_CORE physical-core reservation ([MAINCORE]), ps3_hz_set_main (hitch/[RFBOUND] attribution),
 * and the PS3_PROF_MAIN / PS3_SAMPLE / PS3_AB samplers when their env is set. Must run ON the main guest thread. */
void     ps3_sampler_start(void);

/* ---------------------------------------------------------------------------
 * RSX present backend
 * -----------------------------------------------------------------------*/
/* D3D12 on Windows, Metal on Apple, and the null backend's headless software
 * path anywhere else -- the same selection runtime/host/host_posix.c and the
 * toolkit's boot harness make. All three expose the same three entry points,
 * so the frame clock below is backend-agnostic. */
#if defined(_WIN32)
extern "C" int  rsx_d3d12_backend_init(uint32_t w, uint32_t h, const char* title);
extern "C" void rsx_d3d12_backend_present(void);
extern "C" int  rsx_d3d12_backend_pump_messages(void);
#  define rsx_backend_init    rsx_d3d12_backend_init
#  define rsx_backend_present rsx_d3d12_backend_present
#  define rsx_backend_pump    rsx_d3d12_backend_pump_messages
#  define RSX_BACKEND_NAME    "D3D12"
#elif defined(__APPLE__)
extern "C" int  rsx_metal_backend_init(uint32_t w, uint32_t h, const char* title);
extern "C" void rsx_metal_backend_present(void);
extern "C" int  rsx_metal_backend_pump_messages(void);
#  define rsx_backend_init    rsx_metal_backend_init
#  define rsx_backend_present rsx_metal_backend_present
#  define rsx_backend_pump    rsx_metal_backend_pump_messages
#  define RSX_BACKEND_NAME    "Metal"
#else
extern "C" int  rsx_null_backend_init(uint32_t w, uint32_t h, const char* title);
extern "C" void rsx_null_backend_present(void);
extern "C" int  rsx_null_backend_pump_messages(void);
#  define rsx_backend_init    rsx_null_backend_init
#  define rsx_backend_present rsx_null_backend_present
#  define rsx_backend_pump    rsx_null_backend_pump_messages
#  define RSX_BACKEND_NAME    "null (headless software)"
#endif

/* ---------------------------------------------------------------------------
 * Configuration
 * -----------------------------------------------------------------------*/
/* The flat VM must span every region the PS3 memory map uses, not just the
 * game image: sys_ppu_thread_create puts thread stacks at 0xD0000000, so a
 * smaller arena makes every spawned thread's stack access out of bounds. */
#define VM_SIZE       0x100010000ull   /* the full 32-bit guest space + a 64K guard */
#define STACK_TOP     0x0FF00000u      /* main-thread stack, below the 0x10000000 segment */
#define WINDOW_WIDTH  1280
#define WINDOW_HEIGHT 720

/* ---------------------------------------------------------------------------
 * Frame clock
 * -----------------------------------------------------------------------*/
/* On real hardware the RSX raises a vblank interrupt about 60 times a second
 * and that is what drives the game's frame loop. With no RSX it is synthesized
 * here: a host thread calls cellGcmTickVBlank/TickFlip, which invoke the
 * handlers the guest registered, and drains the GCM FIFO. Without this the
 * title initialises, registers its handlers and then waits forever.
 *
 * The ticks are driven off real elapsed time rather than off how long present()
 * took. A hidden or occluded window makes present block hard, and pacing the
 * ticks behind it paces the whole game behind it. */
static volatile LONG g_frames_presented = 0;

/* Frames handed to the backend at a guest FLIP boundary -- one per frame the
 * guest actually finished. The presents made before the guest's first flip, so
 * a fresh window is not left blank through a long boot, carry no guest frame
 * and are deliberately not counted. */
extern "C" unsigned ppu_boot_frames_presented(void)
{
    return (unsigned)g_frames_presented;
}

/* v1036P [PACE]: host-side cadence of the guest frames handed to the backend, timed
 * with QPC (the [flip] lines use GetTickCount64, whose 15.6 ms granularity turns a
 * steady 16.7 ms cadence into a 58/64 FPS zig-zag and hides single long frames).
 * One line per 10 s window: interval percentiles and the share of frames over
 * 20/25/34/50 ms. On when PACE_STATS=1 or GCM_FLIPCOUNT is set (verify-soak.ps1
 * sets the latter); PACE_STATS=0 forces it off. Diagnostics only. */
#ifdef _WIN32
static double pace_now_ms(void)
{
    static LARGE_INTEGER f = {0};
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}

static int pace_cmp(const void* a, const void* b)
{
    float x = *(const float*)a, y = *(const float*)b;
    return (x > y) - (x < y);
}

static void pace_note_frame(void)
{
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("PACE_STATS");
        on = e ? (*e && *e != '0') : (getenv("GCM_FLIPCOUNT") != NULL);
        if (on) fprintf(stderr, "[V1036P] [PACE] frame-interval stats ON (QPC, 10 s windows)%c", 10);
    }
    if (!on) return;
    enum { CAP = 4096 };
    static float  iv[CAP];
    static int    n = 0;
    static double last = 0.0, win_start = 0.0, t0 = 0.0;
    double now = pace_now_ms();
    if (last == 0.0) { last = win_start = t0 = now; return; }
    double d = now - last;
    last = now;
    if (n < CAP) iv[n++] = (float)d;
    if (now - win_start < 10000.0 || n < 2) return;
    int o20 = 0, o25 = 0, o34 = 0, o50 = 0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        float v = iv[i];
        sum += v;
        if (v > 20.0f) o20++;
        if (v > 25.0f) o25++;
        if (v > 34.0f) o34++;
        if (v > 50.0f) o50++;
    }
    qsort(iv, (size_t)n, sizeof iv[0], pace_cmp);
    float p50 = iv[n / 2];
    float p99 = iv[(int)((n - 1) * 0.99)];
    float p999 = iv[(int)((n - 1) * 0.999)];
    fprintf(stderr, "[PACE] v1036P t=%.0f ms frames=%d mean=%.2f p50=%.2f p99=%.2f p999=%.2f max=%.2f "
                    "over20=%d over25=%d over34=%d over50=%d%c",
            now - t0, n, sum / n, p50, p99, p999, iv[n - 1], o20, o25, o34, o50, 10);
    n = 0;
    win_start = now;
}
#else
static void pace_note_frame(void) {}
#endif

static void present_guest_frame(void)
{
    pace_note_frame();
    rsx_backend_present();
    /* This thread increments and guest threads read, so interlocked rather
     * than a volatile ++, which on arm64 is neither atomic nor a fence. */
    InterlockedIncrement(&g_frames_presented);
}

/* v1038P PS3_BUSY=1 (play.cmd; default off): where the FIFO/frame-clock thread's time goes, one line per 10 s:
 * idle kick wait | FIFO walk (cellGcm_rsx_process_fifo; includes the hand-off waits for the RF_ASYNC render thread,
 * see [RF_ASYNC] with PS3_TELEM=1) | present (present_guest_frame) | window pump | the rest. QPC, a few reads per loop. */
#ifdef _WIN32
enum { FB_WAIT, FB_WALK, FB_PRESENT, FB_PUMP, FB_N };
static int    s_fb_on = -1;
static double s_fb_acc[FB_N];
static unsigned s_fb_cnt[FB_N];
static double s_fb_win0 = 0.0;
static inline double fb_t(void) { return s_fb_on ? pace_now_ms() : 0.0; }
static inline void fb_add(int k, double t0) { if (s_fb_on) { s_fb_acc[k] += pace_now_ms() - t0; s_fb_cnt[k]++; } }
static void fb_report(void)
{
    if (!s_fb_on) return;
    const double now = pace_now_ms();
    if (s_fb_win0 == 0.0) { s_fb_win0 = now; return; }
    const double win = now - s_fb_win0;
    if (win < 10000.0) return;
    double busy = 0.0; for (int k = 0; k < FB_N; k++) busy += s_fb_acc[k];
    double other = win - busy; if (other < 0) other = 0;
    static unsigned fl0 = 0; const unsigned fl = cellGcm_flip_request_count();
    fprintf(stderr, "[V1038P] [BUSY-FIFO] %.1f s, %u guest flips: idle kick-wait %.0f ms (%.1f%%) | FIFO walk %.0f ms (%.1f%%, %u calls) | "
                    "present %.0f ms (%.1f%%, %u) | pump %.0f ms (%.1f%%) | other %.0f ms (%.1f%%) -> %s%c",
            win / 1000.0, fl - fl0,
            s_fb_acc[FB_WAIT], 100.0 * s_fb_acc[FB_WAIT] / win,
            s_fb_acc[FB_WALK], 100.0 * s_fb_acc[FB_WALK] / win, s_fb_cnt[FB_WALK],
            s_fb_acc[FB_PRESENT], 100.0 * s_fb_acc[FB_PRESENT] / win, s_fb_cnt[FB_PRESENT],
            s_fb_acc[FB_PUMP], 100.0 * s_fb_acc[FB_PUMP] / win, other, 100.0 * other / win,
            s_fb_acc[FB_WAIT] < 0.10 * win ? "FIFO thread nearly never idle: the RSX side (walk / render hand-off) is saturated"
                                           : "FIFO thread has idle time: it is not the frame limit", 10);
    fl0 = fl;
    for (int k = 0; k < FB_N; k++) { s_fb_acc[k] = 0.0; s_fb_cnt[k] = 0; }
    s_fb_win0 = now;
}
#else
enum { FB_WAIT, FB_WALK, FB_PRESENT, FB_PUMP, FB_N };
static inline double fb_t(void) { return 0.0; }
static inline void fb_add(int, double) {}
static void fb_report(void) {}
#endif

extern "C" int gf_resolve_target_fps(void);   /* v1042F, defined below */
/* v1129P GF_CLOCK_SUBMS=1 (default 0 = old): with FRAME_CLOCK_QPC the idle wait is min(4, whole ms to the next tick), so
 * for the last < 1 ms before every tick wait_ms is 0 and the loop spins wait(0) / walk / pump until the tick -- at a 190 Hz
 * tick (target 180) up to ~19 % of a core and ~1 M no-op FIFO walk calls per 10 s, which also inflates [BUSY-FIFO]
 * "FIFO walk". =1: that sub-ms remainder is slept on a high-resolution waitable timer due exactly at the tick, together
 * with the FIFO kick event (a guest put store / dry fence poll still wakes the loop at once). Ticks, walks, flips and
 * presents are the same; the loop only stops spinning in between. A kick coalesced while the previous walk ran can
 * wait at most that remainder (< 1 ms). [v1129P] clock line every 10 s when PS3_BUSY=1. */
#ifdef _WIN32
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
static int gf_clock_subms_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("GF_CLOCK_SUBMS");
        on = (e && *e == '1') ? 1 : 0;
        if (on) fprintf(stderr, "[v1129P] frame clock sub-ms idle wait ON (GF_CLOCK_SUBMS: high-res timer + kick event instead of a spin)%c", 10);
    }
    return on;
}
static int gf_clock_subms_wait(double rem_ms)
{
    static HANDLE timer = NULL; static int bad = 0;
    static unsigned long long n_wait = 0, n_kick = 0; static double t_rep = 0.0;
    if (bad) return 0;
    if (!timer) {
        timer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if (!timer) { bad = 1; fprintf(stderr, "[v1129P] GF_CLOCK_SUBMS: no high-resolution timer -> old spin%c", 10); return 0; }
    }
    HANDLE ev = (HANDLE)cellGcm_fifo_kick_event();
    if (!ev) return 0;
    LARGE_INTEGER due; due.QuadPart = -(LONGLONG)(rem_ms * 1e4);   /* ms -> 100 ns units, relative */
    if (due.QuadPart >= 0 || !SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE)) return 0;
    HANDLE hs[2] = { ev, timer };
    const DWORD r = WaitForMultipleObjects(2, hs, FALSE, 2);
    if (r == WAIT_OBJECT_0) { n_kick++; CancelWaitableTimer(timer); }
    n_wait++;
    if (s_fb_on > 0) {
        const double now = pace_now_ms();
        if (t_rep == 0.0) t_rep = now;
        else if (now - t_rep >= 10000.0) {
            fprintf(stderr, "[v1129P] frame clock 10 s: %llu sub-ms timer waits (%llu woken by a kick)%c", n_wait, n_kick, 10);
            t_rep = now; n_wait = n_kick = 0;
        }
    }
    return 1;
}
#endif
extern "C" volatile int g_gf_ffwd;              /* v1042S cellPad.c */
extern "C" volatile int g_gf_turbo;             /* v1043T-4 cellPad.c: name-entry typing steps pending (no present, 500 Hz) */
extern "C" void gf_cutscene_init(void);         /* v1042S gf_cutscene.c */
extern "C" void gf_ragdoll_init(void);          /* v1054R gf_ragdoll.c: NPC death / bone pose logging */
static DWORD WINAPI frame_clock(LPVOID)
{
    const char* title = getenv("PS3_TITLE");
    if (!title || !*title) title = "ps3recomp";

    int rsx_ok = (rsx_backend_init(WINDOW_WIDTH, WINDOW_HEIGHT, title) == 0);
    fprintf(stderr, "[rsx] %s backend init %s\n", RSX_BACKEND_NAME,
            rsx_ok ? "OK -- window open" : "FAILED");

    unsigned  last_flip = 0;
    ULONGLONG last_free_present = 0;   /* v1031P: pre-first-flip presents, throttled */

    /* v1036P FRAME_CLOCK_QPC (play.cmd = 1; code default 0 = the old clock). The vblank/flip
     * ticks ran off GetTickCount64 (15.6 ms granularity) with a fixed 16 ms step: at most one
     * tick per 15.6 ms granule and the schedule slipping 0.375 ms per granule, so every ~42
     * granules (~0.66 s) one granule fired no tick -- a 31.25 ms gap. A flip only turns DONE
     * at a tick, and Godfather polls GetFlipStatus for its previous flip before each SetFlip
     * (0x72E420), so a 60 FPS frame that straddled a gap waited up to ~14.5 ms for it. That is
     * about one long frame every 0.66 s and the ~59.4 instead of 59.94 FPS average. With QPC
     * the ticks follow an exact period and the idle wait is cut short so the next tick is not
     * up to 4 ms late. The default rate is 62.5 Hz (16.0 ms), NOT 59.94: every window between
     * a SetFlip and the next frame's poll (one 16.683 ms limiter period minus a little) must
     * contain a tick. A 59.94 Hz tick would phase-lock to Godfather's 59.94 Hz limiter (both
     * run off the host QPC) and could sit on the window edge for good. Godfather registers no
     * vblank handler and takes its game time from the timebase, so the tick rate itself is
     * not visible to the simulation. FRAME_CLOCK_HZ=<hz> overrides. */
    double tick_period = 16.0;
    int    qpc_clock = 0;
#ifdef _WIN32
    {
        const char* e = getenv("FRAME_CLOCK_QPC");
        qpc_clock = (e && *e && *e != '0') ? 1 : 0;
        if (qpc_clock) {
            const char* hz = getenv("FRAME_CLOCK_HZ");
            /* v1042F: unset/empty -> follow the frame-rate target. Every limiter period must contain a tick (a
             * flip only completes at a tick), so tick a little faster than the target: 62.5 Hz at 60, ~1.05x above. */
            const int tgt = gf_resolve_target_fps();
            double h = (hz && *hz) ? atof(hz) : (tgt > 60 ? tgt * 1.05 + 1.0 : 62.5);
            if (h < 20.0 || h > 500.0) h = 62.5;
            tick_period = 1000.0 / h;
        }
        fprintf(stderr, "[V1036P] frame clock: %s, tick period %.4f ms (FRAME_CLOCK_QPC / FRAME_CLOCK_HZ)%c",
                qpc_clock ? "QPC" : "GetTickCount64 (old)", tick_period, 10);
    }
#endif
#ifdef _WIN32
    double next_tick = qpc_clock ? pace_now_ms() : (double)GetTickCount64();
#else
    double next_tick = (double)GetTickCount64();
#endif

    /* v1031P FIFO_KICK_WAIT (default 1; 0 = the old flat Sleep(4)). The idle wait used to be a
     * flat Sleep(4): the kick event that guest put stores and dry ref polls signal
     * (cellGcm_put_kick / cellGcm_ref_on_poll) was waited on by nobody, so every fence the
     * guest spun on cost up to a full 4-5 ms sleep before the walker ran. Godfather's main
     * thread spent 20-44% of its time in that ref poll ([TELEM] tid 1 sc141 lr 0045F278,
     * [REFPOLL] 100% dry). cellGcm_fifo_kick_wait keeps the same 4 ms timeout when nobody
     * kicks and drains at once when someone does (GCM_FIFO_KICK=0 there = Sleep). */
#ifdef _WIN32
    static int kick_wait = -1;
    if (kick_wait < 0) {
        const char* e = getenv("FIFO_KICK_WAIT");
        kick_wait = (e && *e == '0') ? 0 : 1;
        fprintf(stderr, "[rsx] v1031P frame clock idle wait: %s (FIFO_KICK_WAIT)\n",
                kick_wait ? "kick event, 4 ms timeout" : "Sleep(4)");
    }
#endif

#ifdef _WIN32
    if (s_fb_on < 0) {
        const char* e = getenv("PS3_BUSY");
        s_fb_on = (e && *e == '1') ? 1 : 0;
        if (s_fb_on) fprintf(stderr, "[V1038P] [BUSY-FIFO] FIFO/frame-clock thread time breakdown ON (PS3_BUSY)%c", 10);
    }
#endif
    for (;;) {
        fb_report();
#ifdef _WIN32
        unsigned wait_ms = 4;
        double rem_ms = -1.0;   /* v1129P GF_CLOCK_SUBMS */
        if (qpc_clock) {
            /* v1036P: wake for the next tick, not up to 4 ms after it. */
            double rem = next_tick - pace_now_ms();
            rem_ms = rem;
            wait_ms = rem <= 0.0 ? 0u : (rem >= 4.0 ? 4u : (unsigned)rem);
        }
        { const double fb0 = fb_t();
        if (kick_wait && wait_ms == 0 && rem_ms > 0.02 && gf_clock_subms_on() && gf_clock_subms_wait(rem_ms)) { /* v1129P */ }
        else if (kick_wait) cellGcm_fifo_kick_wait(wait_ms);
        else           Sleep(wait_ms);
        fb_add(FB_WAIT, fb0); }
#else
        Sleep(4);
#endif
        ULONGLONG now = GetTickCount64();
#ifdef _WIN32
        double tnow = qpc_clock ? pace_now_ms() : (double)now;
#else
        double tnow = (double)now;
#endif

        int fired = 0;
        while (tnow >= next_tick && fired < 240) {
            cellGcmTickVBlank();
            cellGcmTickFlip();
            /* Present a pending flip BEFORE draining any further. The flip
             * fires at a get==put frame boundary on the guest thread, so the
             * batch held right now is exactly the completed frame; presenting
             * after the drain races the guest's next-frame writes and shows a
             * mixed one. */
            if (rsx_ok && cellGcm_take_flip_pending()) {
                const double fb0 = fb_t();
                /* A cutscene skip drains its remaining scripted frames, but they must never be
                 * presented: showing them is merely fast-forward. Keep consuming flips/FIFO so
                 * guest cleanup can finish, then the first post-cutscene flip is shown normally. */
                if (!g_gf_ffwd && !g_gf_turbo) {
                    present_guest_frame();
                    fb_add(FB_PRESENT, fb0);
                }
                last_flip = cellGcm_flip_request_count();
            }
            /* Drain the FIFO every tick. This is what writes the RSX sync-fence
             * labels the game's per-frame logic blocks on, so it has to keep
             * advancing at 60 Hz even while present() throttles. */
            if (rsx_ok) { const double fb0 = fb_t(); cellGcm_rsx_process_fifo(); fb_add(FB_WALK, fb0); }
            next_tick += (g_gf_ffwd || g_gf_turbo) ? 2.0 : tick_period;    /* ~60 Hz (v1036P: exact 62.5 Hz with FRAME_CLOCK_QPC); v1042S 500 Hz while a cutscene fast-forwards */
            fired++;
        }
        if (fired >= 240) next_tick = tnow;  /* fell too far behind -- resync */

        /* Drain at the outer cadence too, not only on the 16 ms tick: titles
         * fence every render pass on a label the drain writes, and one wave of
         * those at 16 ms apiece paces the guest into single-figure frame rates.
         * The real RSX writes them in microseconds. */
        if (rsx_ok) {
            if (cellGcm_take_flip_pending()) {
                const double fb0 = fb_t();
                if (!g_gf_ffwd && !g_gf_turbo) {
                    present_guest_frame();
                    fb_add(FB_PRESENT, fb0);
                }
                last_flip = cellGcm_flip_request_count();
            }
            { const double fb0 = fb_t(); cellGcm_rsx_process_fifo(); fb_add(FB_WALK, fb0); }

            const double fb_p0 = fb_t();
            const int pumped = rsx_backend_pump();
            fb_add(FB_PUMP, fb_p0);
            if (pumped != 0) {
                rsx_ok = 0;              /* window closed */
#ifdef _WIN32
                /* v1045F: backstop for the WM_CLOSE exit in rsx_d3d12_backend.c (e.g. WM_QUIT from elsewhere) --
                 * without a window there is nothing to play, so do not keep running headless. */
                { const char* keep = getenv("PS3_KEEP_RUNNING_ON_CLOSE");
                  if (!(keep && *keep == '1')) {
                      printf("[boot] window gone -> exiting\n"); fflush(stdout); fflush(stderr);
                      TerminateProcess(GetCurrentProcess(), 0);
                  } }
#endif
                continue;
            }
            /* Present on a guest flip. A present on a fixed clock can catch the
             * drain mid-frame and flash a partial one. Before the first flip
             * present freely, so the window is not blank during boot. */
            unsigned fc = cellGcm_flip_request_count();
            if (fc != last_flip) {
                const double fb0 = fb_t();
                if (!g_gf_ffwd && !g_gf_turbo) {
                    present_guest_frame();
                    fb_add(FB_PRESENT, fb0);
                }
                last_flip = fc;
            } else if (fc == 0 && now - last_free_present >= 16) {
                /* v1031P: the loop now wakes on every FIFO kick; keep the
                 * boot-time free present at the old ~60 Hz at most. */
                last_free_present = now;
                rsx_backend_present();
            }
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * VFS root
 * -----------------------------------------------------------------------*/
/* Derive the directory holding PS3_GAME from the EBOOT path
 * <root>/PS3_GAME/USRDIR/EBOOT.elf -> <root>. $PS3_VFS_ROOT overrides. */
static char s_vfs_root[1024];

static void derive_vfs_root(const char* eboot)
{
    const char* env = getenv("PS3_VFS_ROOT");
    if (env && *env) { ppu_vfs_root = env; return; }

    strncpy(s_vfs_root, eboot, sizeof s_vfs_root - 1);
    for (char* p = s_vfs_root; *p; p++) if (*p == '\\') *p = '/';
    /* strip EBOOT.elf, USRDIR and PS3_GAME */
    for (int i = 0; i < 3; i++) { char* s = strrchr(s_vfs_root, '/'); if (s) *s = 0; }
    if (!s_vfs_root[0]) strcpy(s_vfs_root, ".");
    ppu_vfs_root = s_vfs_root;
}

/* ---------------------------------------------------------------------------
 * Guest VM
 * -----------------------------------------------------------------------*/
#ifdef _WIN32
/* Demand-paging for the flat VM: reserve the whole 4 GB guest space, which
 * costs no commit, and commit each 64 KB page on first access. Every 32-bit
 * guest offset is then valid, so a garbage guest pointer reads zero instead of
 * killing the process. Faults outside the arena fall through untouched. */
static LONG WINAPI vm_commit_veh(EXCEPTION_POINTERS* ep)
{
    if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
        ULONG_PTR fault = ep->ExceptionRecord->ExceptionInformation[1];
        uintptr_t base  = (uintptr_t)vm_base;
        if (vm_base && fault >= base && fault < base + VM_SIZE) {
            void* page = (void*)(fault & ~(uintptr_t)0xFFFF);
            if (VirtualAlloc(page, 0x10000, MEM_COMMIT, PAGE_READWRITE))
                return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

static bool alloc_guest_vm(void)
{
#ifdef _WIN32
    AddVectoredExceptionHandler(1, vm_commit_veh);
    vm_base = (uint8_t*)VirtualAlloc(NULL, VM_SIZE, MEM_RESERVE, PAGE_READWRITE);
    ppu_vm_size = 0;              /* the whole space is backed; no OOB guard needed */
#else
    /* No vectored exception handlers off Windows, so the arena is committed up
     * front and lazily backed by the OS: only pages the title touches cost
     * anything. It stops below the 0xE0000000 mark, so ppu_vm_size arms the
     * loader's bounds check for what is left. */
    vm_base = (uint8_t*)calloc(1, 0xE0000000u);
    ppu_vm_size = 0xE0000000u;
#endif
    return vm_base != nullptr;
}

/* ---------------------------------------------------------------------------
 * Main
 * -----------------------------------------------------------------------*/
static void harness_guest_caller(uint32_t opd, uint64_t a0, uint64_t a1,
                                 uint64_t a2, uint64_t a3, uint64_t a4,
                                 uint64_t a5, uint64_t a6, uint64_t a7)
{
    ppu_guest_call(opd, a0, a1, a2, a3, a4, a5, a6, a7);
}

extern "C" void gf_prompts_start(void);
/* v1042F frame-rate target. GODFATHER_TARGET_FPS = a number (30..240) or "auto" (the monitor's refresh rate);
 * unset: GODFATHER_60FPS=1 -> 60, else the retail 30. Above 60 the lifted limiter/clock run in measured
 * high-refresh mode (ppu_recomp_002.cpp 0x45F034 / 0x4E637C). */
extern "C" int gf_resolve_target_fps(void)
{
    static int v = 0;
    if (v) return v;
    const char* tf = getenv("GODFATHER_TARGET_FPS");
    const char* e60 = getenv("GODFATHER_60FPS");
    int fps = (e60 && *e60 && *e60 != '0') ? 60 : 30;
    if (tf && *tf) {
        if (!_stricmp(tf, "auto") || !_stricmp(tf, "monitor")) {
#ifdef _WIN32
            DEVMODEA dm; memset(&dm, 0, sizeof dm); dm.dmSize = sizeof dm;
            fps = (EnumDisplaySettingsA(NULL, ENUM_CURRENT_SETTINGS, &dm) && dm.dmDisplayFrequency > 1)
                  ? (int)dm.dmDisplayFrequency : 60;
#else
            fps = 60;
#endif
        } else fps = (int)strtol(tf, 0, 10);
    }
    if (fps < 30) fps = 30;
    if (fps > 240) fps = 240;
    v = fps;
    fprintf(stderr, "[V1042F] frame-rate target %d FPS (GODFATHER_TARGET_FPS=%s)%c", fps, tf ? tf : "unset", 10);
    return v;
}
extern "C" double gf_now_sec(void)
{
#ifdef _WIN32
    static LARGE_INTEGER f; LARGE_INTEGER t;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
#else
    return (double)GetTickCount64() * 0.001;
#endif
}
/* v1121H: retain the absolute deadline but block closer to it before the final precision spin. */
static int gf_wait_lean_enabled(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("GF_WAIT_LEAN"); on = (e && *e == '1') ? 1 : 0; }
    return on;
}
/* v1046P GF_PACER (default 1; 0 = the old scaled guest wait). Above 60 FPS the retail limiter (0x45ED60) waits
 * "budget since its own start stamp" in usleep(500) steps + a spin. That stamp is taken after the previous wait, so
 * every frame also carries the untimed work between the two (0.2-0.5 ms) and the usleep overshoot: soaks Y03/Y04
 * ran 172-175 FPS at a 180 target and 128-139 at 144, with ~3 ms (180) / ~4.4 ms (144) limiter sleep left over.
 * The pacer replaces only the frame limiter's wait: an ABSOLUTE schedule (deadline += period, so error never
 * accumulates), a high-resolution waitable timer up to GF_PACER_SPIN_US (default 400) before the deadline, then a
 * spin. A frame that runs late resyncs instead of bursting to catch up (no uneven fast frames after a hitch).
 * Returns 1 when it waited (the caller then gives the guest loop a zero budget, the same exit as fast-forward). */
#ifdef _WIN32
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#endif
/* v1046J hang reporter (GF_HANG_REPORT, default 1): the user's game froze twice (2026-11-02 18:5x / 19:3x) with the
 * main guest thread stuck in guest code -- no syscall, so the [FREEZE] watchdog never names it. gf_pace_frame (called
 * once per game frame) stamps the time; a watchdog thread notices > 3 s without a frame and samples the main thread's
 * host RIP 32 times. "[HANG] ... rva 0x.... x N" lines in play-fast.err.log; resolve the RVAs with the PDB
 * (llvm-symbolizer --obj=GodfatherRecomp.exe 0x<rva + 0x140000000>). Once per stall. */
#ifdef _WIN32
static volatile double s_hang_last = 0.0;
static HANDLE s_hang_main = NULL;
static DWORD WINAPI hang_watch(LPVOID)
{
    int reported = 0;
    const uintptr_t base = (uintptr_t)GetModuleHandleA(NULL);
    for (;;) {
        Sleep(500);
        const double last = s_hang_last;
        if (last <= 0.0) continue;
        const double stuck = gf_now_sec() - last;
        if (stuck < 3.0) { reported = 0; continue; }
        if (reported) continue;
        reported = 1;
        uintptr_t rva[32]; int n = 0;
        for (int i = 0; i < 32; i++) {
            if (SuspendThread(s_hang_main) != (DWORD)-1) {
                CONTEXT c; memset(&c, 0, sizeof c); c.ContextFlags = CONTEXT_CONTROL;
                if (GetThreadContext(s_hang_main, &c)) rva[n++] = (uintptr_t)c.Rip - base;
                ResumeThread(s_hang_main);
            }
            Sleep(30);
        }
        fprintf(stderr, "[HANG] v1046J main guest thread has not finished a frame for %.1f s; RIP samples (rva, image base 0x%llX):",
                stuck, (unsigned long long)base);
        for (int i = 0; i < n; i++) fprintf(stderr, " %llX", (unsigned long long)rva[i]);
        fprintf(stderr, "%c", 10); fflush(stderr);
    }
    return 0;
}
#endif
/* v1050S (user: "make the camera feel WAY better, I just don't want choppiness"): play-fast at a 180 target ran 90-115 FPS
 * in the city with frame times spread from 5 to 16 ms (presentlog 02:04, mean 10 ms) -> half the frames late and an
 * uneven mix of short / long frames, which the eye reads as camera judder. Shared with the game clock hook
 * (ppu_recomp_002.cpp func_004E60F8, GF_DT_SMOOTH) and gf_footcam.c:
 *   g_gf_pace_per  = the pacer's current frame period (s), 0 = pacer off;
 *   g_gf_clock_dt  = the game timestep the clock hook published last (s), g_gf_clock_seq counts those ticks. */
extern "C" { volatile double g_gf_pace_per = 0.0; volatile double g_gf_clock_dt = 0.0; volatile unsigned g_gf_clock_seq = 0; }
#include <math.h>
extern "C" {   /* v1051P cellGcmSys.c (PRESENT_SCHED shared state) */
    extern volatile double g_ps_vbl_anchor, g_ps_vbl_period, g_ps_vbl_stamp, g_ps_D, g_gf_frame_V, g_gf_sched_dt;
    extern volatile int g_ps_on;
}
/* v1129P GF_PACER_STATS=1 (default 0 = no line, old behaviour): the pacer's wait runs inside the lifted limiter hook, so
 * [BUSY] counts it as "guest code" and calls the main thread busy even when most of its frame is this wait. One line per
 * 10 s: frames, how many waited, wait per frame and as a share of wall time, the frame WORK (pacer return -> next call)
 * mean / p50 / p85 / p95 / p99 and the mean cap period. Wait share high + work well under the period = the cap, not a
 * thread, holds the frame rate. Diagnostics only: no timing changes. */
static int gf_pacer_stats_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("GF_PACER_STATS");
        on = (e && *e == '1') ? 1 : 0;
        if (on) fprintf(stderr, "[v1129P] pacer wait/work stats ON (GF_PACER_STATS, 10 s lines)%c", 10);
    }
    return on;
}
static int gf_pacer_stats_cmp(const void* a, const void* b)
{
    const float x = *(const float*)a, y = *(const float*)b;
    return (x > y) - (x < y);
}
static void gf_pacer_stats_note(double work, double waited, double per)
{
    static float wk[4096];
    static unsigned n = 0, nk = 0, nw = 0;
    static double t0 = 0.0, wsum = 0.0, ksum = 0.0, psum = 0.0;
    const double t = gf_now_sec();
    if (t0 == 0.0) t0 = t;
    n++; psum += per;
    if (waited > 0.00005) { nw++; wsum += waited; }
    if (work >= 0.0 && work < 1.0) { ksum += work; if (nk < 4096) wk[nk] = (float)work; nk++; }
    const double span = t - t0;
    if (span < 10.0) return;
    const unsigned m = nk < 4096 ? nk : 4096;
    double p50 = 0.0, p85 = 0.0, p95 = 0.0, p99 = 0.0;
    if (m) {
        qsort(wk, m, sizeof wk[0], gf_pacer_stats_cmp);
        p50 = wk[(m * 50) / 100]; p85 = wk[(m * 85) / 100 < m ? (m * 85) / 100 : m - 1];
        p95 = wk[(m * 95) / 100 < m ? (m * 95) / 100 : m - 1]; p99 = wk[(m * 99) / 100 < m ? (m * 99) / 100 : m - 1];
    }
    const double kmean = nk ? ksum / nk : 0.0, pmean = n ? psum / n : 0.0, wshare = wsum / span;
    fprintf(stderr, "[v1129P] pacer 10 s: %u frames (%.1f FPS), %u waited (%.0f %%), wait %.2f ms/frame = %.0f %% of wall | work ms mean %.2f p50 %.2f p85 %.2f p95 %.2f p99 %.2f | cap period mean %.2f ms (%.0f FPS) -> %s%c",
            n, n / span, nw, n ? 100.0 * nw / n : 0.0, n ? wsum * 1000.0 / n : 0.0, 100.0 * wshare,
            kmean * 1000.0, p50 * 1000.0, p85 * 1000.0, p95 * 1000.0, p99 * 1000.0, pmean * 1000.0, pmean > 0.0 ? 1.0 / pmean : 0.0,
            wshare > 0.15 ? "PACER-BOUND (the cap holds the rate)" : "pacer not limiting", 10);
    t0 = t; n = nk = nw = 0; wsum = ksum = psum = 0.0;
}
extern "C" int gf_pace_frame(void)
{
    static int on = -1;
    static double period = 0.0, spin = 0.0004, next = 0.0;
    static unsigned long long n_frames = 0, n_late = 0, n_resync = 0;
    /* v1046R GF_PACER_ADAPT (v1046S: default 0 -- user: "rather unstable but higher"; =1 opt-in): with vsync a frame rate between refresh steps (e.g. 75-90 on a 180 Hz screen)
     * is shown as an uneven mix of 2- and 3-refresh frames -> the player visibly jumps every few steps (user report).
     * When the frame work (time outside this wait) cannot keep the full rate, lock to an even divisor of it
     * (180 -> 90 -> 60), and go back up once there is room again. Work = EMA over ~20 frames; down after 0.5 s over,
     * up after 2 s with >= 25 % headroom. GF_PACER_ADAPT=1 turns it on. */
    static int adapt = -1, div = 1; static double ret_t = 0.0, ema = 0.0, over_t = 0.0, under_t = 0.0;
    static double rep_t = 0.0, late_sum = 0.0;
#ifdef _WIN32
    static HANDLE timer = NULL;
#endif
    if (on < 0) {
        const char* e = getenv("GF_PACER");
        on = (e && *e == '0') ? 0 : 1;
        const int tgt = gf_resolve_target_fps();
        period = 1.0 / (double)(tgt > 0 ? tgt : 60);
        const char* sp = getenv("GF_PACER_SPIN_US");
        if (sp && *sp) { double v = atof(sp); if (v >= 0.0 && v <= 4000.0) spin = v * 1e-6; }
#ifdef _WIN32
        if (on) {
            timer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
            if (!timer) timer = CreateWaitableTimerExW(NULL, NULL, 0, TIMER_ALL_ACCESS);
        }
#endif
        fprintf(stderr, "[V1046P] frame pacer %s: %.3f ms period (%d FPS), spin %.0f us (GF_PACER / GF_PACER_SPIN_US)%c",
                on ? "ON" : "OFF", period * 1000.0, tgt, spin * 1e6, 10);
    }
#ifdef _WIN32
    { static int hw = -1;
      if (hw < 0) { const char* e = getenv("GF_HANG_REPORT"); hw = (e && *e == '0') ? 0 : 1;
          if (hw && DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &s_hang_main,
                                    THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0)) {
              HANDLE h = CreateThread(NULL, 64u << 10, hang_watch, NULL, 0, NULL); if (h) CloseHandle(h);
              fprintf(stderr, "[HANG] v1046J main-thread hang reporter ON (GF_HANG_REPORT)%c", 10);
          } }
      s_hang_last = gf_now_sec(); }
#endif
    if (!on) return 0;
    double now = gf_now_sec();
    const double ps_t_in = now, ps_work = ret_t > 0.0 ? now - ret_t : -1.0;   /* v1129P GF_PACER_STATS */
    /* v1051P PRESENT_SCHED: frame starts mapped onto the display's vblank grid (see rsx_d3d12_backend.c). The frame
     * starts as soon as the previous one is done (no fixed cap), except that two frames never share a vblank: the
     * earliest start is previous V + n refreshes - D. Its vblank V = first grid vblank >= S + D; the game timestep is
     * V - previous V, so what is shown at each vblank is exactly the game state for that moment. */
    { static double Vprev = 0.0, tl = 0.0, dsum = 0.0; static unsigned long long nf = 0, h1 = 0, h2 = 0, h3 = 0;
      const double R = g_ps_vbl_period, A = g_ps_vbl_anchor;
      if (g_ps_on == 1 && R > 0.0025 && R < 0.034 && now - g_ps_vbl_stamp < 0.5) {
          int nmin = (int)(period / R + 0.5); if (nmin < 1) nmin = 1;
          double D = g_ps_D; if (D < 0.5 * R) D = 0.5 * R;
          if (Vprev > 0.0 && now - Vprev > 1.0) Vprev = 0.0;          /* long pause (load): start over */
          if (Vprev > 0.0) {
              const double smin = Vprev + nmin * R - D - 0.0002;
              if (now < smin) {
                  const double rem = smin - now;
#ifdef _WIN32
                  const double final_spin = gf_wait_lean_enabled() && spin > 0.00015 ? 0.00015 : spin;
                  if (timer && rem > final_spin) {
                      LARGE_INTEGER due; due.QuadPart = -(LONGLONG)((rem - final_spin) * 1e7);
                      if (due.QuadPart < 0 && SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE)) WaitForSingleObject(timer, 100);
                  }
                  while (gf_now_sec() < smin) YieldProcessor();
#else
                  while (gf_now_sec() < smin) { }
#endif
              }
          }
          const double S = gf_now_sec();
          double V = A + ceil((S + D - A) / R - 0.02) * R;
          if (Vprev > 0.0 && V < Vprev + (nmin - 0.3) * R) V = A + floor((Vprev + nmin * R - A) / R + 0.5) * R;
          const double dt = Vprev > 0.0 ? V - Vprev : nmin * R;
          Vprev = V;
          g_gf_frame_V = V; g_gf_sched_dt = dt; g_gf_pace_per = nmin * R;
          nf++; dsum += dt; { const double k = dt / R; if (k < 1.5) h1++; else if (k < 2.5) h2++; else h3++; }
          if (S - tl >= 10.0) {
              if (tl > 0.0 && nf)
                  fprintf(stderr, "[V1051P] pacer 10 s: %llu frames (%.1f FPS), shown for 1 refresh %llu, 2 %llu, 3+ %llu, D %.2f ms%c",
                          nf, nf / (S - tl), h1, h2, h3, D * 1000.0, 10);
              tl = S; nf = h1 = h2 = h3 = 0; dsum = 0.0;
          }
          next = 0.0;          /* the normal schedule restarts cleanly if the grid goes away */
          ret_t = gf_now_sec();
          return 1;
      }
      if (Vprev != 0.0) { Vprev = 0.0; g_gf_frame_V = 0.0; g_gf_sched_dt = 0.0; }
    }
    if (adapt < 0) { const char* e = getenv("GF_PACER_ADAPT"); adapt = (e && (*e == '1' || *e == '2' || *e == '3')) ? (*e - '0') : 0;   /* v1050T: 3 = dynamic cap */   /* v1046S: opt-in; v1050S: 2 = percentile lock */
                     fprintf(stderr, "[V1046R] pacer even-divisor lock %s (GF_PACER_ADAPT)%c",
                             adapt == 3 ? "OFF -- v1050T dynamic cap instead" : adapt == 2 ? "ON, v1050S percentile mode" : adapt ? "ON" : "OFF", 10); }
    if (adapt == 1 && ret_t > 0.0) {
        double work = now - ret_t; if (work > 0.25) work = 0.25;
        ema = ema <= 0.0 ? work : ema * 0.95 + work * 0.05;
        const double cur = period * div;
        if (ema > cur * 0.93 && div < 3) { if (over_t == 0.0) over_t = now; } else over_t = 0.0;
        if (div > 1 && ema < period * (div - 1) * 0.75) { if (under_t == 0.0) under_t = now; } else under_t = 0.0;
        int nd = div;
        if (over_t != 0.0 && now - over_t >= 0.5) nd = div + 1;
        else if (under_t != 0.0 && now - under_t >= 2.0) nd = div - 1;
        if (nd != div) {
            div = nd; over_t = under_t = 0.0; next = 0.0;
            fprintf(stderr, "[V1046R] pacer locks to %.0f FPS (frame work %.2f ms)%c", 1.0 / (period * div), ema * 1000.0, 10);
        }
    }
    /* v1050S GF_PACER_ADAPT=2: the v1046R lock decided from the AVERAGE frame work, so a load where a quarter of the
     * frames ran long (mean below the period, p90 well above it) never locked -> it stayed uneven. Mode 2 keeps the
     * work of the last 128 frames and every 12 frames looks at the slow ones: p90 > 90 % of the current period twice
     * in a row -> one step down (180 -> 90 -> 60, never below 60 or GF_PACER_MINFPS); p98 < 80 % of the faster
     * period for hold_up seconds -> one step up. hold_up starts at 3 s and doubles (to 40 s) whenever a step up is
     * undone within 6 s, so the rate does not flap; it resets after 30 s without a change. */
    if (adapt == 2 && ret_t > 0.0) {
        static double ring[128], hold_up = 3.0, up_ok_t = 0.0, last_change = 0.0;
        static int rn = 0, rpos = 0, since_eval = 0, over_n = 0, divmax = 0, last_up = 0;
        if (!divmax) {
            const int tgt = gf_resolve_target_fps();
            divmax = tgt / 60; if (divmax < 1) divmax = 1; if (divmax > 4) divmax = 4;
            const char* e = getenv("GF_PACER_MINFPS");
            if (e && *e) { const int m = atoi(e); if (m >= 20) { const int d = tgt / m; if (d >= 1 && d <= 8) divmax = d; } }
            fprintf(stderr, "[V1050S] pacer percentile lock: steps %d", tgt);
            for (int d = 2; d <= divmax; d++) fprintf(stderr, " / %d", tgt / d);
            fprintf(stderr, " FPS%c", 10);
        }
        double work = now - ret_t; if (work > 0.25) work = 0.25;
        ring[rpos] = work; rpos = (rpos + 1) & 127; if (rn < 128) rn++;
        if (++since_eval >= 12 && rn >= 24) {
            since_eval = 0;
            double t[128]; const int n = rn;
            for (int i = 0; i < n; i++) {             /* insertion sort, <= 128 values every 12 frames */
                const double v = ring[i]; int j = i;
                while (j > 0 && t[j - 1] > v) { t[j] = t[j - 1]; j--; }
                t[j] = v;
            }
            const double p90 = t[(n * 90) / 100 < n ? (n * 90) / 100 : n - 1], p98 = t[(n * 98) / 100 < n ? (n * 98) / 100 : n - 1];
            const double cur = period * div;
            int nd = div;
            if (now - last_change > 30.0) hold_up = 3.0;
            if (div < divmax && p90 > cur * 0.90) { if (++over_n >= 2) nd = div + 1; } else over_n = 0;
            if (nd == div && div > 1 && p98 < period * (div - 1) * 0.80) {
                if (up_ok_t == 0.0) up_ok_t = now;
                if (now - up_ok_t >= hold_up) nd = div - 1;
            } else up_ok_t = 0.0;
            if (nd != div) {
                if (nd > div && last_up && now - last_change < 6.0) { hold_up *= 2.0; if (hold_up > 40.0) hold_up = 40.0; }
                last_up = nd < div; last_change = now;
                div = nd; rn = rpos = 0; over_n = 0; up_ok_t = 0.0; next = 0.0;
                fprintf(stderr, "[V1050S] pacer locks to %.0f FPS (frame work p90 %.2f ms, p98 %.2f ms; next step up after %.0f s of headroom)%c",
                        1.0 / (period * div), p90 * 1000.0, p98 * 1000.0, hold_up, 10);
            }
        }
    }
    /* v1050T GF_PACER_ADAPT=3 (user: the 180/90/60 lock "is just locked to 60, I do NOT want that, but I also don't want
     * the choppy camera"): no fixed steps. The frame period follows what the PC can hold CONSISTENTLY right now:
     * GF_PACER_DYN_PCT (85) percentile of the last 128 frames' work x (1 + GF_PACER_DYN_HEADROOM 5 %), between the
     * target (180) and GF_PACER_DYN_MINFPS (72). Slower within ~16 frames when frames run long, faster gently (~1 s),
     * so the rate glides instead of jumping; single hitches (loads) are capped at 3 periods and are below the
     * percentile. Every frame then starts on an even schedule (no 5 / 16 ms mix) and GF_DT_SMOOTH steps the game by
     * exactly that period -> even camera motion at the highest steady rate. */
    static double dynper = 0.0;
    if (adapt == 3 && ret_t > 0.0) {
        static double ring[128], pct = -1.0, head = 1.05, pmax = 1.0 / 72.0, lastlog = 0.0, logged = 0.0;
        static int rn = 0, rpos = 0, since = 0;
        if (pct < 0.0) {
            const char* e = getenv("GF_PACER_DYN_PCT"); pct = (e && *e) ? atof(e) / 100.0 : 0.85;
            if (pct < 0.5) pct = 0.5; if (pct > 0.99) pct = 0.99;
            e = getenv("GF_PACER_DYN_HEADROOM"); head = 1.0 + ((e && *e) ? atof(e) : 5.0) / 100.0;
            e = getenv("GF_PACER_DYN_MINFPS"); { double m = (e && *e) ? atof(e) : 72.0; if (m < 30.0) m = 30.0; pmax = 1.0 / m; }
            if (pmax < period) pmax = period;
            fprintf(stderr, "[V1050T] dynamic frame cap: %.0f..%.0f FPS from the p%.0f frame work + %.0f %% (GF_PACER_DYN_*)%c",
                    1.0 / pmax, 1.0 / period, pct * 100.0, (head - 1.0) * 100.0, 10);
        }
        if (dynper <= 0.0) dynper = period;
        double work = now - ret_t; if (work > 3.0 * dynper) work = 3.0 * dynper;
        ring[rpos] = work; rpos = (rpos + 1) & 127; if (rn < 128) rn++;
        if (++since >= 8 && rn >= 32) {
            since = 0;
            double t[128]; const int n = rn;
            for (int i = 0; i < n; i++) { const double v = ring[i]; int j = i; while (j > 0 && t[j - 1] > v) { t[j] = t[j - 1]; j--; } t[j] = v; }
            int qi = (int)(n * pct); if (qi >= n) qi = n - 1;
            double tgt = t[qi] * head + 0.0002;
            if (tgt < period) tgt = period;
            if (tgt > pmax) tgt = pmax;
            dynper += (tgt - dynper) * (tgt > dynper ? 0.5 : 0.08);
            if (dynper < period) dynper = period;
            if (dynper > pmax) dynper = pmax;
            if (now - lastlog >= 2.0 && (logged <= 0.0 || (dynper - logged > logged * 0.03 || logged - dynper > logged * 0.03))) {
                lastlog = now; logged = dynper;
                fprintf(stderr, "[V1050T] frame cap %.0f FPS (p%.0f work %.2f ms)%c", 1.0 / dynper, pct * 100.0, t[qi] * 1000.0, 10);
            }
        }
    }
    const double per = (adapt == 3 && dynper > 0.0) ? dynper : period * div;
    g_gf_pace_per = per;
    if (next == 0.0 || now - next > per) {           /* first frame, or more than a frame late: resync */
        if (next != 0.0) n_resync++;
        next = now;
    } else if (now < next) {
        double rem = next - now;
#ifdef _WIN32
        const double final_spin = gf_wait_lean_enabled() && spin > 0.00015 ? 0.00015 : spin;
        if (timer && rem > final_spin) {
            LARGE_INTEGER due; due.QuadPart = -(LONGLONG)((rem - final_spin) * 1e7);   /* 100 ns units, relative */
            if (due.QuadPart < 0 && SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE))
                WaitForSingleObject(timer, 100);
        }
        while ((now = gf_now_sec()) < next) YieldProcessor();
#else
        while ((now = gf_now_sec()) < next) { }
#endif
    } else {
        n_late++; late_sum += now - next;
        /* v1050S GF_PACER_REPHASE (default 1): a frame that starts late used to keep the old grid, so the NEXT frame
         * was short by the same amount (13 ms then 9 ms instead of 11 + 11 -> visible judder). Late by more than
         * 15 % of a period -> the schedule restarts from now. 0 = old. */
        { static int rp = -1; if (rp < 0) { const char* e = getenv("GF_PACER_REPHASE"); rp = (e && *e == '0') ? 0 : 1; }
          if (rp && now - next > per * 0.15) next = now; }
    }
    if (gf_pacer_stats_on()) gf_pacer_stats_note(ps_work, gf_now_sec() - ps_t_in, per);   /* v1129P */
    next += per;
    n_frames++;
    ret_t = gf_now_sec();
    if (now - rep_t >= 10.0) {
        if (rep_t != 0.0)
            fprintf(stderr, "[V1046P] pacer 10 s: %llu frames, %llu started late (mean %.2f ms), %llu resyncs%c",
                    n_frames, n_late, n_late ? late_sum * 1000.0 / (double)n_late : 0.0, n_resync, 10);
        rep_t = now; n_frames = n_late = n_resync = 0; late_sum = 0.0;
    }
    return 1;
}
/* v1115A: load settings.ini (written by PC Settings, rsx_ui_overlay.c menu_save) into the environment before anything
 * reads it, so saved values override the launcher's "set" lines. This loader was never ported from the inFamous
 * runner -> settings were saved but every start fell back to the .cmd values. GF_SETTINGS_INI=0 = off. */
static void gf_load_settings_ini(void)
{
    const char* off = getenv("GF_SETTINGS_INI");
    if (off && !strcmp(off, "0")) return;
    FILE* f = fopen("settings.ini", "r");
    if (!f) { fprintf(stderr, "[settings] no settings.ini -- launcher values\n"); return; }
    char line[512]; int n = 0;
    while (fgets(line, sizeof line, f)) {
        char* s = line; while (*s == ' ' || *s == '\t') s++;
        if (*s == ';' || *s == '#' || !*s) continue;
        char* eq = strchr(s, '='); if (!eq) continue;
        *eq = 0; char* k = s; char* v = eq + 1;
        for (char* e = eq - 1; e >= k && (*e == ' ' || *e == '\t'); e--) *e = 0;
        while (*v == ' ' || *v == '\t') v++;
        size_t L = strlen(v); while (L && (v[L-1] == '\n' || v[L-1] == '\r' || v[L-1] == ' ' || v[L-1] == '\t')) v[--L] = 0;
        if (!*k) continue;
#ifdef _WIN32
        _putenv_s(k, v);
#else
        setenv(k, v, 1);
#endif
        n++;
    }
    fclose(f);
    fprintf(stderr, "[settings] v1115A loaded %d values from settings.ini\n", n);
}
#ifdef _WIN32
#include <io.h>
#include <signal.h>
/* v1129L GF_LOG_BUFFERED=1 (default 0 = old: stdout unbuffered, stderr the CRT default = unbuffered). Every printf /
 * fprintf(stderr) from any thread (main guest, FIFO, SPU, audio) was one WriteFile under the stream lock, so threads
 * queued behind each other's syscalls (v1129B_01: main sampled 99 % inside printf). =1: 64 KB full buffering on a
 * redirected stdout and stderr (a console stays as it was), a helper thread flushes both every GF_LOG_FLUSH_MS (default
 * 250, 10..5000), and they are flushed at exit, on abort() and on an unhandled exception, so a crash still leaves the
 * log. The exit paths (window close, quit menus) already fflush both before TerminateProcess. Host logging only. */
static int s_log_buffered = 0;
static unsigned s_log_flush_ms = 250;
static LPTOP_LEVEL_EXCEPTION_FILTER s_log_prev_uef = NULL;
static void gf_log_flush_nolock(void) { _fflush_nolock(stdout); _fflush_nolock(stderr); }   /* dying: do not wait on a lock */
static LONG WINAPI gf_log_uef(EXCEPTION_POINTERS* ep)
{
    gf_log_flush_nolock();
    return s_log_prev_uef ? s_log_prev_uef(ep) : EXCEPTION_CONTINUE_SEARCH;
}
static void __cdecl gf_log_sigabrt(int) { gf_log_flush_nolock(); }
static void __cdecl gf_log_atexit(void) { fflush(stdout); fflush(stderr); }
static DWORD WINAPI gf_log_flusher(LPVOID)
{
    for (;;) { Sleep(s_log_flush_ms); fflush(stdout); fflush(stderr); }
    return 0;
}
static void gf_log_buffered_setup(void)
{
    const char* e = getenv("GF_LOG_BUFFERED");
    if (!(e && *e == '1')) return;
    const char* m = getenv("GF_LOG_FLUSH_MS");
    if (m && *m) { const int v = atoi(m); if (v >= 10 && v <= 5000) s_log_flush_ms = (unsigned)v; }
    fflush(stdout); fflush(stderr);
    int on = 0;
    if (!_isatty(_fileno(stdout)) && setvbuf(stdout, NULL, _IOFBF, 64 * 1024) == 0) on |= 1;
    if (!_isatty(_fileno(stderr)) && setvbuf(stderr, NULL, _IOFBF, 64 * 1024) == 0) on |= 2;
    if (!on) { fprintf(stderr, "[v1129L] buffered logs requested, but stdout/stderr are consoles -> left as they were%c", 10); return; }
    s_log_buffered = 1;
    s_log_prev_uef = SetUnhandledExceptionFilter(gf_log_uef);
    signal(SIGABRT, gf_log_sigabrt);
    atexit(gf_log_atexit);
    HANDLE h = CreateThread(NULL, 64 * 1024, gf_log_flusher, NULL, 0, NULL);
    if (h) CloseHandle(h);
    fprintf(stderr, "[v1129L] buffered logs ON (GF_LOG_BUFFERED; 64 KB on%s%s, flushed every %u ms + exit/abort/crash)%c",
            (on & 1) ? " stdout" : "", (on & 2) ? " stderr" : "", s_log_flush_ms, 10);
}
#endif
int main(int argc, char** argv)
{
    gf_load_settings_ini();   /* v1115A: before any getenv */
#ifdef _WIN32
    gf_log_buffered_setup();  /* v1129L: GF_LOG_BUFFERED (settings.ini may set it) */
#endif
    /* v1121H switches default off: startup owns the single enable log and placement aliases. */
    if (gf_wait_lean_enabled())
        fprintf(stderr, "[v1121H] lean blocking waits ON (GF_WAIT_LEAN; final spin <= 150 us)%c", 10);
    { const char* e = getenv("GF_PLACE2");
      if (e && *e == '1') {
#ifdef _WIN32
          if (!getenv("PS3_FIFO_CORE")) _putenv_s("PS3_FIFO_CORE", "1");
          if (!getenv("RF_CORE")) _putenv_s("RF_CORE", "1");
#else
          if (!getenv("PS3_FIFO_CORE")) setenv("PS3_FIFO_CORE", "1", 0);
          if (!getenv("RF_CORE")) setenv("RF_CORE", "1", 0);
#endif
          fprintf(stderr, "[v1121H] second-stage placement ON (GF_PLACE2; FIFO/render P-core claims, audio helper E-cores)%c", 10);
      } }
    if (argc < 2) {
        printf("usage: %s <PPU ELF>\n", argv[0]);
        return 2;
    }

#ifdef _WIN32
#pragma comment(lib, "winmm.lib")
    /* 1 ms timer resolution. The default granularity is about 15.6 ms, which
     * inflates every shorter wait the title makes and throttles the whole
     * thing. POSIX timers are already fine-grained. */
    timeBeginPeriod(1);
    if (!s_log_buffered) setvbuf(stdout, NULL, _IONBF, 0);   /* unbuffered: do not lose prints on a kill (v1129L: GF_LOG_BUFFERED=1 keeps the buffer) */
#endif

    printf("=== ps3recomp game runner ===\n");

    if (!alloc_guest_vm()) {
        fprintf(stderr, "ERROR: could not allocate the guest address space\n");
        return 1;
    }

    uint32_t entry = ppu_load_elf(argv[1]);
    if (!entry) {
        fprintf(stderr, "ERROR: could not load %s\n", argv[1]);
        return 1;
    }

    derive_vfs_root(argv[1]);
    printf("[boot] VFS root: %s\n", ppu_vfs_root);

    ppu_recomp_register();   /* lifted function table -> address map */
    ps3_load_prx_modules();  /* this game's lifted system PRX, if it has any */
    ppu_hle_init();          /* firmware import NID -> HLE handlers */
    ppu_sysprx_register();   /* boot-critical CRT */
    ppu_fs_register();       /* cellFs over the game directory */
    lv2_init_syscalls();     /* the lv2 syscall table */

    /* Install the guest-callback hook, then start the frame clock. It no-ops
     * until the title registers its vblank and flip handlers during init. */
    g_ps3_guest_caller = harness_guest_caller;
    CreateThread(NULL, 4u * 1024 * 1024, frame_clock, NULL, 0, NULL);
    gf_cutscene_init();      /* v1042S: hold Cross / Space to skip in-engine cutscenes (gf_cutscene.c) */
    gf_ragdoll_init();       /* v1054S: ragdoll deaths (gf_ragdoll.c), GF_RAG=0 = off, GF_RAG_PHYS=0 = log only */
    gf_prompts_start();      /* v1042G: SIXAXIS prompts -> keyboard/mouse or controller wording (gf_prompts.c) */

    printf("\n[boot] dispatching entry OPD 0x%08X (stack top 0x%08X)\n\n",
           entry, STACK_TOP);

    /* v1038P PS3_MAIN_THREAD_SETUP=1 (play.cmd; default 0 = old): this template harness never called the toolkit's
     * ps3_sampler_start, so the main guest thread -- the gameplay frame limiter -- ran at NORMAL priority on any
     * core (E-cores included), PS3_MAIN_CORE never reserved it a core, ps3_hz_set_main never ran (the backend's
     * [RFBOUND] fence wait and [SPIKE] io attribution read 0 for it) and PS3_PROF_MAIN could not start. ppu_run runs
     * the guest on THIS thread, so the setup has to happen here. Host scheduling only, no guest-visible change. */
    {
        const char* e = getenv("PS3_MAIN_THREAD_SETUP");
        if (e && *e == '1') {
#ifdef _WIN32
            fprintf(stderr, "[V1038P] main guest thread setup: ps3_sampler_start on hosttid %lu (P-cores + HIGHEST, PS3_MAIN_CORE, hz main, PS3_PROF_MAIN)%c",
                    (unsigned long)GetCurrentThreadId(), 10);
#endif
            ps3_sampler_start();
        } else {
            fprintf(stderr, "[V1038P] main guest thread setup off (PS3_MAIN_THREAD_SETUP=1 = P-cores + HIGHEST + core reservation)%c", 10);
        }
    }

    int rc = ppu_run(entry, STACK_TOP);
    printf("\n[boot] ppu_run returned %d (entry function unwound)\n", rc);

    /* A title that called sys_process_exit never reaches this line: that path
     * ends in the host exit(). Getting here means the entry function returned
     * instead, so hand back the status the guest published if it published
     * one. Returning a hardcoded 0 would report success for a run that never
     * got anywhere. */
    return g_sys_process_exit_called ? (int)g_sys_process_exit_code : 0;
}
