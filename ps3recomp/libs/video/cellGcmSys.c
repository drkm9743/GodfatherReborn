/*
 * ps3recomp - cellGcmSys HLE module implementation
 *
 * Manages RSX initialization, display buffer registration, flip control,
 * command buffer control, tile/zcull configuration, IO memory mapping,
 * report/label areas, and address-to-offset translation.
 *
 * Actual rendering is handled elsewhere -- this module just tracks state.
 */

#include "rsx_live_draw.h"
#include "rsx_draw_engine.h"
#include "cellGcmSys.h"
#include "../../runtime/platform/win32_compat.h"
#include "../../runtime/ppu/ppu_memory.h"   /* vm_write32 (translate + byte-swap, OOB-safe) */
#include "../../runtime/memory/vm.h"    /* VM_HLE_INJECT_BASE */
#include "rsx_commands.h"                    /* rsx_state, rsx_process_command_buffer */
#include "../../include/ps3emu/guest_call.h" /* g_ps3_guest_caller for direct handler dispatch */
#include "../../include/ps3emu/hz_stats.h"   /* v225A guest frame cadence for [SPIKE] */

/* Guest EA of the GCM context (begin/end/current/callback) the title writes its
 * command stream into; recorded by cellGcmSetupContext, drained by the RSX. */
static u32 s_gcm_context_ea = 0;
#include <stdio.h>
#include <string.h>
#include <time.h>   /* v50 semaphore acquire timeout */
#include <stdlib.h>
#include <stdint.h>
#if (defined(_M_X64) || defined(__x86_64__)) && defined(__AVX2__)   /* v242B: AVX2 builds only (PORT_CPU_COMPAT -> per-word loop) */
#include <immintrin.h>   /* v221C TC_VEC */
#define GCM_TC_VEC_X86 1
#endif
#if (defined(_M_X64) || defined(__x86_64__)) && defined(__SSSE3__)   /* v1047A GCM_FASTWALK: 128-bit byte swap */
#include <immintrin.h>
#define GCM_V1047_SSSE3 1
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

static inline u32 gcm_read_be32_direct(const u8* p)
{
    u32 raw;
    memcpy(&raw, p, sizeof(raw));
    return ps3_bswap32(raw);
}

/* ---------------------------------------------------------------------------
 * Timestamp helpers (same pattern as sys_timer)
 * -----------------------------------------------------------------------*/

#ifdef _WIN32
static LARGE_INTEGER s_qpc_freq;
static int           s_qpc_init = 0;

static void ensure_qpc_init(void)
{
    if (!s_qpc_init) {
        QueryPerformanceFrequency(&s_qpc_freq);
        s_qpc_init = 1;
    }
}

static u64 get_timestamp_ns(void)
{
    LARGE_INTEGER now;
    ensure_qpc_init();
    QueryPerformanceCounter(&now);
    /* Convert to nanoseconds: (count * 1e9) / freq */
    return (u64)((double)now.QuadPart * 1000000000.0 / (double)s_qpc_freq.QuadPart);
}
#else
static u64 get_timestamp_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ULL + (u64)ts.tv_nsec;
}
#endif

/* ---------------------------------------------------------------------------
 * Internal state
 * -----------------------------------------------------------------------*/

static int  s_gcm_initialized = 0;
static u32  s_flip_mode   = CELL_GCM_DISPLAY_VSYNC;
static u32  s_flip_status = CELL_GCM_FLIP_STATUS_DONE;

/* Count of guest flip requests (SetFlipCommand / SetPrepareFlip). The present
 * loop uses this as the frame boundary: presenting only when a flip arrived
 * keeps partially-drained frames off screen (the ticker otherwise presents on
 * a fixed 16ms clock, and a drain that catches the guest mid-frame -- e.g.
 * while it's blocked in the FIFO-wrap recycle callback -- would show a
 * clear+few-draws frame as a visible flicker). */
static volatile u32 s_flip_request_count = 0;

u32 cellGcm_flip_request_count(void)
{
    return s_flip_request_count;
}
static u32  s_debug_level = CELL_GCM_DEBUG_LEVEL0;

/* Display buffers */
static CellGcmDisplayInfo s_display_buffers[CELL_GCM_MAX_DISPLAY_BUFFER_NUM];
static int s_display_buffer_set[CELL_GCM_MAX_DISPLAY_BUFFER_NUM];
static u32 s_current_display_buffer_id = 0;
/* Set by the flip (guest thread, at a get==put frame boundary); consumed by
 * the ticker, which presents the accumulated batch BEFORE draining further --
 * presenting on a raw flip-count change raced the drain and showed empty or
 * mixed batches (wave: black flashes, layout flicker). */
static volatile int s_flip_pending = 0;
/* A flip _cellGcmSetFlipCommand queued in the FIFO: | bufferId, bit 8 = ours. */
#define GCM_FLIP_MARKER 0xFEAD0100u
#define GCM_FLIP_NOTE 0xFEAE0100u     /* v62: see gcm_flip_note_into_fifo */
volatile u32 g_gcm_fifo_flip_off = 0;
/* v1034F FLIP_WALKED_BOUNDARY: every flip the FIFO walker actually PASSES (v62 note or in-FIFO
 * flip marker) bumps g_gcm_fifo_flip_seq; g_gcm_fifo_flip_seq_fc is the guest flip-request count
 * at that moment. The D3D12 backend ends a frame at the first display clear after a walked flip
 * instead of after a guest-side (PPU-time) flip count change, which can land mid-frame. */
volatile u32 g_gcm_fifo_flip_seq = 0;
/* v1051P PRESENT_SCHED (rsx_d3d12_backend.c / Godfather main.cpp gf_pace_frame): shared state.
 *   g_ps_vbl_*      vblank grid from the backend's WaitForVBlank thread (QPC seconds): a vblank time, the refresh
 *                   period, when it was last updated; g_ps_on = 1 once the grid is valid and PRESENT_SCHED=1.
 *   g_ps_D          scheduling latency (s): a frame started at S is shown at the first vblank >= S + D (backend tunes it).
 *   g_gf_frame_V    the vblank the guest frame now running is scheduled for (pacer); g_gf_sched_dt its game timestep.
 *   g_ps_flip_V[]   per flip request number: the vblank its frame is scheduled for (recorded with the v62 flip note).
 *   g_gcm_walked_req newest flip request whose note the FIFO walker has passed. */
volatile double g_ps_vbl_anchor = 0.0, g_ps_vbl_period = 0.0, g_ps_vbl_stamp = 0.0, g_ps_D = 0.0;
volatile double g_gf_frame_V = 0.0, g_gf_sched_dt = 0.0;
volatile double g_ps_flip_V[64];
volatile u32 g_gcm_walked_req = 0;
volatile int g_ps_on = 0;
volatile u32 g_gcm_fifo_flip_seq_fc = 0;
volatile u32 g_gcm_flip_note_skips = 0;   /* v1034F: notes that could not be written */
static int s_v1034_title_segs = 0;        /* v1034F: title steps its own ring segments (v1032H) */
/* v1034F-2: IO offsets of the notes written by gcm_flip_note_into_fifo that the walker has not
 * reached yet. Only a note walked at one of these offsets counts as a flip in g_gcm_fifo_flip_seq:
 * the 1034black boot log walked 0xFEAE01xx words that no flip had written (seq 14 at guest flip 10,
 * 38 "flips" walked during guest flip 10..11 -- stale ring contents / desynced parses). */
static volatile u32 s_v1034_note_io[8];
static volatile u32 s_v1034_note_w = 0;
volatile u32 g_gcm_flip_note_stale = 0;
static int v1034f_on(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("FLIP_WALKED_BOUNDARY"); v = (e && *e == '1') ? 1 : 0; }
    return v;
}
/* ponytail: fixed headroom, far above one frame's commands (Simpsons ~10 KB);
 * derive it from the observed per-frame volume if a title outgrows it. */
#define GCM_FLIP_WRAP_BYTES 0x40000u

int cellGcm_take_flip_pending(void)
{
    int v = s_flip_pending;
    s_flip_pending = 0;
    return v;
}


/* Configuration */
static CellGcmConfig s_config;
/* Offset pages (1MB) known to be LOCAL-memory-derived (set by
 * cellGcmAddressToOffset when the guest converts a localAddress-range EA). */
static u8 s_local_offset_page[1024];

/* Command buffer control */
static CellGcmControl s_control;

/* Offset table storage
 *
 * set_offset_table/clear_offset_table keep these two host arrays in step with
 * the IO mappings, and that part was always fine. What was not: the tables
 * cellGcmGetOffsetTable hands out are indexed BY THE GUEST --
 * ioAddress[ea >> 20] and eaAddress[io >> 20] -- and it handed over 64-bit host
 * pointers through a struct whose guest form is two 4-byte fields, so the title
 * got two truncated addresses pointing at nothing it could read.
 *
 * The guest-visible copy lives in the reserved GCM window alongside the label
 * and control blocks, published on each call. 4096 pages covers the 4 GiB
 * address space at 1 MiB granularity, 8 KiB a side, and this is not a hot path.
 * 0xFFFF means "not mapped", as it does in the host tables. */
/* Default home for the HLE-visible GCM window; a port overrides it before
 * running guest code when its title claims this address range. */
uint32_t ppu_hle_inject_base = 0x20000000u;

static u16 s_io_address_table[65536];
static u16 s_ea_address_table[65536];

#define GCM_OFFSET_TABLE_PAGES   4096u                 /* 4 GiB / 1 MiB       */
#define GCM_OFFSET_TABLE_IO_ADDR (VM_HLE_INJECT_BASE + 0x3000u)           /* ea >> 20  -> io>>20 */
#define GCM_OFFSET_TABLE_EA_ADDR (VM_HLE_INJECT_BASE + 0x5000u)           /* io >> 20  -> ea>>20 */

/* Local memory bump allocator */
static u32 s_local_mem_allocated = 0;  /* next free offset in local memory */

/* IO memory mapping table */
typedef struct IoMapping {
    u32 ea;         /* effective address in main memory */
    u32 io;         /* IO offset (RSX-visible) */
    u32 size;       /* size of mapping */
    int active;     /* 1 if this slot is in use */
} IoMapping;

static IoMapping s_io_mappings[CELL_GCM_MAX_IO_MAPPINGS];
static int s_io_mapping_count = 0;

/* Callback handlers — these are GUEST OPD addresses passed by the
 * recompiled game, not host function pointers. Stored as uint32_t and
 * invoked through g_ps3_guest_caller (see ps3emu/guest_call.h). */

/* Handler mask exported for the RSX event delivery path. Bit layout matches
 * the RPCS3 SYS_RSX_EVENT enum (vblank=0x02, flip=0x04, user_cmd=0x80).
 * The runner's vblank tick reads this to decide which events to send to
 * the libgcm interrupt thread. Updated by Set*Handler below.
 *
 * If g_gcm_driver_info_ea is non-zero, the mask is also written to
 * driver_info + 0x12C0 so the game's own libgcm interrupt path can read it
 * directly from guest memory. The runner sets g_gcm_driver_info_ea during
 * RSX context allocation. */
uint32_t g_gcm_handler_mask = 0;
uint32_t g_gcm_driver_info_ea = 0;

static u32 s_flip_handler_opd     = 0;

/* GCM_FLIPCB_ONTICK=1: fire the guest flip handler only when the flip
 * completes (cellGcmTickFlip), not also when it is requested. */
static int gcm_flipcb_on_tick_only(void)
{
    static int v = -1;
    if (v < 0) v = getenv("GCM_FLIPCB_ONTICK") ? 1 : 0;
    return v;
}

static u32 s_vblank_handler_opd   = 0;
static u32 s_user_handler_opd     = 0;
static u32 s_second_v_handler_opd = 0;
/* YDKJ_HANDLERFIX: the handler OPD's code word (*(opd)) gets clobbered to 0 by a
 * mislifted store, so g_ps3_guest_caller reads code=0 -> "not registered" and the
 * game's per-flip/vblank callback never fires -> render/advance loop stalls. We
 * capture the code at Set*Handler time (OPD still valid) and restore the OPD word
 * before each invocation if it's been clobbered. */
extern u32 vm_read32(unsigned int);
static u32 s_flip_handler_code    = 0;
static u32 s_vblank_handler_code  = 0;
static void ydkj_restore_handler_opd(u32 opd, u32 code) {
    if (!getenv("YDKJ_HANDLERFIX") || !opd || !code) return;
    extern u8* vm_base; if (!vm_base) return;
    if (vm_read32(opd) == 0) {
        u8* p = vm_base + opd; p[0]=(u8)(code>>24); p[1]=(u8)(code>>16); p[2]=(u8)(code>>8); p[3]=(u8)code;
        static int _n=0; if((_n < 6 && _n++ < 6)) fprintf(stderr,"[HANDLERFIX] restored clobbered OPD 0x%08X code=0x%08X\n",opd,code);
    }
}
static void gcm_update_handler_mask(void)
{
    uint32_t m = 0;
    if (s_vblank_handler_opd) m |= 0x02;
    if (s_flip_handler_opd)   m |= 0x04;
    if (s_user_handler_opd)   m |= 0x80;
    g_gcm_handler_mask = m;
    if (g_gcm_driver_info_ea)
        vm_write32(g_gcm_driver_info_ea + 0x12C0, m);
}

/* Legacy host-typed slots kept around for any caller still treating
 * these as host pointers. New code should use the _opd slots. */
static CellGcmFlipHandler    s_flip_handler    = NULL;
static CellGcmVBlankHandler  s_vblank_handler  = NULL;
static CellGcmUserHandler    s_user_handler    = NULL;
static CellGcmSecondVHandler s_second_v_handler = NULL;

/* Flip timing */
static u64 s_last_flip_time = 0;

/* VBlank counter (incremented on each flip as approximation) */
static u32 s_vblank_count = 0;

/* IO map reservation tracking */
static u32 s_io_map_reserved = 0;  /* bytes reserved for IO mapping */

/* Default FIFO mode */
static s32 s_default_fifo_mode = 0;

/* Graphics / queue handlers */
static CellGcmGraphicsHandler s_graphics_handler = NULL;
static CellGcmQueueHandler    s_queue_handler    = NULL;

/* Second V / VBlank frequency */
static u32 s_second_v_frequency = 0;
static u32 s_vblank_frequency   = 0;

/* User command */
#ifdef _WIN32
static volatile LONG s_user_command = 0;
#define GCM_USER_STORE(cmd) InterlockedExchange(&s_user_command, (LONG)(cmd))
/* Pair the pending marker with its cause so a later producer cannot
 * overwrite a cause already claimed by the callback pump. */
static volatile LONG64 s_user_pending = 0;
#define GCM_USER_PENDING_STORE(value) InterlockedExchange64(&s_user_pending, (LONG64)(value))
#define GCM_USER_PENDING_TAKE() ((u64)InterlockedExchange64(&s_user_pending, 0))
#else
#include <stdatomic.h>
static atomic_uint s_user_command = 0;
#define GCM_USER_STORE(cmd) atomic_store(&s_user_command, (cmd))
static _Atomic(u64) s_user_pending = 0;
#define GCM_USER_PENDING_STORE(value) atomic_store(&s_user_pending, (value))
#define GCM_USER_PENDING_TAKE() atomic_exchange(&s_user_pending, 0)
#endif

/* Tile configuration (up to 15 tiles, 8 commonly used) */
static CellGcmTileInfo s_tiles[CELL_GCM_MAX_TILE_COUNT];

/* Zcull configuration (8 zcull regions) */
static CellGcmZcullInfo s_zcull[CELL_GCM_MAX_ZCULL_COUNT];

/* Report data area (256 slots, each 16 bytes) */
static CellGcmReportData s_report_data[CELL_GCM_MAX_REPORT_COUNT];

/* Label area (256 labels, each u32). Labels live in GUEST memory so the
 * recompiled game can poll them via vm_read32; cellGcmGetLabelAddress returns a
 * guest address into this window (16-byte spaced as on hardware). A host array
 * pointer would be read as a guest offset and land out of bounds. */
#define GCM_LABEL_GUEST_BASE  (VM_HLE_INJECT_BASE + 0x0000u)
#define GCM_LABEL_STRIDE      0x10u
static u32 s_labels[CELL_GCM_MAX_LABEL_COUNT];

/* GCM control register (put/get/ref). This MUST live in guest memory: the title
 * takes the pointer from cellGcmGetControlRegister, writes `put` when it kicks
 * the FIFO, and spins reading `get`/`ref` to know the RSX has caught up. A host
 * pointer would be read through vm_base as a guest address and never update, so
 * every FIFO-space / cellGcmFinish wait would hang. Placed just past the 256
 * label slots (0x03000000..0x03001000). Fields are big-endian (vm_write32). */
#define GCM_CONTROL_GUEST_ADDR (VM_HLE_INJECT_BASE + 0x2000u)

/* Synthetic guest code EA for the FIFO command-buffer-full callback. The title's
 * inline gcmReserve calls context->callback(context, count) when `current` nears
 * `end`; we point the context's callback OPD at this EA and register it in the
 * PPU function table (ppu_sysprx.cpp: hle_gcm_callback) so the indirect call
 * routes back into cellGcm_fifo_recycle(). Lives in the injected control page,
 * never real guest code. MUST match GCM_FIFO_CALLBACK_SENTINEL_EA there. */
#define GCM_FIFO_CALLBACK_SENTINEL_EA (VM_HLE_INJECT_BASE + 0x2F00u)

/* EA the ticker's RSX drain has consumed up to. Exposed so the wrap callback can
 * wait for the RSX to catch up before recycling the ring (else the frame's
 * just-written commands are lost). Updated in cellGcm_rsx_process_fifo. */
volatile u32 g_gcm_fifo_drained_ea = 0;

/* The guest variable cellGcmInit's ctx_out pointed at -- the title's own
 * gCellGcmCurrentContext. A title that switches to a command buffer of its own
 * does it by repointing this, and until something reads it back the runtime has
 * no way to notice. GCM_CTXDBG=1 reports it. */
static u32 s_gcm_ctx_out_ea = 0;

/* Set by cellGcm_syscall_bringup: this run reaches RSX through the lv2 sys_rsx_*
 * syscalls (libs/video/sys_rsx.c) rather than through the HLE cellGcm* imports. */
static int s_gcm_syscall_mode = 0;

/* ---------------------------------------------------------------------------
 * Internal helpers
 * -----------------------------------------------------------------------*/

/* Populate the offset table entries for an IO mapping */
static void populate_offset_table(u32 ea, u32 io, u32 size)
{
    /*
     * The offset table uses 1MB pages (20-bit shift).
     * ioAddress[ea >> 20] = io >> 20   (EA -> IO offset)
     * eaAddress[io >> 20] = ea >> 20   (IO offset -> EA)
     */
    /* Round UP: a mapping is described by whole 1MB table entries, so a size
     * that is not a multiple of 1MB still needs its final partial page mapped.
     * Truncating dropped that page -- and dropped the mapping ENTIRELY for any
     * size below 1MB (pages == 0) -- after which cellGcmResolveOffset finds no
     * IO entry and falls back to VRAM, where the guest never wrote anything. */
    u32 pages = (size + 0xFFFFFu) >> 20;  /* number of 1MB pages, rounded up */
    u32 ea_page = ea >> 20;
    u32 io_page = io >> 20;

    for (u32 i = 0; i < pages && (ea_page + i) < 65536 && (io_page + i) < 65536; i++) {
        s_io_address_table[ea_page + i] = (u16)(io_page + i);
        s_ea_address_table[io_page + i] = (u16)(ea_page + i);
    }
}

/* Clear offset table entries for an IO mapping */
static void clear_offset_table(u32 ea, u32 io, u32 size)
{
    u32 pages = size >> 20;
    u32 ea_page = ea >> 20;
    u32 io_page = io >> 20;

    for (u32 i = 0; i < pages && (ea_page + i) < 65536 && (io_page + i) < 65536; i++) {
        s_io_address_table[ea_page + i] = 0xFFFF;
        s_ea_address_table[io_page + i] = 0xFFFF;
    }
}

/* Find an IO mapping by EA */
static IoMapping* find_mapping_by_ea(u32 ea)
{
    for (int i = 0; i < CELL_GCM_MAX_IO_MAPPINGS; i++) {
        if (s_io_mappings[i].active && s_io_mappings[i].ea == ea)
            return &s_io_mappings[i];
    }
    return NULL;
}

/* Find an IO mapping by IO offset */
static IoMapping* find_mapping_by_io(u32 io)
{
    for (int i = 0; i < CELL_GCM_MAX_IO_MAPPINGS; i++) {
        if (s_io_mappings[i].active && s_io_mappings[i].io == io)
            return &s_io_mappings[i];
    }
    return NULL;
}

/* Find a free IO mapping slot */
static IoMapping* find_free_mapping(void)
{
    for (int i = 0; i < CELL_GCM_MAX_IO_MAPPINGS; i++) {
        if (!s_io_mappings[i].active)
            return &s_io_mappings[i];
    }
    return NULL;
}

/* Valid tiled pitch sizes (power-of-two aligned, common RSX values) */
static const u32 s_valid_pitches[] = {
    0x0200, 0x0300, 0x0400, 0x0500, 0x0600, 0x0700, 0x0800,
    0x0A00, 0x0C00, 0x0D00, 0x0E00, 0x1000, 0x1400, 0x1800,
    0x1A00, 0x1C00, 0x2000, 0x2800, 0x3000, 0x3400, 0x3800,
    0x4000, 0x5000, 0x6000, 0x6800, 0x7000, 0x8000, 0xA000,
    0xC000, 0xD000, 0xE000, 0x10000
};
static const int s_valid_pitch_count = sizeof(s_valid_pitches) / sizeof(s_valid_pitches[0]);

/* ---------------------------------------------------------------------------
 * API implementations -- Initialization
 * -----------------------------------------------------------------------*/

/* NID: 0xB2E761D4 */
s32 cellGcmInit(u32 cmdSize, u32 ioSize, u32 ioAddress)
{
    printf("[cellGcmSys] Init(cmdSize=0x%X, ioSize=0x%X, ioAddr=0x%08X)\n",
           cmdSize, ioSize, ioAddress);

    if (s_gcm_initialized) {
        printf("[cellGcmSys] WARNING: already initialized\n");
        return CELL_GCM_ERROR_FAILURE;
    }

    memset(s_display_buffers, 0, sizeof(s_display_buffers));
    memset(s_display_buffer_set, 0, sizeof(s_display_buffer_set));
    memset(&s_config, 0, sizeof(s_config));
    memset(&s_control, 0, sizeof(s_control));
    memset(s_io_mappings, 0, sizeof(s_io_mappings));
    memset(s_tiles, 0, sizeof(s_tiles));
    memset(s_zcull, 0, sizeof(s_zcull));
    memset(s_report_data, 0, sizeof(s_report_data));
    memset(s_labels, 0, sizeof(s_labels));
    memset(s_io_address_table, 0xFF, sizeof(s_io_address_table));
    memset(s_ea_address_table, 0xFF, sizeof(s_ea_address_table));

    /*
     * Populate a plausible configuration.
     * In a real implementation these would reflect actual allocated memory.
     */
    s_config.localAddress    = 0xC0000000;  /* typical RSX local mem base */
    s_config.localSize       = 256 * 1024 * 1024;  /* 256 MB */
    s_config.ioAddress       = ioAddress;
    s_config.ioSize          = ioSize;
    s_config.memoryFrequency = 650000000;   /* 650 MHz */
    s_config.coreFrequency   = 500000000;   /* 500 MHz */

    s_local_mem_allocated = 0;
    s_io_mapping_count = 0;
    /* Unmapped = 0xFFFF. The tables are static (zero-initialized), so without
     * this every unmapped EA page silently aliased io page 0 (and vice versa),
     * producing plausible-but-wrong offsets instead of a translation failure. */
    memset(s_io_address_table, 0xFF, sizeof(s_io_address_table));
    memset(s_ea_address_table, 0xFF, sizeof(s_ea_address_table));
    s_current_display_buffer_id = 0;
    s_flip_handler = NULL;
    s_vblank_handler = NULL;
    s_user_handler = NULL;
    s_second_v_handler = NULL;
    s_last_flip_time = get_timestamp_ns();
    s_flip_status = CELL_GCM_FLIP_STATUS_DONE;
    s_flip_mode = CELL_GCM_DISPLAY_VSYNC;
    s_debug_level = CELL_GCM_DEBUG_LEVEL0;
    s_vblank_count = 0;
    s_io_map_reserved = 0;
    s_default_fifo_mode = 0;
    s_graphics_handler = NULL;
    s_queue_handler = NULL;
    s_second_v_frequency = 0;
    s_vblank_frequency = 0;
    GCM_USER_STORE(0);
    GCM_USER_PENDING_STORE(0);

    /* Set up the initial IO mapping for the command buffer region */
    if (ioAddress != 0 && ioSize > 0) {
        populate_offset_table(ioAddress, 0, ioSize);

        s_io_mappings[0].ea     = ioAddress;
        s_io_mappings[0].io     = 0;
        s_io_mappings[0].size   = ioSize;
        s_io_mappings[0].active = 1;
        s_io_mapping_count = 1;
    }

    /* Zero the guest-visible control register (put/get/ref). */
    vm_write32(GCM_CONTROL_GUEST_ADDR + 0, 0);
    vm_write32(GCM_CONTROL_GUEST_ADDR + 4, 0);
    vm_write32(GCM_CONTROL_GUEST_ADDR + 8, 0);

    /* Watch SPU DMA for pushbuffers. A title that builds its command stream on
     * an SPU never tells libgcm where that stream is, so the write is the only
     * evidence of it -- see gcm_note_spu_put. Costs one NULL check per SPU DMA
     * for titles that do not. */
    { extern void (*g_spu_put_hook)(u32 ea, u32 size);
      void gcm_note_spu_put(u32 ea, u32 size);       /* defined with the walker */
      void gcm_forget_spu_blocks(void);
      gcm_forget_spu_blocks();
      g_spu_put_hook = gcm_note_spu_put; }

    s_gcm_initialized = 1;
    return CELL_OK;
}

/* Reusable _cellGcmInitBody core (NID 0x15BAE46B). See cellGcmSys.h for why the
 * guest vm is injected as callbacks. The CellGcmContextData layout here is the
 * one proven in the shipping Simpsons port (begin@0/end@4/current@8/callback@C);
 * getting it wrong makes the game read `current` as the callback OPD and stall
 * in cellGcmFlush. The command buffer is the start of the game's IO region. */
u32 cellGcmSetupContext(u32 ctx_out_addr, u32 cmdSize, u32 ioSize, u32 ioAddress,
                        CellGcmGuestAlloc galloc, CellGcmGuestWrite32 gwrite32)
{
    if (cmdSize < 0x10000)
        cmdSize = 0x10000;

    /* PSL1GHT's gcmInitBody passes ioAddress = 0, expecting gcm to allocate the
     * IO region itself (the official SDK has the app allocate and pass it).
     * Taking the 0 literally put the FIFO command buffer at guest address 0 --
     * the guest then wrote its command stream over low memory and every offset
     * resolution failed. Allocate a guest region of ioSize instead. */
    if (ioAddress == 0 && galloc)
        ioAddress = galloc(ioSize ? ioSize : 0x100000, 0x100000);

    cellGcmInit(cmdSize, ioSize, ioAddress);

    u32 cmdbuf = ioAddress;                 /* FIFO lives at the IO region start */
    u32 cdata  = galloc ? galloc(16, 16) : 0;
    if (cdata && gwrite32) {
        gwrite32(cdata + 0x0, cmdbuf);              /* begin   */
        gwrite32(cdata + 0x4, cmdbuf + cmdSize);    /* end     */
        gwrite32(cdata + 0x8, cmdbuf);              /* current */
        /* Command-buffer-full callback: a guest OPD {func, toc} routed to
         * hle_gcm_callback -> cellGcm_fifo_recycle. Without a real callback the
         * ring never recycles and the FIFO wedges once `current` reaches `end`. */
        u32 opd = galloc ? galloc(8, 8) : 0;
        if (opd) {
            gwrite32(opd + 0x0, GCM_FIFO_CALLBACK_SENTINEL_EA);  /* func EA */
            gwrite32(opd + 0x4, 0);                              /* toc (unused) */
            gwrite32(cdata + 0xC, opd);                          /* callback OPD */
        } else {
            gwrite32(cdata + 0xC, 0);
        }
        if (ctx_out_addr)
            gwrite32(ctx_out_addr, cdata);          /* *context = &ctxdata */
        s_gcm_ctx_out_ea = ctx_out_addr;   /* the title's gCellGcmCurrentContext */
        s_gcm_context_ea = cdata;                   /* RSX drains commands from here */
    }
    return cdata;
}

s32 cellGcmGetConfiguration(CellGcmConfig* config)
{
    /* `config` is a GUEST address; write the 6 u32 fields big-endian via
     * vm_write32 (a raw *config = s_config faults / is host-endian). The game
     * reads localAddress/localSize from here to lay out VRAM framebuffers. */
    uint32_t cfg = (uint32_t)(uintptr_t)config;
    if (!cfg)
        return CELL_GCM_ERROR_INVALID_VALUE;

    vm_write32(cfg +  0, s_config.localAddress);
    vm_write32(cfg +  4, s_config.ioAddress);
    vm_write32(cfg +  8, s_config.localSize);
    vm_write32(cfg + 12, s_config.ioSize);
    vm_write32(cfg + 16, s_config.memoryFrequency);
    vm_write32(cfg + 20, s_config.coreFrequency);
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Command buffer control
 * -----------------------------------------------------------------------*/

/* NID: 0x8572A8E0 */
CellGcmControl* cellGcmGetControlRegister(void)
{
    /* Return the GUEST address of the control register (not &s_control, a host
     * pointer). The recompiled title reads/writes it through vm_base, so it must
     * be a guest EA for put/get/ref to actually flow. */
    return (CellGcmControl*)(uintptr_t)GCM_CONTROL_GUEST_ADDR;
}

/* ---------------------------------------------------------------------------
 * Display / flip
 * -----------------------------------------------------------------------*/

/* NID: 0xDB23E867 */
u32 cellGcmGetCurrentField(void)
{
    /* Returns 0 or 1 for interlaced; always 0 for progressive. */
    return 0;
}

/* NID: 0xA53D12AE */
void cellGcmSetFlipMode(u32 mode)
{
    printf("[cellGcmSys] SetFlipMode(mode=%u)\n", mode);
    s_flip_mode = mode;
}

/* NID: 0xC44D8F34 */
void cellGcmSetWaitFlip(void)
{
    /* Block until the pending flip completes (the present thread marks it done
     * via cellGcmTickFlip, ~once per display refresh). This throttles the title
     * to vsync; without it the guest loops unthrottled and many frames batch
     * into a single host present -> the console text stacks/duplicates. */
#ifdef _WIN32
    __declspec(dllimport) void __stdcall Sleep(unsigned long);
#endif  /* elsewhere: runtime/platform/win32_compat.h */
    /* A libgcm WaitFlip is queued on the RSX; it does not have to park the PPU.
     * Some older ports relied on the host-side wait below for pacing, so retain
     * it by default. Titles with their own limiter can request the hardware-like
     * path. Leave the status WAITING: the 60 Hz ticker and ppu_gcm_pump publish
     * DONE, and the title's subsequent GetFlipStatus poll observes it. */
    static int s_wf_gpu = -1;
    if (s_wf_gpu < 0) {
        const char* e = getenv("GCM_WAITFLIP_GPU");
        s_wf_gpu = (e && *e && *e != '0') ? 1 : 0;
        fprintf(stderr, "[GCM] v1035P cellGcmSetWaitFlip GPU-side mode %s (GCM_WAITFLIP_GPU)%c",
                s_wf_gpu ? "ON" : "off", 10);
    }
    if (s_wf_gpu) return;
    /* v1031P: deliver the pending flip tick from INSIDE the wait. The ticker (frame_clock ->
     * cellGcmTickFlip) only marks the flip pending; s_flip_status turns DONE in ppu_gcm_pump, which
     * runs at HLE-call entry. The waiting thread makes no HLE calls while it sleeps here, so unless
     * another guest thread happened to call an import, every wait ran to the 64-iteration cap:
     * Godfather (main thread, once per frame) spent 100-130 ms per call, ~6 s of every 10 s
     * ([FREEZE-SLOW] "HLE 0x983FB9AA 67x 6923 ms max 127"). Pumping here ends the wait at the next
     * 60 Hz tick (<= ~16 ms), the vsync pacing the comment above intends. The pump is reentrancy-
     * guarded and is exactly what this HLE call's entry already ran on this thread.
     * GCM_WAITFLIP_PUMP=0 = old behaviour; GCM_WAITFLIP_MAX=<n> = iteration cap (default 64). */
    extern void ppu_gcm_pump(void);
    static int s_wf_pump = -1, s_wf_cap = 64;
    if (s_wf_pump < 0) {
        const char* e = getenv("GCM_WAITFLIP_PUMP");
        const char* m = getenv("GCM_WAITFLIP_MAX");
        s_wf_cap = m ? atoi(m) : 64;
        if (s_wf_cap < 0) s_wf_cap = 0;
        s_wf_pump = (e && *e == '0') ? 0 : 1;
        fprintf(stderr, "[GCM] v1031P cellGcmSetWaitFlip: pump pending flip ticks while waiting %s, cap %d x 1 ms (GCM_WAITFLIP_PUMP / GCM_WAITFLIP_MAX)%c",
                s_wf_pump ? "ON" : "off", s_wf_cap, 10);
    }
    for (int i = 0; i < s_wf_cap && s_flip_status == CELL_GCM_FLIP_STATUS_WAITING; i++) {
        if (s_wf_pump) {
            ppu_gcm_pump();
            if (s_flip_status != CELL_GCM_FLIP_STATUS_WAITING) break;
        }
        Sleep(1);
    }
    s_flip_status = CELL_GCM_FLIP_STATUS_DONE;
}

/* NID: 0xDF6476BD. The same wait without libgcm's command-buffer space check;
 * this HLE writes no command, so it is the same call. GH3 makes it ~36 times
 * a boot and got a faked CELL_OK (no wait at all) before this was registered. */
void cellGcmSetWaitFlipUnsafe(void) { cellGcmSetWaitFlip(); }

/* NID: 0x51C9D62B */
void cellGcmResetFlipStatus(void)
{
    { static int n = 0;
      if (getenv("FLIP_DBG") && (n < 12 && n++ < 12))
          fprintf(stderr, "[FLIP] ResetFlipStatus%c", 10); }
    s_flip_status = CELL_GCM_FLIP_STATUS_WAITING;
}

static void v1048_flipstatus_note(u32 st);   /* v1048D, below */
/* NID: 0xE315A0B2 */
u32 cellGcmGetFlipStatus(void)
{
    /* Avoid host-calling the guest OPDs directly (they're not valid host
     * function pointers). Use cellGcmTickVBlank() / TickFlip() instead,
     * which dispatch via g_ps3_guest_caller — see below. */
    /* Report the CURRENT status; cellGcmTickFlip (the ticker's 60Hz vblank
     * beat) completes pending flips. The old self-completing version made
     * every wait-for-flip loop exit on its first poll, so titles ran
     * completely unpaced (wave: 95 fps with a fixed-dt simulation). */
    { static int n = 0;
      if (getenv("FLIP_DBG") && (n < 24 && n++ < 24))
          fprintf(stderr, "[FLIP] GetFlipStatus -> %u (0=DONE 1=WAITING)%c",
                  s_flip_status, 10); }
    v1048_flipstatus_note(s_flip_status);   /* v1048D stats (V1048D_PHASE / V1048D_STATS) */
    return s_flip_status;
}

/* Host-side ticks. The game's host driver (e.g. flow main.cpp) calls
 * these periodically so registered guest handlers fire — that's what
 * drives the title-screen state machine for many games. */
#include "ps3emu/guest_call.h"

/* ---------------------------------------------------------------------------
 * Serialized vblank/flip/user handler delivery.
 *
 * Tick producers publish pending work. The runner pumps it at HLE boundaries
 * or on a dedicated callback thread. The pump serializes guest handlers with
 * one another; it does not stop unrelated guest execution on other threads.
 * -----------------------------------------------------------------------*/
#ifdef _WIN32
#include <windows.h>
static volatile LONG s_gcm_pending = 0;    /* bit0 = vblank, bit1 = flip */
#define GCM_PENDING_SET(bits)  InterlockedOr(&s_gcm_pending, (bits))
#define GCM_PENDING_TAKE()     InterlockedExchange(&s_gcm_pending, 0)
static volatile LONG s_gcm_pumping = 0;
#define GCM_PUMP_TRY_ENTER() (InterlockedCompareExchange(&s_gcm_pumping, 1, 0) == 0)
#define GCM_PUMP_LEAVE()     InterlockedExchange(&s_gcm_pumping, 0)
#else
#include <stdatomic.h>
static atomic_int s_gcm_pending = 0;
#define GCM_PENDING_SET(bits)  atomic_fetch_or(&s_gcm_pending, (bits))
#define GCM_PENDING_TAKE()     atomic_exchange(&s_gcm_pending, 0)
static atomic_flag s_gcm_pumping = ATOMIC_FLAG_INIT;
#define GCM_PUMP_TRY_ENTER() (!atomic_flag_test_and_set_explicit(&s_gcm_pumping, memory_order_acquire))
#define GCM_PUMP_LEAVE()     atomic_flag_clear_explicit(&s_gcm_pumping, memory_order_release)
#endif

/* Called by the vblank ticker thread. NO guest code -- advance the vblank count
 * and mark a vblank+flip tick pending for the main thread to deliver. */
static u32 s_fifo_getoff;   /* tentative: defined below */

void cellGcm_request_tick(void)
{
    s_vblank_count++;
    /* GCM_FLIP_NEEDS_FIFO=1: only mark a flip pending once the walker has
     * actually consumed up to `put`.
     *
     * A title's fifo-finish can use flip status as a fast path -- Twisted
     * Metal's reads it first and, if the flip is done, skips waiting for `get`
     * entirely. Completing a flip on the 60 Hz beat regardless of FIFO
     * progress therefore tells it the RSX has caught up when it has not, and
     * it recycles a FIFO block the walker is still inside, overwriting the
     * JUMP out of it. Off by default: a title whose FIFO never drains would
     * stop flipping entirely. */
    { static int need = -1;
      if (need < 0) { const char* e = getenv("GCM_FLIP_NEEDS_FIFO"); need = e ? atoi(e) : 0; }
      if (need) {
          u32 put = vm_read32(GCM_CONTROL_GUEST_ADDR + 0);
          GCM_PENDING_SET(s_fifo_getoff == put ? 3 : 1);
          return;
      } }
    GCM_PENDING_SET(3);
}

/* Deliver pending handlers on the calling host thread. Serialize the entire
 * claim-and-deliver sequence across threads: an HLE boundary and a host ticker
 * may both pump, and newer causes must not overtake already claimed callbacks.
 * A nonblocking guard also prevents nested callbacks from reentering. Leave
 * pending notifications untouched when another pump is active. */
void ppu_gcm_pump(void)
{
    /* v157: plain early-out before the CAS. ppu_gcm_pump runs on EVERY ps3_hle_call of every guest thread
     * (~1% of the main thread: 3 locked atomics on cache lines shared by all guest threads). Nothing pending
     * is the overwhelmingly common case; a bit set right after this read is picked up by the next HLE call,
     * exactly as if it had arrived after the pump finished. GCM_PUMP_EARLY=0 = old. */
    { static int eo = -1;
      if (eo < 0) { const char* e = getenv("GCM_PUMP_EARLY"); eo = (e && *e == '0') ? 0 : 1; }
#ifdef _WIN32
      if (eo && s_gcm_pending == 0 && s_user_pending == 0) return;
#else
      if (eo && atomic_load_explicit(&s_gcm_pending, memory_order_relaxed) == 0 &&
          atomic_load_explicit(&s_user_pending, memory_order_relaxed) == 0) return;
#endif
    }
    if (!GCM_PUMP_TRY_ENTER()) return;
    long p = (long)GCM_PENDING_TAKE();
    u64 user = GCM_USER_PENDING_TAKE();
    /* GCM_PUMP_DBG=1: how often each guest handler is actually delivered. */
    { static int d = -1; static unsigned nv, nf, calls;
      if (d < 0) d = getenv("GCM_PUMP_DBG") ? 1 : 0;
      if (d) { calls++; if (p & 1) nv++; if (p & 2) nf++;
               if ((calls % 20000) == 0)
                   fprintf(stderr, "[gcm-pump] calls=%u vblank=%u flip=%u vcount=%u%c",
                           calls, nv, nf, s_vblank_count, 10); } }
    if ((p & 1) && s_vblank_handler_opd && g_ps3_guest_caller) {
        ydkj_restore_handler_opd(s_vblank_handler_opd, s_vblank_handler_code);
        g_ps3_guest_caller(s_vblank_handler_opd, (uint64_t)s_vblank_count,
                           0, 0, 0, 0, 0, 0, 0);
    }
    if (p & 2) {
        s_flip_status = CELL_GCM_FLIP_STATUS_DONE;
        if (s_flip_handler_opd && g_ps3_guest_caller) {
            ydkj_restore_handler_opd(s_flip_handler_opd, s_flip_handler_code);
            g_ps3_guest_caller(s_flip_handler_opd, 1, 0, 0, 0, 0, 0, 0, 0);
        }
    }
    if (user) {
        u32 cmd = (u32)user;
        if (s_user_handler_opd && g_ps3_guest_caller)
            g_ps3_guest_caller(s_user_handler_opd, (uint64_t)cmd, 0, 0, 0, 0, 0, 0, 0);
    }
    GCM_PUMP_LEAVE();
}

/* Back-compat: the old direct entry points now just mark a tick pending (so any
 * caller other than the ticker still works) -- delivery stays on the main thread. */
void cellGcmTickVBlank(void) { s_vblank_count++; GCM_PENDING_SET(1); }
/* Only deliver a flip completion when a flip is actually outstanding.
 *
 * Ticking unconditionally at 60 Hz fires the guest's flip handler even when
 * the guest never asked for a flip. A title whose handler touches the flip
 * status then livelocks: Tokyo Jungle's main loop is
 *
 *     do { usleep(100); } while (cellGcmGetFlipStatus() != DONE);
 *     cellGcmResetFlipStatus();
 *
 * and its handler runs code that puts the status back to WAITING, so the
 * poll never saw DONE and the game never rendered another frame. On hardware
 * the callback IS the flip interrupt -- no flip, no callback. */
/* v1049G GCM_FLIP_GATE=1 (default 0 = old; Godfather's launchers set 1): a flip completes when the FIFO walker has
 * EXECUTED it, as on hardware (the flip interrupt fires when the RSX reaches the flip command), not at the next
 * frame-clock tick regardless of where the walker is. Before, the 60-180 Hz ticker completed every flip at once,
 * so the guest (and its SPU jobs) ran several frames ahead of the walker and recycled per-frame buffers the walker
 * had not read yet: EA's command lists CALLed from the ring were rewritten under it (FIFO desync: "[RSX-UNK] method
 * 0x0000" then a CALL/JUMP to unmapped IO and a resync that drops commands = corrupted / half-drawn frames), SPU
 * labels ran past the walker's semaphore acquires ("[SEMA] acquire timed out ... label > want -- skipped"), and
 * batches of 3-30k records spanned several guest frames ([FRAME-STALL] merges). Zero resyncs in ~150 soaks from
 * v1037E to v1046; 5 of the 12 v1047/v1048 soaks desynced, two then froze in the ring manager (func_008A3454).
 * Gate: the tick completes a flip only when the walker passed the v62 note of the CURRENT request. Released anyway
 * (old behaviour) when that flip has no note (ring tail: note skipped), when the walker has drained everything the
 * guest submitted (note not flushed yet: holding could never end), or after GCM_FLIP_GATE_MS (default 200).
 * The walker passing the note completes the flip at once (no wait for the next tick -- what GCM_FLIP_ON_WALK did).
 * Log "[V1049G]" once and every 10 s: completions by walk / by tick, ticks held, releases (drained / no note /
 * timeout). */
static u32 s_fifo_getoff_pub(void);   /* v1049G: walker get (defined after s_fifo_getoff) */
static int s_v1049_gate = -1;
static u32 s_v1049_lag = 1u, s_v1049_lag_now = 0u;   /* v1049L: flips the guest may lead the walker by (gate 3) */
static long s_v1049_gate_ms = 200;
static volatile u32 s_v1049_noted_req = 0;    /* request number the newest written note belongs to */
static volatile u32 s_v1049_walked_req = 0;   /* newest request whose note the walker has passed */
static volatile unsigned long s_v1049_n[6];   /* 0 walk, 1 tick (passed), 2 held, 3 drained, 4 no note, 5 timeout */
static int v1049_gate_on(void)
{
    if (__builtin_expect(s_v1049_gate < 0, 0)) {
        const char* e = getenv("GCM_FLIP_GATE"); const char* t = getenv("GCM_FLIP_GATE_MS");
        if (t && *t) { long v = atol(t); s_v1049_gate_ms = v < 1 ? 1 : v; }
        s_v1049_gate = (e && (*e == '1' || *e == '2' || *e == '3')) ? *e - '0' : 0;
        { const char* l = getenv("GCM_FLIP_GATE_LAG"); s_v1049_lag = (l && *l) ? (u32)atoi(l) : 1u; if (s_v1049_lag > 8u) s_v1049_lag = 8u; }
        fprintf(stderr, "[V1049G] flip completes when the FIFO walker executes it (GCM_FLIP_GATE): %s, fallback %ld ms%c",
                s_v1049_gate == 1 ? "ON" : s_v1049_gate == 2 ? "only during a cutscene skip" :
                s_v1049_gate == 3 ? "only when the walker is behind by more than GCM_FLIP_GATE_LAG flips" : "off", s_v1049_gate_ms, 10);
        if (s_v1049_gate == 3) fprintf(stderr, "[V1049L] gate 3: the guest may run %u flip(s) ahead of the FIFO walker; held only beyond that (cutscene skip: 0)%c", s_v1049_lag, 10);
    }
    /* v1049L GCM_FLIP_GATE=3 (user: full gate = choppy camera at 180 FPS, no gate = desync after a free-aim lock-on swung
     * the camera and the walker fell behind): the flip completes once the walker is within GCM_FLIP_GATE_LAG (default
     * 1) flips of it -- a small lag never waits (pacing as ungated), a runaway guest (several frames ahead, recycling
     * command lists the walker has not read) is held. During a skip the lag is 0. */
    if (s_v1049_gate == 3) { extern volatile int g_gf_ffwd; s_v1049_lag_now = g_gf_ffwd ? 0u : s_v1049_lag; return 1; }
    s_v1049_lag_now = 0u;
    /* v1049K GCM_FLIP_GATE=2: only while a cutscene skip runs (g_gf_ffwd, cellPad.c: no frame limiter, the guest
     * races ahead and rewrote EA's command lists under the walker -- user log: [RSX-UNK] method 0x0000 right after a
     * skip, minimap roads gone afterwards). */
    if (s_v1049_gate == 2) { extern volatile int g_gf_ffwd; return g_gf_ffwd ? 1 : 0; }
    return s_v1049_gate;
}
static void v1049_report(void)
{
    static u64 t0 = 0; const u64 now = get_timestamp_ns();
    if (!t0) { t0 = now; return; }
    if (now - t0 < 10000000000ull) return;
    t0 = now;
    unsigned long n[6]; for (int i = 0; i < 6; i++) { n[i] = s_v1049_n[i]; s_v1049_n[i] = 0; }
    fprintf(stderr, "[V1049G] 10 s: flips completed by the walker %lu, by a tick after the walk %lu | ticks held %lu | "
            "released without the walk: FIFO drained %lu, no note %lu, timeout %lu%c", n[0], n[1], n[2], n[3], n[4], n[5], 10);
}
void cellGcmTickFlip(void)
{
    if (s_flip_status != CELL_GCM_FLIP_STATUS_WAITING) return;
    if (v1049_gate_on()) {
        const u32 cur = s_flip_request_count;
        if (s_v1049_noted_req != cur) s_v1049_n[4]++;
        else if ((int)(s_v1049_walked_req + s_v1049_lag_now - cur) >= 0) s_v1049_n[1]++;
        else {
            const u32 put = vm_read32(GCM_CONTROL_GUEST_ADDR + 0);
            if (put == s_fifo_getoff_pub()) s_v1049_n[3]++;
            else if ((get_timestamp_ns() - s_last_flip_time) / 1000000ull < (u64)s_v1049_gate_ms) { s_v1049_n[2]++; v1049_report(); return; }
            else s_v1049_n[5]++;
        }
        v1049_report();
    }
    GCM_PENDING_SET(2);
}

/* Drain the game's GCM FIFO into the RSX backend. Called from the present thread
 * (boot_main vblank_ticker). Not yet active: s_control.put stays 0 because
 * cellGcmGetControlRegister returns a HOST pointer, so the title's put writes
 * don't reach this struct (the control register must live in guest memory for
 * the recompiled code to update it via vm_write). Once that's fixed (+ the title
 * runs past its early self-exit to actually draw), parse get..put here with
 * rsx_process_command_buffer so the game's clears/draws render. */
/* Translate an RSX IO offset to a guest EA via the offset table (1MB pages,
 * populated by cellGcmInit / cellGcmMapMainMemory / MapEaIoAddress). */
static u32 gcm_io2ea(u32 io)
{
    u32 page = io >> 20;
    if (page >= 65536) return 0;
    u16 ea_page = s_ea_address_table[page];
    if (ea_page == 0xFFFF) return 0;
    return ((u32)ea_page << 20) | (io & 0xFFFFFu);
}

/* RSX get pointer (IO offset) + one-deep CALL return slot. The FIFO-wrap
 * recycle path teleports these when it resets a ring without a JUMP command. */
unsigned long long ps3_ms_now(void)
{
#ifdef _WIN32
    return (unsigned long long)GetTickCount64();
#else
    /* Was 0: every Linux [flip] timestamp read "at 0 ms". */
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000ull + (unsigned long long)(ts.tv_nsec / 1000000);
#endif
}

/* v103A [VRAM-DMA]: SPU DMA that touches RSX local memory (EA 0xC0000000-0xCFFFFFFF), bucketed per 1 MB of
 * local offset, reported every 10 s (first 60 reports, then one per minute). The D3D12 renderer keeps render
 * targets GPU-only: an SPU GET from a render target's offset reads stale guest bytes, an SPU PUT into a texture
 * the renderer cached is only seen if its checksum changes (playbook fact 63). Called from the SPU DMA path
 * (proposal _xfer/proposals/v103_A_spu_dma.diff); VRAM_DMA_LOG=0 turns the report off. inFamous' surfaces:
 * 0x1010000 zeta, 0x13B0000/0x1734000 display, 0x1AB8000/0x1E3C000/0x1F22000/0x2008000/0x2080000/0x21C0000
 * colour targets, 0xF340000 shadow depth. */
static volatile long s_vdma_n[2][256], s_vdma_kb[2][256];
static volatile long long s_vdma_t0 = 0;
void gcm_note_spu_vram_dma(int is_put, unsigned ea, unsigned size, int image_id)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("VRAM_DMA_LOG"); on = (e && *e == '0') ? 0 : 1; }
    if (!on) return;
    unsigned off = ea - 0xC0000000u;
    if (off >= 0x10000000u) return;
    unsigned b = off >> 20, d = is_put ? 1u : 0u;
    __sync_fetch_and_add(&s_vdma_n[d][b], 1);
    __sync_fetch_and_add(&s_vdma_kb[d][b], (long)((size + 1023u) >> 10));
    long long now = (long long)ps3_ms_now(), t0 = s_vdma_t0;
    if (!t0) { __sync_bool_compare_and_swap(&s_vdma_t0, 0, now); return; }
    if (now - t0 < 10000 || !__sync_bool_compare_and_swap(&s_vdma_t0, t0, now)) return;
    static int nrep = 0, skip = 0;
    if (nrep >= 60 && (++skip % 6) != 0) { memset((void*)s_vdma_n, 0, sizeof s_vdma_n); memset((void*)s_vdma_kb, 0, sizeof s_vdma_kb); return; }
    nrep++;
    for (int k = 0; k < 2; k++) {
        long tn = 0, tkb = 0; char buf[640]; int ln = 0;
        for (int i = 0; i < 256; i++) {
            long n = __sync_lock_test_and_set(&s_vdma_n[k][i], 0), kb = __sync_lock_test_and_set(&s_vdma_kb[k][i], 0);
            if (!n) continue;
            tn += n; tkb += kb;
            if (ln < 560) ln += snprintf(buf + ln, sizeof buf - ln, " 0x%03X00000:%ldx/%ldKB", (unsigned)i, n, kb);
        }
        if (tn) fprintf(stderr, "[VRAM-DMA] last %.1f s: SPU %s %ld transfers / %ld KB in RSX local memory (img %d last); per 1 MB of offset:%s%c",
                        (now - t0) / 1000.0, k ? "PUT" : "GET", tn, tkb, image_id, ln ? buf : "", 10);
    }
}

static u32 s_sema_offset = 0;   /* NV406E semaphore offset (label window) */
/* Backlog past which the drain stops honouring one-flip-per-tick and
 * catches up instead. Sized well above a frame's command list so a
 * title that keeps up never sees this path. */
#define GCM_FIFO_CATCHUP_BYTES 0x10000u
/* How close `current` must get to `end` before the runtime will recycle a ring
 * on the title's behalf. One gcmReserve is 8 bytes; a page of slack keeps a
 * title that is merely near the end from being touched. */
#define GCM_RECYCLE_SLACK 0x1000u
/* How far past the ring's start the walker must have read before the runtime
 * will recycle on the title's behalf. The guest resumes writing at `begin`, so
 * this is the amount of already-consumed head room it gets. */
#define GCM_RECYCLE_MARGIN 0x20000u
static u32 gcm_ea2io(u32 ea);   /* defined with the wrap callback below */
static u32 s_fifo_getoff  = 0;
static u32 s_fifo_getoff_pub(void) { return s_fifo_getoff; }   /* v1049G */
static u32 s_last_ring_off = 0;   /* last offset the walker actually consumed */
static u32 s_fifo_calloff = 0;
/* v1037D GCM_DEFAULT_CTX_FOLLOW (default 0; Godfather's play.cmd = 1). cellGcmSetDefaultCommandBuffer with
 * GCM_DEFAULT_CTX_REPOINT rewinds the default context's `current` to `begin`, so the title's next commands land
 * at begin -- but nothing moved the walker there. It kept walking forward from the old `put`, through the stale
 * tail of the ring and on into mapped main memory, decoding data as methods (Godfather frontend: texture units
 * bound to 0x4011xxxx pointers, depth range [0,0], garbage shaders/surfaces; up to 2.2 MB walked before an
 * unmapped CALL stopped it). Whenever that garbage walk ran long the frontend stayed black for the whole run
 * (every black verify run had such [TEXOFF]/[TEXFILT] garbage lines, no good run had any). With FOLLOW the
 * walker finishes the stream submitted before the switch (up to the put seen at the switch) and then
 * continues where the title writes next, which is what the RSX sees on hardware. Log "[V1037D]". */
static volatile u32 s_v1037_follow  = 0;   /* 1 = a default-buffer switch waits for the walker */
static volatile u32 s_v1037_old_put = 0;   /* guest put when the title switched */
static volatile u32 s_v1037_new_io  = 0;   /* IO offset of the default context's current after it */
static int v1037_follow_on(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("GCM_DEFAULT_CTX_FOLLOW"); v = (e && *e == '1') ? 1 : 0;
        fprintf(stderr, "[V1037D] walker follows cellGcmSetDefaultCommandBuffer to the default context's current (GCM_DEFAULT_CTX_FOLLOW): %s%c",
                v ? "ON" : "off", 10); }
    return v;
}

/* ---------------------------------------------------------------------------
 * 2D transfer engines (FIFO subchannels != 0).
 *
 * libgcm binds: sub 1/6 = NV0039 (m2mf buffer copy), sub 2 = NV3062 (context
 * surface 2D), sub 3 = NV309E (swizzled surface), sub 4 = NV308A (image from
 * CPU / inline transfer), sub 5 = NV3089 (scaled image).
 *
 * cellGcmSetFragmentProgramParameter patches a fragment program's inline
 * constants by emitting NV3062 (destination surface) + NV308A (COLOR data
 * words) through the FIFO -- it does NOT CPU-write the ucode copy. Without
 * this engine the patched constants stay zero in local memory (demosaic: all
 * its texture-size constants -> UVs scaled by 0 -> uniform output).
 * -----------------------------------------------------------------------*/
extern u32 cellGcmResolveLocated(int local, u32 offset);

static struct {
    u32 dst_dma;      /* NV3062 0x0188 SET_CONTEXT_DMA_IMAGE_DESTIN */
    u32 color_fmt;    /* NV3062 0x0300 SET_COLOR_FORMAT             */
    u32 pitch;        /* NV3062 0x0304 SET_PITCH (dst in bits 16-31) */
    u32 dst_offset;   /* NV3062 0x030C SET_OFFSET_DESTIN            */
    u32 point;        /* NV308A 0x0304 SET_POINT ((y<<16)|x)        */
    u32 size_out;     /* NV308A 0x0308 SET_SIZE_OUT                 */
} s_gcm2d;

static struct {
    u32 src_dma, dst_surf, fmt, op, clip_pt, clip_sz, out_pt, out_sz;
    u32 ds_dx, dt_dy, in_sz, in_fmt, in_off, in_uv;
} s_nv3089;

/* NV309E "swizzled surface" -- the destination object for a scaled-image blit
 * whose target is a swizzled (Morton-ordered) texture rather than a linear
 * surface. Rubber Ducky binds it to the same subchannel this file previously
 * assumed was always NV308A (image-from-CPU), so its SET_OFFSET was being read
 * as an NV308A "point" and the blit destination was never picked up at all.
 * The two are told apart by method 0x0300: NV308A has no such method. */
static struct {
    u32 fmt;        /* 0x0300: (log2h << 24) | (log2w << 16) | format */
    u32 offset;     /* 0x0304: destination offset                     */
    int active;
    int subch;      /* which subchannel NV309E is bound to. Per-subchannel:
                     * this title has NV309E on 4 and NV308A on 5, and a global
                     * flag made 5's NV308A POINT (also 0x0304) be swallowed as
                     * an NV309E offset, breaking every inline transfer. */
} s_nv309e;

/* Execute one NV3089 blit into the currently-bound NV3062 destination surface.
 * Point-sampled: ds/dx and dt/dy are 20.12 fixed point and are 0x100000 (1.0)
 * for the straight uploads this exists for; a scaled blit still lands, just
 * without filtering. */
/* Morton/Z-order offset for an NV309E swizzled surface: interleave the low
 * min(l2w,l2h) bits of x and y, then append the remaining bits of the larger
 * dimension above them. */
static u32 nv309e_swz(u32 x, u32 y, u32 l2w, u32 l2h)
{
    u32 n = l2w < l2h ? l2w : l2h, off = 0;
    for (u32 i = 0; i < n; i++)
        off |= (((x >> i) & 1u) << (2 * i)) | (((y >> i) & 1u) << (2 * i + 1));
    if (l2w > l2h)      off |= (x >> n) << (2 * n);
    else if (l2h > l2w) off |= (y >> n) << (2 * n);
    return off;
}

/* v1043Z: set by the D3D12 backend; records a GPU copy for a blit out of a GPU-rendered surface (raw local offsets). */
void (*volatile g_rsx_blit_hook)(u32 src, u32 src_pitch, u32 dst, u32 dst_pitch, u32 w, u32 h) = 0;
static void nv3089_blit(void)
{
    u32 out_w = s_nv3089.out_sz & 0xFFFF, out_h = s_nv3089.out_sz >> 16;
    u32 out_x = s_nv3089.out_pt & 0xFFFF, out_y = s_nv3089.out_pt >> 16;
    u32 in_pitch = s_nv3089.in_fmt & 0xFFFF;
    u32 dst_pitch = s_gcm2d.pitch >> 16;
    if (!out_w || !out_h || !in_pitch || !dst_pitch) return;
    /* NV3089_GATE=<file>: log every blit's geometry while <file> exists, so a
     * capture covers the scene of interest instead of the first N since boot. */
    { static const char* gate = (const char*)-1; static unsigned poll; static int open_gate;
      if (gate == (const char*)-1) gate = getenv("NV3089_GATE");
      if (gate) {
          if ((poll++ & 63u) == 0) { FILE* g = fopen(gate, "rb"); open_gate = g != NULL; if (g) fclose(g); }
          if (open_gate) {
              int _sl = (s_nv3089.src_dma != 0xFEED0001u), _dl = (s_gcm2d.dst_dma != 0xFEED0001u);
              printf("[NV3089G] fmt=0x%X out=%ux%u at %u,%u in=%ux%u uv=0x%08X ds=0x%X dt=0x%X src=0x%08X/%u dst=0x%08X/%u swz=%d%c",
                     s_nv3089.fmt & 0xFF, out_w, out_h, out_x, out_y,
                     s_nv3089.in_sz & 0xFFFF, s_nv3089.in_sz >> 16, s_nv3089.in_uv,
                     s_nv3089.ds_dx, s_nv3089.dt_dy,
                     cellGcmResolveLocated(_sl, s_nv3089.in_off), in_pitch,
                     cellGcmResolveLocated(_dl, s_gcm2d.dst_offset), dst_pitch,
                     s_nv309e.active, 10);
          }
      } }
    /* v1043Z NV3089_LOGNEW (default 1; user video: a grapple slam shows an old street frame behind the live HUD):
     * log each distinct blit (src, dst, size, scale, fmt) once with a timestamp, so a replay shows which copy feeds
     * the full-screen effect. Max 300 lines. */
    { static int ln = -1; static u32 seen[300][5]; static int ns = 0;
      if (ln < 0) { const char* e = getenv("NV3089_LOGNEW"); ln = (e && *e == '0') ? 0 : 1; }
      if (ln && ns < 300) {
          const int _sl = (s_nv3089.src_dma != 0xFEED0001u), _dl = (s_gcm2d.dst_dma != 0xFEED0001u);
          const u32 k[5] = { cellGcmResolveLocated(_sl, s_nv3089.in_off), cellGcmResolveLocated(_dl, s_gcm2d.dst_offset),
                             s_nv3089.out_sz, s_nv3089.ds_dx ^ (s_nv3089.dt_dy << 1), (s_nv3089.fmt & 0xFF) | ((u32)s_nv309e.active << 8) };
          int known = 0;
          for (int i = 0; i < ns && !known; i++) known = !memcmp(seen[i], k, sizeof k);
          if (!known) {
              memcpy(seen[ns++], k, sizeof k);
              printf("[NV3089N] t=%llu ms fmt=0x%X out=%ux%u at %u,%u in=%ux%u uv=0x%08X ds=0x%X dt=0x%X src=0x%08X/%u dst=0x%08X/%u swz=%d%c",
                     (unsigned long long)GetTickCount64(), s_nv3089.fmt & 0xFF, out_w, out_h, out_x, out_y,
                     s_nv3089.in_sz & 0xFFFF, s_nv3089.in_sz >> 16, s_nv3089.in_uv, s_nv3089.ds_dx, s_nv3089.dt_dy,
                     k[0], in_pitch, k[1], dst_pitch, s_nv309e.active, 10);
              fflush(stdout);
          }
      } }
    /* v1043Z: 1:1 32-bit local -> local blit, not into a swizzled texture -> mirror it on the GPU (D3D12 backend) */
    if (g_rsx_blit_hook && s_nv3089.ds_dx == 0x100000u && s_nv3089.dt_dy == 0x100000u && !s_nv309e.active &&
        s_nv3089.src_dma != 0xFEED0001u && s_gcm2d.dst_dma != 0xFEED0001u) {
        const u32 _f = s_nv3089.fmt & 0xFF;
        if (_f == 3 || _f == 7 || _f == 8 || _f == 0xD) {
            const u32 u0 = (s_nv3089.in_uv & 0xFFFF) >> 4, v0 = (s_nv3089.in_uv >> 16) >> 4;
            g_rsx_blit_hook(s_nv3089.in_off + v0 * in_pitch + u0 * 4u, in_pitch,
                            s_gcm2d.dst_offset + out_y * dst_pitch + out_x * 4u, dst_pitch, out_w, out_h);
        }
    }
    /* A copy out of a surface the live renderer drew exists only on the GPU;
     * mirror 1:1 blits there too (the guest-memory copy below still runs). */
    if (s_nv3089.ds_dx == 0x100000u && s_nv3089.dt_dy == 0x100000u &&
        rsx_live_draw_enabled()) {
        const u32 u0 = (s_nv3089.in_uv & 0xFFFF) >> 4, v0 = (s_nv3089.in_uv >> 16) >> 4;
        const int sl = (s_nv3089.src_dma != 0xFEED0001u);
        const int dl = (s_gcm2d.dst_dma  != 0xFEED0001u);
        rsx_live_draw_blit(sl ? 0u : 1u, s_nv3089.in_off + v0 * in_pitch + u0 * 4u, in_pitch,
                           dl ? 0u : 1u, s_gcm2d.dst_offset + (s_nv3089.out_pt >> 16) * dst_pitch
                                         + (s_nv3089.out_pt & 0xFFFF) * 4u, dst_pitch,
                           out_w, out_h);
    } else if (rsx_live_draw_enabled()) {
        /* Scaled: a resolve. Copy it on the GPU into the display buffer; fall
         * back to presenting the source when that cannot be done. */
        const u32 sl = s_nv3089.src_dma != 0xFEED0001u ? 0u : 1u;
        const u32 dl = s_gcm2d.dst_dma  != 0xFEED0001u ? 0u : 1u;
        const u32 u0 = (s_nv3089.in_uv & 0xFFFF) >> 4, v0 = (s_nv3089.in_uv >> 16) >> 4;
        const int rok = rsx_live_draw_resolve_blit(sl, s_nv3089.in_off + v0 * in_pitch + u0 * 4u, in_pitch,
                                        s_nv3089.ds_dx, s_nv3089.dt_dy, dl, s_gcm2d.dst_offset,
                                        dst_pitch, out_x, out_y, out_w, out_h);
        if (!rok && s_nv3089.in_uv <= 0x00100010u && !s_nv3089.out_pt)
            rsx_live_draw_note_resolve(sl, s_nv3089.in_off, dl, s_gcm2d.dst_offset);
    }
    /* Only the 32-bit colour formats are handled; anything else would need a
     * per-format converter and is better skipped loudly than written wrong. */
    u32 f = s_nv3089.fmt & 0xFF;
    /* NOTE: format 0x3 appears here as a downsample chain (1024x512 -> 1024x256
     * -> 512x128 -> 256x64 -> 128x32, all into one destination) that builds a
     * mip/blur pyramid, and skipping it leaves that texture empty. Treating it
     * as 4-byte A8R8G8B8 like 7/8/0xD is WRONG -- it floods the scene blue
     * (flat-blue coverage 17k -> 537k pixels), so its pixel layout differs.
     * Decode it properly before enabling; do not just add it to this list. */
    /* NV309E swizzled destination. Colour format 0x3 is A8R8G8B8 in the
     * scaled-image enum; this title uses it to build a mip/blur pyramid into a
     * swizzled texture (1024x512 -> 1024x256 -> ... all through NV309E).
     * Writing it LINEARLY into the NV3062 destination is what flooded the scene
     * blue -- wrong destination and wrong layout. */
    { u32 l2w = (s_nv309e.fmt >> 16) & 0xFF, l2h = (s_nv309e.fmt >> 24) & 0xFF;
      static int en = -1;
      /* On by default: this is the mip/blur pyramid. Without it textures have
       * no mip levels, which shows up as aliased blocky tiles and a noise patch
       * where a minified surface should be, and the shell that samples the
       * pyramid draws untextured. NV309E_BLIT=0 disables. */
      if (en < 0) { const char* e = getenv("NV309E_BLIT"); en = e ? atoi(e) : 1; }
      if (en && f == 3 && s_nv309e.active && s_nv309e.offset &&
          (1u << l2w) == out_w && (1u << l2h) == out_h) {
        int sl = (s_nv3089.src_dma != 0xFEED0001u);
        u32 src2 = cellGcmResolveLocated(sl, s_nv3089.in_off);
        u32 dst2 = cellGcmResolveLocated(1, s_nv309e.offset);
        u32 u0b = s_nv3089.in_uv & 0xFFFF, v0b = s_nv3089.in_uv >> 16;
        /* Copy words verbatim through raw pointers rather than vm_read32/
         * vm_write32: this moves ~500k pixels per blit and several blits run per
         * frame, and the two byte swaps would cancel anyway. */
        { extern u8* vm_base;
          const u8* sbase = vm_base + src2;
          u8* dbase = vm_base + dst2;
          for (u32 y = 0; y < out_h; y++) {
              u64 sv = ((u64)v0b << 8) + (u64)y * s_nv3089.dt_dy;
              const u8* srow = sbase + (u32)(sv >> 20) * in_pitch;
              for (u32 x = 0; x < out_w; x++) {
                  u64 su = ((u64)u0b << 8) + (u64)x * s_nv3089.ds_dx;
                  memcpy(dbase + (u64)nv309e_swz(x, y, l2w, l2h) * 4,
                         srow + (u32)(su >> 20) * 4, 4);
              }
          } }
        { static int _n = 0; static int cap = -1;
          if (cap < 0) { const char* e = getenv("NV3089_DBG"); cap = e ? atoi(e) : 0; }
          if (cap && _n < cap) { _n++;
            printf("[NV3089] swizzled %ux%u fmt=0x%X src=0x%08X -> NV309E dst=0x%08X%c",
                   out_w, out_h, f, src2, dst2, 10); } }
        return;
      } }
    /* 0x3 is A8R8G8B8 in the scaled-image format enum. This title uses it for
     * two different things: the swizzled mip/blur pyramid handled above, and
     * (with a LINEAR NV3062 destination) copies of the rendered scene surface
     * into the textures its water and effects sample -- e.g.
     * src=0xCE340000 (the scene) -> dst=0xC3746200, which is the single most
     * sampled blended texture in the frame. Skipping those left every one of
     * those textures empty. Same 4-byte pixel layout as the formats already
     * handled. */
    if (f != 3 /* A8R8G8B8 (scale-format enum) */ &&
        f != 7 /* A8R8G8B8 */ && f != 8 /* X8R8G8B8 */ && f != 0xD /* A8B8G8R8 */) {
        /* Log the GEOMETRY of a skipped blit, not just the format: a full-screen
         * copy from the scene surface to a registered display buffer is the
         * composite this title needs, and is worth telling apart from some
         * incidental small transfer. */
        { int _sl = (s_nv3089.src_dma != 0xFEED0001u);
          int _dl = (s_gcm2d.dst_dma  != 0xFEED0001u);
          printf("[NV3089] SKIPPED fmt=0x%X  %ux%u  src=0x%08X(pitch %u,%s) -> dst=0x%08X(pitch %u,%s) at %u,%u%c",
                 f, out_w, out_h,
                 cellGcmResolveLocated(_sl, s_nv3089.in_off), in_pitch, _sl?"LOCAL":"MAIN",
                 cellGcmResolveLocated(_dl, s_gcm2d.dst_offset), dst_pitch, _dl?"LOCAL":"MAIN",
                 out_x, out_y, 10); }
        static int _w = 0;
        if ((_w < 4 && _w++ < 4)) printf("[NV3089] unsupported colour format 0x%X, blit skipped\n", f);
        return;
    }
    int src_local = (s_nv3089.src_dma != 0xFEED0001u);
    int dst_local = (s_gcm2d.dst_dma  != 0xFEED0001u);
    u32 src = cellGcmResolveLocated(src_local, s_nv3089.in_off);
    u32 dst = cellGcmResolveLocated(dst_local, s_gcm2d.dst_offset);
    u32 u0 = s_nv3089.in_uv & 0xFFFF, v0 = s_nv3089.in_uv >> 16;   /* 12.4 start */
    /* Clip to SET_CLIP_POINT/SIZE (destination coordinates). A title tiling a
     * resolve in 512x512 blocks relies on it to stop at the surface edge:
     * Tornado Outbreak's bottom band is 512 rows tall with 208 left in the
     * display buffer, and writing all 512 ran on into the VRAM after it --
     * where its full-screen quad's vertices were staged, so the quad drew with
     * all-zero positions. A zero clip size means unclipped. */
    u32 cx0 = s_nv3089.clip_pt & 0xFFFF, cy0 = s_nv3089.clip_pt >> 16;
    u32 cx1 = (s_nv3089.clip_sz & 0xFFFF) ? cx0 + (s_nv3089.clip_sz & 0xFFFF) : 0xFFFFFFFFu;
    u32 cy1 = (s_nv3089.clip_sz >> 16)    ? cy0 + (s_nv3089.clip_sz >> 16)    : 0xFFFFFFFFu;
    /* Bytes per pixel from the SDK enums (CELL_GCM_TRANSFER_SCALE_FORMAT_*:
     * 7 = R5G6B5, 8 = Y8 -- not 32-bit as they were once labelled). When the
     * destination surface format agrees it is a plain copy of that width; a
     * real format conversion keeps the old 4-byte copy rather than guess. */
    u32 sbpp = (f == 7) ? 2u : (f == 8) ? 1u : 4u;
    u32 dbpp = (s_gcm2d.color_fmt == 4u /* R5G6B5 */) ? 2u
             : (s_gcm2d.color_fmt == 1u /* Y8 */)     ? 1u : 4u;
    u32 bpp = (sbpp == dbpp) ? sbpp : 4u;
    /* GF_BLIT_FAST=1 (code default 0 = old per-pixel vm_read/vm_write): the rows of this scaled copy are
     * plain byte moves (a read-swap followed by a write-swap of the same width cancels out), but every pixel
     * paid a coherence call + a swap pair + a reservation test; the profile has this loop at ~11 % of the
     * FIFO thread in rain / fire / explosion scenes. Per row: when no 128-byte line of the destination span
     * is SPU-reserved and the row is outside the write-watch window, copy through host pointers (memmove for
     * 1:1 rows, raw element copies for scaled ones) and break PPU reservations once per line. Otherwise that
     * row takes the old path, so the guest-visible result is the same. */
    static int s_blit_fast = -1;
    if (s_blit_fast < 0) { const char* e = getenv("GF_BLIT_FAST"); s_blit_fast = (e && *e == '1') ? 1 : 0;
                           if (s_blit_fast) fprintf(stderr, "[GF_BLIT_FAST] NV3089 scaled blit: host-pointer rows ON\n"); }
    for (u32 y = 0; y < out_h; y++) {
        u32 dy = out_y + y;
        if (dy < cy0 || dy >= cy1) continue;
        u64 sv = ((u64)v0 << 8) + (u64)y * s_nv3089.dt_dy;         /* 20.12 */
        u32 sy = (u32)(sv >> 20);
        if (s_blit_fast) {
            u32 xa = 0, xb = out_w;                                   /* clipped x range of this row */
            if (out_x < cx0) xa = cx0 - out_x;
            if (cx1 != 0xFFFFFFFFu) { u32 lim = (cx1 > out_x) ? cx1 - out_x : 0; if (lim < xb) xb = lim; }
            if (xa >= xb) continue;
            const u32 d0 = dst + dy * dst_pitch + (out_x + xa) * bpp;
            const u32 d1 = d0 + (xb - xa) * bpp;                      /* exclusive */
            int ok = !(d0 < g_ww_hi && d1 > g_ww_lo);
            for (u32 l = d0 & ~127u; ok && l < d1; l += 128u) if (spu_coh_is_reserved(l)) ok = 0;
            if (ok) {
                uint8_t* hd = vm_ptr8(d0);
                const u32 srow = src + sy * in_pitch;
                if (s_nv3089.ds_dx == (1u << 20)) {
                    const u32 sx0 = (u32)((((u64)u0 << 8) + (u64)xa * s_nv3089.ds_dx) >> 20);
                    memmove(hd, vm_ptr8(srow + sx0 * bpp), (size_t)(xb - xa) * bpp);
                } else {
                    for (u32 x = xa; x < xb; x++, hd += bpp) {
                        const u32 sx = (u32)((((u64)u0 << 8) + (u64)x * s_nv3089.ds_dx) >> 20);
                        const uint8_t* hs = vm_ptr8(srow + sx * bpp);
                        if (bpp == 4) memcpy(hd, hs, 4); else if (bpp == 2) memcpy(hd, hs, 2); else *hd = *hs;
                    }
                }
                if (g_resv_store_active > 0)
                    for (u32 l = d0 & ~127u; l < d1; l += 128u) ppu_resv_break_store(l < d0 ? d0 : l);
                continue;
            }
        }
        for (u32 x = 0; x < out_w; x++) {
            u32 dx = out_x + x;
            if (dx < cx0 || dx >= cx1) continue;
            u64 su = ((u64)u0 << 8) + (u64)x * s_nv3089.ds_dx;
            u32 sx = (u32)(su >> 20);
            u32 s = src + sy * in_pitch + sx * bpp;
            u32 d = dst + dy * dst_pitch + dx * bpp;
            if (bpp == 4)      vm_write32(d, vm_read32(s));
            else if (bpp == 2) vm_write16(d, vm_read16(s));
            else               vm_write8(d, vm_read8(s));
        }
    }
    { static int _n = 0; static int cap = -1;
      if (cap < 0) { const char* e = getenv("NV3089_DBG"); cap = e ? atoi(e) : 0; }
      if (cap && _n < cap) { _n++;
        printf("[NV3089] %ux%u fmt=0x%X src=0x%08X(pitch %u) -> dst=0x%08X(pitch %u) at %u,%u\n",
               out_w, out_h, f, src, in_pitch, dst, dst_pitch, out_x, out_y); } }
}

/* v1033M NV2D_GUARD (default 1; 0 = old): the NV309E latch below decided "this subchannel is NV309E" from
 * ONE method-0x0300 write and kept it forever, after which every 0x0304 on that subchannel was taken as an
 * NV309E offset. Godfather's 2026-10-30 21:39 run latched it from FIFO-desync garbage (pointer table words
 * parsed as headers: "[NV309E] format=0x40108308", log2 height 0x40), so every later NV308A POINT of the
 * inline transfers that patch fragment-program constants was swallowed. The constant words then landed at a
 * stale point: on top of instruction slots (the frontend UI FP's ADD slot read k=1,0; FP 0x16E01 rebuilt 29
 * pipelines during one fade) and never at the constant the fade/dim layer reads -> black frontend.
 * Now: (a) 0x0300 latches NV309E only for a plausible SET_FORMAT (colour format 1..0xB, log2 w/h <= 12);
 * (b) a latched 0x0304 also updates the NV308A point (harmless for a real NV309E: inline transfers always
 * set POINT before COLOR); (c) an NV308A-only method (SIZE_OUT/SIZE_IN/COLOR) on the latched subchannel
 * releases the latch. Log: "[NV2D] v1033M". */
static int nv2d_guard_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("NV2D_GUARD"); on = (e && *e == '0') ? 0 : 1;
                  printf("[NV2D] v1033M NV309E/NV308A subchannel guard %s (NV2D_GUARD)%c", on ? "ON" : "off", 10); }
    return on;
}

static void gcm_2d_method(u32 subch, u32 method, u32 data)
{
    { extern void (*g_rsx_vtx_sync)(void); if (g_rsx_vtx_sync) g_rsx_vtx_sync(); }   /* v33 */
    /* GCM2D_TRACE=1: histogram of (subchannel, method) actually seen. The
     * subchannel a title binds each 2D object to is libgcm-version-specific and
     * SET_OBJECT binds are not tracked, so state landing on an unexpected
     * subchannel is silently dropped -- which looks like a blit using stale
     * destination state rather than like missing state. */
    { static int tr = -1;
      if (tr < 0) { const char* e = getenv("GCM2D_TRACE"); tr = e ? atoi(e) : 0; }
      if (tr) {
          static u8 seen[8][1024]; static u64 n = 0;
          u32 mi = (method >> 2) & 1023;
          if (subch < 8 && !seen[subch][mi]) {
              seen[subch][mi] = 1;
              printf("[GCM2D-TRACE] subch=%u method=0x%04X (first, data=0x%08X)%c",
                     subch, method, data, 10);
          }
          if (++n == 400000ull) {
              printf("[GCM2D-TRACE] --- subchannels used: ");
              for (u32 sc = 0; sc < 8; sc++) {
                  u32 c = 0; for (u32 m = 0; m < 1024; m++) c += seen[sc][m];
                  if (c) printf("%u(%u methods) ", sc, c);
              }
              printf("---%c", 10);
          }
      } }
    /* Subchannel bindings are libgcm-version-specific (SET_OBJECT binds are
     * not tracked): demosaic's SDK emits dest-surface on sub 3 and image-from-
     * CPU on sub 5 (cellGcmSetInlineTransfer: 0x4630C / 0xCA304 / 0xA400
     * headers); other versions use the classic 2 / 4. Accept both. */
    if (subch == 2 || subch == 3) {         /* NV3062 context surface 2D */
        switch (method) {
        case 0x0184: case 0x0188: s_gcm2d.dst_dma = data; return;
        case 0x0300: s_gcm2d.color_fmt  = data; return;
        case 0x0304: s_gcm2d.pitch      = data; return;
        case 0x030C: s_gcm2d.dst_offset = data; return;
        }
        return;
    }
    if (subch == 4 || subch == 5) {         /* NV308A image-from-CPU, or NV309E */
        /* NV309E swizzled-surface state. SET_FORMAT (0x0300) does not exist on
         * NV308A, so seeing it identifies the object bound here. */
        if (method == 0x0300 && nv2d_guard_on()) {   /* v1033M (a): only a plausible NV309E SET_FORMAT latches */
            const u32 l2w = (data >> 16) & 0xFFu, l2h = (data >> 24) & 0xFFu, cf = data & 0xFFFFu;
            if (!(cf >= 1u && cf <= 0xBu && l2w <= 12u && l2h <= 12u)) {
                static int _n = 0; if ((_n < 8 && _n++ < 8))
                    printf("[NV2D] v1033M ignored implausible NV309E SET_FORMAT 0x%08X on subch %u (not latched)%c",
                           data, subch, 10);
                return; } }
        if (method == 0x0300) { s_nv309e.fmt = data; s_nv309e.active = 1;
                                s_nv309e.subch = (int)subch;
            { static int _n = 0; if ((_n < 4 && _n++ < 4))
                printf("[NV309E] format=0x%08X (log2 %ux%u fmt=0x%X)%c", data,
                       1u << ((data >> 16) & 0xFF), 1u << ((data >> 24) & 0xFF),
                       data & 0xFFFF, 10); }
            return; }
        if (method == 0x0304 && s_nv309e.active && s_nv309e.subch == (int)subch) {
            s_nv309e.offset = data;
            if (nv2d_guard_on()) s_gcm2d.point = data;   /* v1033M (b): may be an NV308A POINT */
            return; }
        if (s_nv309e.active && s_nv309e.subch == (int)subch && nv2d_guard_on() &&
            (method == 0x0308 || method == 0x030C || (method >= 0x0400 && method <= 0x07FC))) {
            s_nv309e.active = 0;   /* v1033M (c): NV308A-only method -> this subchannel is NV308A */
            static int _n = 0; if ((_n < 8 && _n++ < 8))
                printf("[NV2D] v1033M subch %u carries NV308A method 0x%04X -> NV309E latch released%c", subch, method, 10);
        }
        switch (method) {
        case 0x0304: s_gcm2d.point    = data; return;
        case 0x0308: s_gcm2d.size_out = data; return;
        case 0x030C: return;                /* SIZE_IN */
        }
        if (method >= 0x0400 && method <= 0x07FC) {
            /* COLOR data word: write into the destination surface. Inline
             * transfers use format Y32 (4 bytes/px, one row per point.y).
             * DMA 0xFEED0000 = local memory, 0xFEED0001 = main memory. */
            u32 idx = (method - 0x0400) >> 2;
            u32 px  = (s_gcm2d.point & 0xFFFF) + idx;
            u32 py  = s_gcm2d.point >> 16;
            u32 dst_pitch = s_gcm2d.pitch >> 16;
            int local = (s_gcm2d.dst_dma != 0xFEED0001u);
            u32 base = cellGcmResolveLocated(local, s_gcm2d.dst_offset);
            { static int _it = 0;
              if ((_it < 6 && _it++ < 6))
                  printf("[GCM2D] inline write dst=0x%08X (off=0x%X dma=0x%X pt=%u,%u pitch=%u) = 0x%08X\n",
                         base + py * dst_pitch + px * 4, s_gcm2d.dst_offset,
                         s_gcm2d.dst_dma, px, py, dst_pitch, data); }
            /* GCM2D_ALL=1: trace later fragment-constant uploads too.  The
             * normal six-line bring-up log above stops before titles begin
             * swapping their frontend shader variants, which is exactly when
             * a bad POINT/OFFSET pairing can overwrite an instruction slot. */
            { static int all = -1, all_n = 0;
              if (all < 0) all = getenv("GCM2D_ALL") ? 1 : 0;
              if (all && all_n++ < 2048)
                  printf("[GCM2D-ALL] sub=%u dst=0x%08X base=0x%08X off=0x%X pt=%u,%u idx=%u pitch=%u data=0x%08X\n",
                         subch, base + py * dst_pitch + px * 4, base,
                         s_gcm2d.dst_offset, s_gcm2d.point & 0xFFFF,
                         s_gcm2d.point >> 16, idx, dst_pitch, data); }
            vm_write32(base + py * dst_pitch + px * 4, data);
            return;
        }
        return;
    }
    /* NV3089 "scaled image from memory" -- the blit PSGL uses to move a texture
     * it has assembled elsewhere into the VRAM the sampler reads. Without it a
     * title's textures stay whatever the destination was before (zero), so
     * geometry renders flat no matter how correct the sampler is.
     *
     * Registers: 0x0184 source DMA, 0x0198 destination surface (the NV3062
     * state above), 0x0300 colour format, 0x0304 operation, 0x0308/0x030C clip
     * point/size, 0x0310/0x0314 out point/size, 0x0318/0x031C ds/dx dt/dy in
     * 20.12 fixed point, 0x0400 in size, 0x0404 (origin<<16)|pitch, 0x0408 in
     * offset, 0x040C in u/v start -- which is also the trigger. */
    if (subch == 6 || subch == 7) {
        switch (method) {
        case 0x0184: s_nv3089.src_dma   = data; return;
        case 0x0198: s_nv3089.dst_surf  = data; return;
        case 0x0300: s_nv3089.fmt       = data; return;
        case 0x0304: s_nv3089.op        = data; return;
        case 0x0308: s_nv3089.clip_pt   = data; return;
        case 0x030C: s_nv3089.clip_sz   = data; return;
        case 0x0310: s_nv3089.out_pt    = data; return;
        case 0x0314: s_nv3089.out_sz    = data; return;
        case 0x0318: s_nv3089.ds_dx     = data; return;
        case 0x031C: s_nv3089.dt_dy     = data; return;
        case 0x0400: s_nv3089.in_sz     = data; return;
        case 0x0404: s_nv3089.in_fmt    = data; return;
        case 0x0408: s_nv3089.in_off    = data; return;
        case 0x040C: s_nv3089.in_uv     = data; nv3089_blit(); return;
        }
        return;
    }
    /* NV0039 / NV309E: not implemented yet -- log first sightings. */
    /* GCM2D_DBG=<N> raises the cap and adds the data word: the fixed 8 showed
     * only that SOMETHING was unhandled, not enough to implement it. */
    static int warned = 0, wcap = -1;
    if (wcap < 0) { const char* e = getenv("GCM2D_DBG"); wcap = e ? atoi(e) : 8; }
    if (warned < wcap) { warned++;
        printf("[GCM2D-UNH] subch %u method 0x%04X data=0x%08X\n", subch, method, data); }
    if (0)
        printf("[cellGcmSys] FIFO subch %u method 0x%04X (unhandled 2D engine)\n",
               subch, method);
}

/* Present gate for the ticker: a flip is ready once the drain has consumed
 * everything up to put. The guest blocks in its own WaitFlip right after
 * flipping, so the FIFO holds EXACTLY the completed frame -- no guest-side
 * blocking needed (the old spin serialized two vsync-class waits per frame
 * and halved the frame rate). Call AFTER draining. */
int cellGcm_take_flip_pending_synced(void)
{
    if (!s_flip_pending) return 0;
    u32 put = vm_read32(GCM_CONTROL_GUEST_ADDR + 0);
    u32 get = vm_read32(GCM_CONTROL_GUEST_ADDR + 4);
    if (get != put) return 0;   /* frame not fully drained yet */
    s_flip_pending = 0;
    return 1;
}

/* ---- fence (SET_REFERENCE) observability ---------------------------------
 * cellGcmFinish waits for ctrl->ref with EQUALITY. On hardware every fence
 * value persists in the register for the RSX-time until the next fence
 * (~a frame), so an equality poll cannot miss it. Our drain consumes a whole
 * backlog in microseconds; publishing only the final value made intermediate
 * fences invisible -- LBP's boot deadlocked with a thread spinning on a fence
 * the counter had already passed (waiting 0x13 while ref climbed 0x51->0x1F4).
 * Queue every fence as it is drained and publish ONE per tick: the GPU work
 * is never throttled (draw/recycle proceed at full speed), only the ref
 * register advances at most one fence per tick, so every value is observable
 * for >= 16 ms, like hardware. (Throttling the DRAIN at fence boundaries
 * instead starved wrap recycles and wedged the backend -- don't.) */
#define GCM_REF_QLEN 4096
static u32 s_ref_q[GCM_REF_QLEN];
static volatile u32 s_ref_qhead = 0, s_ref_qtail = 0;   /* single producer+consumer: the ticker */

/* A JUMP/CALL whose target has no IO mapping must not be taken. Taking one
 * teleports the walker into unmapped space, where it can never reach `put`
 * again: the FIFO silently stops being processed, no fences are published, and
 * a title spinning on ctrl->ref waits forever with no indication why. Tokyo
 * Jungle did exactly this -- get left the 1 MB ring for IO 0x00F00000, which it
 * never mapped, while put sat at 0x604. Stop at the bad word and say so. */
static void gcm_fifo_bad_branch(const char* kind, u32 target, u32 word)
{
    static int n = 0;
    if ((n < 8 && n++ < 8))
        fprintf(stderr, "[cellGcmSys] FIFO %s to unmapped IO 0x%08X (word 0x%08X) "
                "-- not taken, drain stops here\n", kind, target, word);
}

/* A bad branch nearly always means the walker mis-counted a method batch a few
 * dwords back and is now reading vertex data -- a float 1.0f is 0x3F800000,
 * which decodes as JUMP 0x1F800000. Dump the run-up: the command that consumed
 * the wrong count is in here. */
/* Last words the walker actually consumed, so a bad branch can show the decode
 * chain that led there rather than raw memory around it. */
#define GCM_TRACE_N 64
static u32 s_tr_off[GCM_TRACE_N], s_tr_w[GCM_TRACE_N];
static u32 s_tr_i = 0;

/* A branch we cannot take leaves the walker pointing at whatever it
 * mis-decoded, and it never recovers -- get freezes for the rest of the
 * run and nothing renders. Resynchronise to `put` instead, the same trade
 * the unmapped-get path already makes: the commands between here and put
 * are lost, but the next frame is written from a clean boundary and the
 * FIFO lives. */
static void gcm_ref_push_at(u32 v, u32 getoff);

/* ---- SPU-built pushbuffers ------------------------------------------------
 * Some titles do not build the RSX command stream on the PPU at all. The
 * Orange Box is one: its libgcm context points at a 64 KB buffer in the
 * binary's own .bss that is never mapped into IO space, because that buffer is
 * where the PPU records draw commands for an SPU to consume. CB.SPU reads it
 * and DMAs the real pushbuffer into mapped main memory, then `put` is set past
 * the end of what it wrote.
 *
 * So libgcm is never told where the stream is, and the walker -- which starts
 * at IO 0 and has no JUMP to follow, because the SPU side did the binding --
 * scans tens of megabytes of unrelated memory, NOPs through the zeros (a zero
 * word is a valid NOP) and dies on the first vertex float it meets.
 *
 * The DMA is the only evidence there is, so record it. Each PUT that lands in
 * a mapped IO range is a candidate pushbuffer; the one that ends just short of
 * `put` is the block the RSX is actually being pointed at. */
#define GCM_SPU_BLK_N 16
static struct { u32 io_start, io_end; } s_spu_blk[GCM_SPU_BLK_N];
static int s_spu_blk_w = 0;

/* EA -> IO through the mapping list (range lookup; the by-ea helper is exact). */
static u32 gcm_ea2io_range(u32 ea, u32* room)
{
    for (int i = 0; i < CELL_GCM_MAX_IO_MAPPINGS; i++) {
        IoMapping* m = &s_io_mappings[i];
        if (!m->active || !m->size) continue;
        if (ea >= m->ea && ea < m->ea + m->size) {
            if (room) *room = m->ea + m->size - ea;
            return m->io + (ea - m->ea);
        }
    }
    return 0xFFFFFFFFu;
}

void gcm_forget_spu_blocks(void)
{
    memset(s_spu_blk, 0, sizeof s_spu_blk);
    s_spu_blk_w = 0;
}

/* v220 VC_WSRC diagnostic (rsx_d3d12_backend.c): sees EVERY SPU DMA PUT, including RSX-local and non-IO memory. */
void (*g_vc_put_note)(u32 ea, u32 size) = 0;
void gcm_note_spu_put(u32 ea, u32 size)
{
    if (g_vc_put_note) g_vc_put_note(ea, size);
    if (!size || size > 0x40000u) return;
    u32 room = 0;
    u32 io = gcm_ea2io_range(ea, &room);
    if (io == 0xFFFFFFFFu || size > room) return;   /* not in RSX-visible memory */

    /* Coalesce with the previous block when an SPU emits one buffer in several
     * transfers -- otherwise only the last fragment is remembered and the
     * walker still starts in the middle of the stream. */
    int prev = (s_spu_blk_w + GCM_SPU_BLK_N - 1) % GCM_SPU_BLK_N;
    if (s_spu_blk[prev].io_end == io) { s_spu_blk[prev].io_end = io + size; return; }

    s_spu_blk[s_spu_blk_w].io_start = io;
    s_spu_blk[s_spu_blk_w].io_end   = io + size;
    s_spu_blk_w = (s_spu_blk_w + 1) % GCM_SPU_BLK_N;
}

/* The recorded block the RSX is being pointed at, or 0xFFFFFFFF.
 *
 * `put` sits at or just past the end of the block the title wants executed --
 * the gap is whatever libgcm appended after the SPU's data, a fence and its
 * argument in this title's case. Require the block to END within a small window
 * before `put` and to START before it, so an older double-buffered block (the
 * SPU alternates between two) is not chosen over the current one. */
static u32 gcm_spu_block_for_put(u32 put)
{
    u32 best = 0xFFFFFFFFu, best_end = 0;
    for (int i = 0; i < GCM_SPU_BLK_N; i++) {
        u32 s = s_spu_blk[i].io_start, e = s_spu_blk[i].io_end;
        if (!e || s >= put || e > put) continue;
        if (put - e > 0x40u) continue;              /* too far back to be current */
        if (e >= best_end) { best_end = e; best = s; }
    }
    return best;
}

/* A resync jumps `get` straight to `put`, which is what keeps the FIFO alive
 * when the walker cannot make progress -- but everything in between is
 * dropped, and if a SET_REFERENCE is in there the cellGcmFinish waiting on it
 * never wakes. Tokyo Jungle lost exactly one that way: two fences in the ring
 * (arg 0 at io 0x09A0, arg 1 at io 0x19E8), the walker parked before the
 * second, and the resync to put=0x19F0 stepped straight over it -- ref stayed
 * 0 while the title spun for ref==1 forever.
 *
 * Dropping DRAWS on a resync is survivable: a frame is lost. Dropping a FENCE
 * is not, it deadlocks the title. Sweep the skipped range first. */
/* ---- v1053A GCM_CALL_ABORT (default 1; 0 = the old v1052B behaviour) -----------------------------------------------
 * User session 2026-11-03 18:2x (play-fast.err.log): "[V1052B] garbage packet at IO 0x03276EC8 (header 0x42340000)" 20+
 * times at the SAME address, each one a "FIFO resync (garbage-packet) -> put" that threw away up to a whole frame of
 * ring commands (SET_REFERENCE fences only half-rescued, SEMAPHORE_RELEASE / label writes / flip notes not at all).
 * Right after: "[SEMA] acquire waits: label 0x810 has CB77, want CB83..CBE3" + "FIFO resync (stuck-get)" = the
 * skipped releases stall the next frames, which are skipped too (walked-vs-guest flip gap 29 -> 47; later +30 / +41 at
 * t~1270 / ~1292 s, invisible because the resync log is capped at 8 lines per run). 0x03xxxxxx is EA's command-list
 * heap (EA = IO + 0x3FF00000, the same 0x43xxxxxx heap as the camera objects, 0x42340000 = 45.0f): the ring CALLs a
 * list whose memory now holds game data. Only that list is bad -- the ring around it is fine.
 * So: while the walker is inside a CALL made from the Init ring window (IO < ioSize) and reads garbage outside that
 * window (V1052B packet, CALL/JUMP to unmapped IO, get on unmapped IO, budget-burning circling), the subroutine is
 * abandoned as if it had executed RET: the walk continues in the ring right after the CALL, every fence / semaphore /
 * flip of the frame executes in order, nothing is rescued or skipped. Also: RET with no CALL no longer jumps to IO 0
 * (walked a stale ring lap) but resyncs; a resync clears the return slot; stuck-get does not fire while the walker is
 * legitimately blocked in a semaphore acquire inside its timeout (GCM_STUCK_SEMA=0 = old: it fired after 8 passes,
 * before the 50 ms acquire timeout). Logs "[V1053A]": first 8 aborts, then a 10 s summary of aborts and resyncs by
 * reason (uncapped counters). */
static int s_v1053 = -1, s_v1053_ss = -1;
static u32 s_v1053_call_tgt = 0;                         /* target of the last CALL taken */
static int s_v1053_semwait = 0;                           /* this pass ended blocked in a semaphore acquire */
static volatile unsigned long s_v1053_n[10];              /* 0 aborts, 1 nested CALL, 2 RET w/o CALL, 3.. resyncs */
static u32 s_v1053_last_at = 0, s_v1053_last_tgt = 0, s_v1053_last_site = 0; static const char* s_v1053_last_why = "";
static int v1053_on(void)
{
    if (__builtin_expect(s_v1053 < 0, 0)) {
        const char* e = getenv("GCM_CALL_ABORT"); s_v1053 = (e && *e == '0') ? 0 : 1;
        const char* t = getenv("GCM_STUCK_SEMA"); s_v1053_ss = (t && *t == '0') ? 0 : 1;
        fprintf(stderr, "[V1053A] garbage inside a CALLed command list returns to the ring instead of resyncing: %s (GCM_CALL_ABORT); "
                "stuck-get waits for a pending semaphore acquire: %s (GCM_STUCK_SEMA)%c", s_v1053 ? "ON" : "off", s_v1053_ss ? "ON" : "off", 10);
    }
    return s_v1053;
}
/* Abandon the current subroutine (see above). Returns 1 when it did; the caller then `continue`s the walk. */
static int v1053_abort_call(const char* why, u32 at, u32 w)
{
    if (!v1053_on() || !s_fifo_calloff) return 0;
    const u32 site = s_fifo_calloff - 4u;
    if (!(site < s_config.ioSize) || at < s_config.ioSize) return 0;   /* only: CALL from the ring, garbage in a list */
    s_v1053_n[0]++;
    s_v1053_last_at = at; s_v1053_last_tgt = s_v1053_call_tgt; s_v1053_last_site = site; s_v1053_last_why = why;
    { static int n = 0; if (n < 8) { n++;
        fprintf(stderr, "[V1053A] %s at IO 0x%08X (word 0x%08X) inside the list CALLed at 0x%08X from ring 0x%08X -> list abandoned, "
                "walk continues at 0x%08X (no resync)%c", why, at, w, s_v1053_call_tgt, site, s_fifo_calloff, 10); } }
    s_fifo_getoff = s_fifo_calloff;
    s_fifo_calloff = 0;
    return 1;
}
static void v1053_report(void)
{
    static u64 t0 = 0; const u64 now = get_timestamp_ns();
    if (!t0) { t0 = now; return; }
    if (now - t0 < 10000000000ull) return;
    t0 = now;
    unsigned long n[10], tot = 0; for (int i = 0; i < 10; i++) { n[i] = s_v1053_n[i]; s_v1053_n[i] = 0; tot += n[i]; }
    if (!tot) return;
    fprintf(stderr, "[V1053A] 10 s: lists abandoned %lu (last: %s at 0x%08X, list 0x%08X from ring 0x%08X), nested CALLs %lu, RET without CALL %lu | "
            "ring resyncs: garbage %lu, unmapped branch %lu, stuck-get %lu, circling %lu, jump-park %lu, other %lu%c",
            n[0], s_v1053_last_why, s_v1053_last_at, s_v1053_last_tgt, s_v1053_last_site, n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8], 10);
}
static void gcm_fifo_resync_why(const char* why, u32* getoff, u32 put)
{
    static int n = 0;
    /* v1053A: count every resync by reason (the line below is capped at 8 per run) and drop the return slot -- the
     * walker lands in the ring, a later RET must not jump back into the abandoned list / an old ring position. */
    if (v1053_on()) {
        const int k = !strcmp(why, "garbage-packet") ? 3 : (!strncmp(why, "unmapped", 8) ? 4 : !strcmp(why, "stuck-get") ? 5 :
                      !strcmp(why, "circling") ? 6 : !strcmp(why, "jump-park") ? 7 : 8);
        s_v1053_n[k]++;
        s_fifo_calloff = 0;
    }
    if ((n < 8 && n++ < 8))
        fprintf(stderr, "[cellGcmSys] FIFO resync (%s) 0x%08X -> put 0x%08X\n",
                why, *getoff, put);
    /* If an SPU DMA'd a pushbuffer that ends just short of `put`, that is the
     * stream the RSX is being pointed at -- land at its START and walk it,
     * instead of landing on `put` and skipping it. Resyncing to `put` keeps the
     * FIFO alive but drops every command in between, and SET_BEGIN_END is one
     * of them: the draws still arrive, with no primitive type, and are dropped
     * by the backend as prim=0. */
    s_v1037_follow = 0;   /* v1037D: a resync lands on the live put; the switch is moot */
    u32 land = gcm_spu_block_for_put(put);
    if (land == 0xFFFFFFFFu) land = put;
    else { static int b = 0; if ((b < 8 && b++ < 8))
        fprintf(stderr, "[cellGcmSys] resync landing on SPU pushbuffer "
                "0x%08X (put 0x%08X) instead of skipping to put\n", land, put); }

    /* Sweep only the range still being skipped; the walker will consume the
     * fences inside the block itself, and pushing them here too would publish
     * every one twice. */
    u32 from = (*getoff <= land) ? *getoff : s_last_ring_off;
    if (from < land) {
        unsigned rescued = 0;
        for (u32 io = from; io + 8 <= land; io += 4) {
            u32 ea = gcm_io2ea(io); if (!ea) continue;
            u32 w = vm_read32(ea);
            if ((w >> 29) != 0 || (w & 0x1FFCu) != 0x0050u) continue;
            u32 dea = gcm_io2ea(io + 4); if (!dea) continue;
            /* v1049R GCM_RESCUE_FILTER (default 1; 0 = old): on a title-stepped ring (Godfather: SET_REFERENCE(segment
             * EA)) the swept bytes can be the PREVIOUS lap's commands; a fence pointing past `land` (ahead of the
             * guest's write position) published as ref made EA's ring manager wait forever for space (user 23:48:
             * last executed ref 0x49F7F234, published 0x49FB6E48 > current 0x49FB6B90). Ring-EA values outside the
             * swept range are dropped; other values (plain counters) are kept. */
            { static int rf = -1; if (rf < 0) { const char* e = getenv("GCM_RESCUE_FILTER"); rf = (e && *e == '0') ? 0 : 1; }
              const u32 fv = vm_read32(dea), fio = gcm_ea2io(fv & ~3u);
              if (rf && s_v1034_title_segs && fio != 0xFFFFFFFFu && (fio < from || fio > land)) {
                  static int nd = 0; if (nd < 8) { nd++; fprintf(stderr, "[V1049R] resync: stale fence 0x%08X (io 0x%08X) outside 0x%08X..0x%08X not published%c", fv, fio, from, land, 10); }
                  continue; } }
            gcm_ref_push_at(vm_read32(dea), io);
            rescued++;
        }
        if (rescued) { static int m = 0; if ((m < 8 && m++ < 8))
            fprintf(stderr, "[cellGcmSys] resync rescued %u fence(s) from"
                    " 0x%08X..0x%08X\n", rescued, from, land); }
    }
    *getoff = land;
}

static void gcm_fifo_dump_around(u32 getoff)
{
    static int d = 0;
    if (!getenv("GCM_RECDBG") || d++ >= 3) return;
    for (u32 t = 0; t < GCM_TRACE_N; t++) {
        u32 k = (s_tr_i + t) % GCM_TRACE_N;
        if (!s_tr_w[k]) continue;
        u32 w = s_tr_w[k];
        fprintf(stderr, "[FIFOSTEP] io=%08X w=%08X  type=%u method=0x%04X count=%u\n",
                s_tr_off[k], w, w >> 29, w & 0x1FFCu, (w >> 18) & 0x7FFu);
    }
    for (int k = -20; k <= 2; k++) {
        u32 io = getoff + (u32)(k * 4);
        u32 ea = gcm_io2ea(io);
        if (!ea) continue;
        fprintf(stderr, "[FIFODUMP] %+3d io=%08X w=%08X%s\n",
                k, io, vm_read32(ea), k ? "" : "   <-- bad word");
    }
}


static void gcm_ref_push_at(u32 v, u32 getoff)
{
    { static int _d = -1; if (_d < 0) _d = getenv("GCM_REFLOG") ? 1 : 0;
      if (_d) fprintf(stderr, "[refq] drained fence 0x%X at getoff=%08X\n", v, getoff); }
    u32 t = s_ref_qtail;
    if (t - s_ref_qhead >= GCM_REF_QLEN) {   /* overflow: drop oldest (keeps liveness) */
        s_ref_qhead++;
        static int _o = 0;
        if ((_o < 4 && _o++ < 4)) fprintf(stderr, "[cellGcmSys] fence queue overflow -- oldest dropped\n");
    }
    s_ref_q[t % GCM_REF_QLEN] = v;
    s_ref_qtail = t + 1;
}

/* Global fence-publication counter: lets the lwmutex convoy trace correlate a
 * long lock hold with the number of paced fence publications inside it (the
 * 200us pacing x hundreds of one-ahead fences = the ~156ms holds). */
volatile long long g_gcm_ref_pub_count = 0;
/* Absolute microseconds from the QPC hardware counter -- the SAME origin in every
 * translation unit, so two probes in different files can be compared directly.
 * That is the whole point: every wrong conclusion about this stall came from
 * comparing orderings instead of times. */
unsigned long long ps3_qpc_us(void)
{
    static LARGE_INTEGER f; if (!f.QuadPart) QueryPerformanceFrequency(&f);
    LARGE_INTEGER n; QueryPerformanceCounter(&n);
    return (unsigned long long)(n.QuadPart / (f.QuadPart / 1000000));
}

/* ---- v1048D: guest -> RSX -> guest round trip (2026-11-02) ---------------------------------------------------------
 * Godfather city (soak Y34): the main thread sleeps ~6 ms per frame in two poll loops, ~3.6 ms of it in EA's ring
 * manager (func_008A3454: usleep(30) via its sleep vtable while ctrl->ref == current, then until (ref - current) is
 * big enough -- an INEQUALITY / distance wait on the segment EAs it writes with SET_REFERENCE) and the rest in
 * func_0072E420 (GetFlipStatus poll, usleep(300) at lr 0x72E450, before each SetFlip). [REFPOLL] shows 98 % of the
 * ~5.5k ref polls/s with drained fences waiting for the 200 us publication gap, mean depth ~35: the walker is far
 * AHEAD, the guest-visible ref lags it by ~35 x 200 us, and the publication cap (~5k fences/s, ~65 per city frame)
 * is itself a ~80 FPS ceiling.
 * GCM_REF_LATEST=1 (default 0 = old paced publication): a ref poll (and the walker's per-walk publish) writes the
 * NEWEST drained fence at once -- what a real RSX does (the register holds the last SET_REFERENCE it executed).
 * Caveat: a thread waiting for ref == an OLDER value that is already passed would not see it (the reason for the
 * v85 gap: LBP boot); Godfather's ring manager and libgcm's cellGcmFinish (waits for the value it just wrote, the
 * newest) are fine.
 * GCM_FLIP_ON_WALK=1 (default 0): a flip also completes (cellGcmTickFlip) as soon as the FIFO walker passes the v62
 * flip note of the CURRENT flip request, instead of only at the next frame-clock tick (~6.6 ms period at 144).
 * The tick still completes it too (whichever comes first). Needs the v62 notes (GCM_FLIP_ORDERED, default on).
 * V1048D_PHASE=1: in-run A/B -- the enabled switches act only in even V1048D_PHASE_SEC (default 12) s phases, odd =
 * old; per-phase [V1048D] lines (main thread frame-time p50/p90/p99, flip-status and fence wait per flip) and a
 * running A/B for city phases (>= 1000 draws per flip). V1048D_STATS=1: the same lines every 10 s without A/B. */
static volatile int s_v1048_init = 0;
static int s_v1048_ref = 0, s_v1048_flip = 0, s_v1048_ph = 0, s_v1048_st = 0;
static unsigned long long s_v1048_t00 = 0, s_v1048_len = 12000000ull;
static u32 s_v1048_note_req[8];          /* flip request number each v62 note slot (s_v1034_note_io) belongs to */
static void v1048_init(void)
{
    if (s_v1048_init) return;
    const char* e;
    e = getenv("GCM_REF_LATEST");   s_v1048_ref  = (e && *e == '1') ? 1 : 0;
    e = getenv("GCM_FLIP_ON_WALK"); s_v1048_flip = (e && *e == '1') ? 1 : 0;
    e = getenv("V1048D_PHASE");     s_v1048_ph   = (e && *e == '1' && (s_v1048_ref || s_v1048_flip)) ? 1 : 0;
    e = getenv("V1048D_STATS");     s_v1048_st   = (s_v1048_ph || (e && *e == '1')) ? 1 : 0;
    e = getenv("V1048D_PHASE_SEC");
    { long sec = e ? atol(e) : 12; if (sec < 2) sec = 2; s_v1048_len = (unsigned long long)sec * 1000000ull; }
    s_v1048_t00 = ps3_qpc_us();
    fprintf(stderr, "[V1048D] ref publication: newest drained fence at once %s (GCM_REF_LATEST)%c", s_v1048_ref ? "ON" : "off", 10);
    fprintf(stderr, "[V1048D2] flip completes when the FIFO walker passes its flip note %s (GCM_FLIP_ON_WALK)%c", s_v1048_flip ? "ON" : "off", 10);
    if (s_v1048_ph)
        fprintf(stderr, "[V1048D] in-run A/B: switches active only in even %llu s phases, odd = old (V1048D_PHASE=1)%c",
                s_v1048_len / 1000000ull, 10);
    else if (s_v1048_st)
        fprintf(stderr, "[V1048D] per-10 s main-thread frame/wait stats ON (V1048D_STATS)%c", 10);
    s_v1048_init = 1;
}
static inline int v1048_active(int sw)
{
    if (__builtin_expect(!s_v1048_init, 0)) v1048_init();
    if (!sw) return 0;
    if (!s_v1048_ph) return 1;
    return !((int)((ps3_qpc_us() - s_v1048_t00) / s_v1048_len) & 1);
}
static inline int v1048_ref_now(void)  { if (__builtin_expect(!s_v1048_init, 0)) v1048_init(); return s_v1048_ref  ? v1048_active(1) : 0; }
static inline int v1048_flip_now(void) { if (__builtin_expect(!s_v1048_init, 0)) v1048_init(); return s_v1048_flip ? v1048_active(1) : 0; }

#define V1048_NB 400   /* frame-interval histogram: 0.25 ms bins, 0..100 ms (+ overflow) */
typedef struct {
    unsigned long long n, hist[V1048_NB + 1], flipw_polls, refpolls, depth_sum, draws, walkdone;
    double sum_ms, max_ms, flipw_ms, fence_ms, secs, phases;
} v1048_stat;
static v1048_stat s_v1048_cur, s_v1048_acc[2];   /* main guest thread only (ps3_hz_is_main) */
static double v1048_pct(const v1048_stat* s, double q)
{
    if (!s->n) return 0.0;
    unsigned long long want = (unsigned long long)(q * (double)s->n), c = 0;
    for (int b = 0; b <= V1048_NB; b++) { c += s->hist[b]; if (c > want) return b * 0.25 + 0.125; }
    return V1048_NB * 0.25;
}
static void v1048_print(const char* head, const v1048_stat* s)
{
    const double n = s->n ? (double)s->n : 1.0;
    fprintf(stderr, "%s %.1f s, %llu flips (%.1f/s) | frame ms mean %.2f p50 %.2f p90 %.2f p99 %.2f max %.1f | main wait per flip: "
            "flip-status %.2f ms (%.1f WAITING polls), fence %.2f ms | ref polls/flip %.1f, queued depth %.1f | walk-completed flips %.0f%% | draws/flip %.0f%c",
            head, s->secs, s->n, s->secs > 0 ? s->n / s->secs : 0.0, s->sum_ms / n, v1048_pct(s, 0.50), v1048_pct(s, 0.90),
            v1048_pct(s, 0.99), s->max_ms, s->flipw_ms / n, s->flipw_polls / n, s->fence_ms / n, s->refpolls / n,
            s->refpolls ? (double)s->depth_sum / (double)s->refpolls : 0.0, 100.0 * s->walkdone / n, s->draws / n, 10);
}
static void v1048_report(int ph, double secs)
{
    v1048_stat* c = &s_v1048_cur;
    const int on = s_v1048_ph ? !(ph & 1) : (s_v1048_ref || s_v1048_flip);
    c->secs = secs;
    char head[96];
    snprintf(head, sizeof head, "[V1048D] phase %d (%s):", ph, on ? "ON" : "off");
    v1048_print(head, c);
    if (s_v1048_ph && c->n && secs > 3.0 && (double)c->draws / (double)c->n >= 1000.0) {   /* city phases */
        v1048_stat* a = &s_v1048_acc[on ? 0 : 1];
        a->n += c->n; for (int b = 0; b <= V1048_NB; b++) a->hist[b] += c->hist[b];
        a->flipw_polls += c->flipw_polls; a->refpolls += c->refpolls; a->depth_sum += c->depth_sum; a->draws += c->draws;
        a->walkdone += c->walkdone; a->sum_ms += c->sum_ms; if (c->max_ms > a->max_ms) a->max_ms = c->max_ms;
        a->flipw_ms += c->flipw_ms; a->fence_ms += c->fence_ms; a->secs += c->secs; a->phases += 1.0;
        if (s_v1048_acc[0].n && s_v1048_acc[1].n) {
            snprintf(head, sizeof head, "[V1048D] A/B city ON  (%.0f phases):", s_v1048_acc[0].phases); v1048_print(head, &s_v1048_acc[0]);
            snprintf(head, sizeof head, "[V1048D] A/B city off (%.0f phases):", s_v1048_acc[1].phases); v1048_print(head, &s_v1048_acc[1]);
        }
    }
}
static void v1048_on_flip(unsigned draws)   /* gcm_flip_request: once per guest frame */
{
    if (__builtin_expect(!s_v1048_init, 0)) v1048_init();
    if (!s_v1048_st) return;
    extern int ps3_hz_is_main(void);
    if (!ps3_hz_is_main()) return;
    static unsigned long long last = 0, ph_t0 = 0; static int ph_cur = -1;
    const unsigned long long now = ps3_qpc_us();
    const int ph = (int)((now - s_v1048_t00) / (s_v1048_ph ? s_v1048_len : 10000000ull));
    if (ph != ph_cur) {
        if (ph_cur >= 0) v1048_report(ph_cur, (double)(now - ph_t0) * 1e-6);
        ph_cur = ph; ph_t0 = now; memset(&s_v1048_cur, 0, sizeof s_v1048_cur);
        last = 0;   /* no interval across a switch */
    }
    if (last) {
        const double ms = (double)(now - last) * 1e-3;
        int b = (int)(ms * 4.0); if (b > V1048_NB) b = V1048_NB; if (b < 0) b = 0;
        s_v1048_cur.hist[b]++; s_v1048_cur.n++; s_v1048_cur.sum_ms += ms;
        if (ms > s_v1048_cur.max_ms) s_v1048_cur.max_ms = ms;
        s_v1048_cur.draws += draws;
    }
    last = now;
}
static void v1048_flipstatus_note(u32 st)   /* cellGcmGetFlipStatus */
{
    if (!s_v1048_st) return;
    extern int ps3_hz_is_main(void);
    if (!ps3_hz_is_main()) return;
    static unsigned long long t0 = 0;
    const unsigned long long now = ps3_qpc_us();
    if (st != CELL_GCM_FLIP_STATUS_DONE) { if (!t0) t0 = now; s_v1048_cur.flipw_polls++; }
    else if (t0) { s_v1048_cur.flipw_ms += (double)(now - t0) * 1e-3; t0 = 0; }
}
static void v1048_refpoll_note(u32 depth)   /* cellGcm_ref_on_poll: main-thread fence polls < 200 us apart = waiting */
{
    extern int ps3_hz_is_main(void);
    if (!ps3_hz_is_main()) return;
    static unsigned long long last = 0;
    const unsigned long long now = ps3_qpc_us();
    if (last && now - last < 200ull) s_v1048_cur.fence_ms += (double)(now - last) * 1e-3;
    last = now;
    s_v1048_cur.refpolls++; s_v1048_cur.depth_sum += depth;
}
/* GCM_REF_LATEST: publish the newest drained fence, consuming the whole queue. Caller holds s_ref_pub_lock. */
static void gcm_ref_publish_latest(void)
{
    const u32 h = s_ref_qhead, t = s_ref_qtail;
    if (h == t) return;
    vm_write32(GCM_CONTROL_GUEST_ADDR + 8, s_ref_q[(t - 1u) % GCM_REF_QLEN]);
    s_ref_qhead = t;
    g_gcm_ref_pub_count += (long long)(u32)(t - h);
}
/* ---- end v1048D ------------------------------------------------------------------------------------------------- */

/* v1049H: called from the [HOTREAD] GCM-control spin detector (ppu_loader.cpp) -- the state a ring-manager
 * freeze needs: what the walker last executed vs what the guest sees, and the default context's ring. */
void cellGcm_hotread_diag(void)
{
    extern u32 g_rsx_last_reference;
    const u32 ctx = s_gcm_context_ea;
    fprintf(stderr, "[V1049H] walker get 0x%08X, last SET_REFERENCE executed 0x%08X, fence queue %u queued (head %u tail %u) | "
            "default ctx 0x%08X begin 0x%08X end 0x%08X current 0x%08X | flip req %u noted %u walked %u status %u%c",
            s_fifo_getoff, g_rsx_last_reference, s_ref_qtail - s_ref_qhead, s_ref_qhead, s_ref_qtail,
            ctx, ctx ? vm_read32(ctx) : 0u, ctx ? vm_read32(ctx + 4) : 0u, ctx ? vm_read32(ctx + 8) : 0u,
            s_flip_request_count, s_v1049_noted_req, s_v1049_walked_req, s_flip_status, 10);
}

static void gcm_ref_publish_one(void)
{
    u32 h = s_ref_qhead;
    if (h == s_ref_qtail) return;
    vm_write32(GCM_CONTROL_GUEST_ADDR + 8, s_ref_q[h % GCM_REF_QLEN]);
    /* stderr, deliberately: [DRAIN] uses printf and stdout is block-buffered
     * when redirected, so its interleaving with an stderr probe is not a
     * timeline. Same stream = same ordering. */
    { static int _rd = -1; if (_rd < 0) _rd = getenv("GCM_REFPUB") ? 1 : 0;
      if (_rd) { static unsigned long _n = 0; if (++_n <= 24)
        fprintf(stderr, "[refpub] t=%lluus #%lu wrote 0x%08X -> readback 0x%08X\n", ps3_qpc_us(), _n,
                s_ref_q[h % GCM_REF_QLEN], vm_read32(GCM_CONTROL_GUEST_ADDR + 8)); } }
    s_ref_qhead = h + 1;
    g_gcm_ref_pub_count++;
}

/* Read-driven fence publication. cellGcmFinish / FIFO-space waits spin-read the
 * ref register (GCM_CONTROL+8). The 60 Hz present-thread ticker publishes only
 * one queued fence per frame, so when present overruns during heavy boot load
 * the tick rate -- and thus fence publication -- collapses and an equality-
 * waiter stalls seconds per fence (the [finspin] boot crawl; the game is not
 * deadlocked, just starved). Let the spinning reader drive publication too:
 * this is called from vm_read32 on each ref read, advancing one queued fence,
 * globally paced to at most one per millisecond so every value stays visible
 * >= 1 ms to any equality-waiter (no skip-past). Decoupled from present, so a
 * wait makes progress even when the ticker is starved. Idempotent + additive:
 * it only ever publishes fences the ticker would eventually publish anyway. */
/* Serializes the FIFO walker between the 60 Hz present thread and the
 * poll-path below. TryAcquire on the poll path avoids re-entrancy (the walker
 * itself performs guest reads) and never blocks a spinning game thread. */
static SRWLOCK s_gcm_fifo_lock = SRWLOCK_INIT;
static void gcm_rsx_process_fifo_unlocked(void);

/* Kick event: a game thread whose fence wait finds the queue DRY signals the
 * present thread to run the FIFO walker NOW instead of on its next 16 ms
 * tick. A movie frame issues ~46 one-ahead cellGcmFinish waits; paying a
 * 16 ms tick per fence was ~0.75 s/frame (the 2 FPS intro). The walker (and
 * the D3D12 backend it feeds) stays on the present thread -- running it from
 * the polling game thread killed the process silently. */
static HANDLE s_gcm_kick_ev = NULL;
static INIT_ONCE s_gcm_kick_once = INIT_ONCE_STATIC_INIT;
static volatile LONG s_gcm_kick_pending = 0;
static BOOL CALLBACK gcm_kick_init(PINIT_ONCE once, PVOID param, PVOID* context)
{
    (void)once; (void)param; (void)context;
    s_gcm_kick_ev = CreateEventA(NULL, FALSE, FALSE, NULL);
    return s_gcm_kick_ev != NULL;
}
void* cellGcm_fifo_kick_event(void)
{
    InitOnceExecuteOnce(&s_gcm_kick_once, gcm_kick_init, NULL, NULL);
    return (void*)s_gcm_kick_ev;
}

/* The present thread's idle wait. It used to be a flat Sleep(4), which meant
 * the kick above was created, signalled -- and waited on by nobody, so a
 * starved guest still paid the full tick. Waiting here instead costs nothing
 * when no one kicks (the timeout is the old sleep) and drains immediately when
 * someone does. */
/* v28: guest store to the FIFO put register while the present thread idles
 * in cellGcm_fifo_kick_wait -> wake it now (called from ppu_loader vm_write32). */
volatile long g_rsx_idle_wait = 0;
int v243c_rfbound_on(void);                    /* v243C RFBOUND (defined below) */
extern volatile long long g_v243c_st[16];
static volatile long long g_v243c_last_put_qpc;
void cellGcm_put_kick(void)
{
    if (__builtin_expect(v243c_rfbound_on(), 0)) {   /* v243C RFBOUND: guest put stores */
        LARGE_INTEGER q; QueryPerformanceCounter(&q); g_v243c_last_put_qpc = q.QuadPart; g_v243c_st[0]++; }
    /* Keep kicks made BETWEEN the last drain and entering the wait. Testing
     * idle alone lost that notification and added a full 4ms timeout. Coalesce
     * active-period stores so we do not issue a kernel call for every PUT. */
    if (!InterlockedExchange(&s_gcm_kick_pending, 1)) {
        HANDLE ev = (HANDLE)cellGcm_fifo_kick_event();
        if (ev) SetEvent(ev);
    }
}
/* v1129W GF_FIFO_KICK2 (default 0 = old): lost FIFO wake-ups. cellGcm_put_kick coalesces put stores while
 * s_gcm_kick_pending is 1, and only this wait clears it. A put stored while the walker is busy (pending still 1
 * from the put that woke it) therefore sets no event: the walk ends at the put it read on entry, this wait clears
 * pending and sleeps up to its timeout (<= 4 ms) with unseen commands in the ring -- unless a dry fence poll of
 * the main thread happens to kick. The FIFO thread then idles while it is the frame limit ([RFBOUND] fifo-walk,
 * [BUSY-FIFO] idle 55-60 %, put -> drained 1.4 ms of which 1.0 ms walking).
 * Bit 1: after clearing pending, skip the sleep when the guest put differs from the put the last walk read (a
 * store after this check finds pending == 0 and sets the event, so nothing is lost either way).
 * Bit 2 (A/B separately): also skip it when the last walk stopped at a flip word with commands still queued
 * behind it (the flip has been presented by the caller before it waits again).
 * Same commands, same order, same flips and presents -- the walker only starts them sooner.
 * [v1129W] 10 s report: sleeps skipped per reason / sleeps taken. */
static u32 s_v1129w_walk_put = 0;
static int s_v1129w_walk_valid = 0, s_v1129w_flip_backlog = 0;
static int s_v1129w_kick2 = -1;
static unsigned long long s_v1129w_n[3];
static int v1129w_kick2_mode(void)
{
    if (__builtin_expect(s_v1129w_kick2 < 0, 0)) {
        const char* e = getenv("GF_FIFO_KICK2");
        s_v1129w_kick2 = e ? (int)(strtol(e, NULL, 0) & 3) : 0;
        if (s_v1129w_kick2)
            fprintf(stderr, "[v1129W] FIFO kick wait: no sleep while the walker has unseen work ON (GF_FIFO_KICK2=%d:%s%s)%c",
                    s_v1129w_kick2, (s_v1129w_kick2 & 1) ? " new put" : "", (s_v1129w_kick2 & 2) ? " + flip backlog" : "", 10);
    }
    return s_v1129w_kick2;
}
static void v1129w_kick2_report(void)
{
    static ULONGLONG t0 = 0;
    const ULONGLONG now = GetTickCount64();
    if (!t0) { t0 = now; return; }
    if (now - t0 < 10000ull) return;
    t0 = now;
    fprintf(stderr, "[v1129W] 10 s: kick-wait sleeps skipped: new put %llu, flip backlog %llu | sleeps taken %llu%c",
            s_v1129w_n[0], s_v1129w_n[1], s_v1129w_n[2], 10);
    s_v1129w_n[0] = s_v1129w_n[1] = s_v1129w_n[2] = 0;
}
void cellGcm_fifo_kick_wait(unsigned ms)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("GCM_FIFO_KICK"); on = (e && *e == '0') ? 0 : 1; }
    HANDLE ev = on ? (HANDLE)cellGcm_fifo_kick_event() : NULL;
    if (ev) {
        InterlockedExchange(&s_gcm_kick_pending, 0);
        const int k2 = v1129w_kick2_mode();
        if (k2) {   /* v1129W GF_FIFO_KICK2 (the exchange above is a full barrier: a put stored before it is seen here) */
            v1129w_kick2_report();
            if (s_v1129w_walk_valid) {
                if ((k2 & 1) && vm_read32(GCM_CONTROL_GUEST_ADDR + 0) != s_v1129w_walk_put) { s_v1129w_n[0]++; return; }
                if ((k2 & 2) && s_v1129w_flip_backlog) { s_v1129w_flip_backlog = 0; s_v1129w_n[1]++; return; }
            }
            s_v1129w_n[2]++;
        }
        WaitForSingleObject(ev, ms);
    }
    else    Sleep(ms);
}


/* Serializes fence publication between the present ticker and every guest
 * thread poll-reading the ref register (REFPOLL default-on made this path
 * multi-threaded; the SPSC queue head raced and publication order broke --
 * boot froze with all threads parked on fences that were never published in
 * order). TryAcquire: a contended poll just reads the current ref. */
static SRWLOCK s_ref_pub_lock = SRWLOCK_INIT;
/* v207: where the FIFO thread is (main.cpp frame_clock + rf_wait_pumping): 0 other, 1 idle kick wait, 2 walking the FIFO,
 * 3 present/hand-off, 4 window pump, 5 waiting for the render thread. [REFPOLL] splits dry polls by it. */
volatile int g_fifo_where = 0;
static unsigned long long s_dry_where[8];

/* v243C RFBOUND=1 (default off: nothing below runs): wall time the MAIN guest thread spends in its fence poll loops
 * (func_001167C4 usleep(5) on ctrl->ref), split by what the FIFO thread was doing at each poll (g_fifo_where; 6 = the
 * FIFO thread waits for the GPU of frame N-1, set by the backend while RFBOUND is on). Consecutive main-thread polls
 * less than 200 us apart count as waiting. Read per frame by the backend's [RFBOUND] report. */
volatile long long g_v243c_fence_qpc[8];
volatile int g_v243c_fifo_gpuwait = 0;
/* [0] guest put stores, [1] main fence waits begun, [2] FIFO backlog bytes (put - get) at wait start, [3] QPC since the
 * last put store at wait start, [4] QPC the FIFO walker spun in NV406E ACQUIRE (SPU label waits), [5] the part of [4]
 * while the main thread was in a fence wait, [6] ACQUIRE spins, [7] QPC of the main thread's last fence poll,
 * [8] QPC from a put store until the walker drained it (get == put), [9] the walker's time inside the FIFO walk in
 * that window, [10] such windows */
volatile long long g_v243c_st[16];
int v243c_rfbound_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("RFBOUND"); on = (e && *e == '1') ? 1 : 0; }
    return on;
}
static volatile long long g_v243c_last_put_qpc = 0;
static void v243c_fence_note(void)
{
    extern int ps3_hz_is_main(void);
    if (!ps3_hz_is_main()) return;
    static LARGE_INTEGER f; static LONGLONG last = 0, thr = 0;
    if (!f.QuadPart) { QueryPerformanceFrequency(&f); thr = f.QuadPart / 5000; }   /* 200 us */
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    if (last && now.QuadPart - last < thr) {
        int w = g_fifo_where & 7;
        if (w == 5 && g_v243c_fifo_gpuwait) w = 6;
        g_v243c_fence_qpc[w] += now.QuadPart - last;
    } else {   /* a new wait begins: how far behind is the walker, how long since the guest published commands? */
        const u32 put = vm_read32(GCM_CONTROL_GUEST_ADDR + 0);
        const u32 back = put - s_fifo_getoff;
        g_v243c_st[1]++;
        if (back < 0x01000000u) g_v243c_st[2] += back;
        if (g_v243c_last_put_qpc) g_v243c_st[3] += now.QuadPart - g_v243c_last_put_qpc;
    }
    last = now.QuadPart;
    g_v243c_st[7] = now.QuadPart;
}
void cellGcm_ref_on_poll(void)
{
    if (__builtin_expect(v243c_rfbound_on(), 0)) v243c_fence_note();   /* v243C RFBOUND */
    /* Pace: leave each published value observable for >= ~200us of spins
     * (equality-waiters at any poll rate see every value; still ~5000/s so
     * loading's fence storms clear in seconds, not minutes). */
    static LARGE_INTEGER s_freq;
    static volatile LONGLONG s_last_pub;
    if (!TryAcquireSRWLockExclusive(&s_ref_pub_lock))
        return;
    if (!s_freq.QuadPart) QueryPerformanceFrequency(&s_freq);
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    /* v85: gap configurable (GCM_REF_GAP_US, default 200). inFamous' frame-begin fence wait
     * (func_001167C4, usleep(5) loop on ctrl->ref) is an INEQUALITY wait, so a backlog of
     * queued fences published 200 us apart is pure latency on the main thread. PS3_AB phase
     * mask 1 uses GCM_REF_GAP_AB_US (default 10). [REFPOLL] says which limit applies:
     * "queued" polls = a drained fence was waiting for the pacing gap, "dry" = the FIFO
     * walker had not reached the fence yet. */
    static LONGLONG g_base = -1, g_ab = -1;
    if (g_base < 0) { const char* e = getenv("GCM_REF_GAP_US"); long us = e ? atol(e) : 200;
                      const char* a = getenv("GCM_REF_GAP_AB_US"); long ua = a ? atol(a) : 10;
                      g_ab = (LONGLONG)ua * s_freq.QuadPart / 1000000; g_base = (LONGLONG)us * s_freq.QuadPart / 1000000;
                      fprintf(stderr, "[REFPOLL] v85 fence publication gap %ld us (A/B phase 1: %ld us)%c", us, ua, 10); }
    extern volatile int g_ab_mask;
    LONGLONG min_gap = (g_ab_mask == 1) ? g_ab : g_base;
    int dry = 0;
    static unsigned long long st_polls, st_queued, st_dry, st_depth; static LONGLONG st_t0;
    st_polls++;
    { u32 dep = s_ref_qtail - s_ref_qhead; if (dep) { st_queued++; st_depth += dep; } else { st_dry++; s_dry_where[g_fifo_where & 7]++; }
      if (__builtin_expect(s_v1048_st, 0)) v1048_refpoll_note(dep); }   /* v1048D stats */
    if (v1048_ref_now()) {   /* v1048D GCM_REF_LATEST: no pacing -- newest drained fence; nothing new = kick the walker */
        dry = (s_ref_qhead == s_ref_qtail);
        gcm_ref_publish_latest();
        s_last_pub = now.QuadPart;
    } else
    if (now.QuadPart - s_last_pub >= min_gap) {
        gcm_ref_publish_one();
        s_last_pub = now.QuadPart;
        dry = (s_ref_qhead == s_ref_qtail);
    }
    if (!st_t0) st_t0 = now.QuadPart;
    else if (now.QuadPart - st_t0 >= s_freq.QuadPart * 10) {
        fprintf(stderr, "[REFPOLL] 10s: %llu ref polls, %.1f%% with a drained fence queued (mean depth %.2f), %.1f%% dry (FIFO behind); gap %lld us (ab mask %d)%c",
                st_polls, st_polls ? 100.0 * st_queued / st_polls : 0.0, st_queued ? (double)st_depth / st_queued : 0.0,
                st_polls ? 100.0 * st_dry / st_polls : 0.0, (long long)(min_gap * 1000000 / s_freq.QuadPart), g_ab_mask, 10);
        { unsigned long long t = 0; for (int k = 0; k < 8; k++) t += s_dry_where[k]; if (!t) t = 1;
          fprintf(stderr, "[REFPOLL] v207 dry polls by FIFO-thread state: walking %.1f%% | render-thread wait %.1f%% | present %.1f%% | idle %.1f%% | pump %.1f%% | other %.1f%%%c",
                  100.0 * s_dry_where[2] / t, 100.0 * s_dry_where[5] / t, 100.0 * s_dry_where[3] / t, 100.0 * s_dry_where[1] / t,
                  100.0 * s_dry_where[4] / t, 100.0 * s_dry_where[0] / t, 10);
          for (int k = 0; k < 8; k++) s_dry_where[k] = 0; }
        st_polls = st_queued = st_dry = st_depth = 0; st_t0 = now.QuadPart;
    }
    ReleaseSRWLockExclusive(&s_ref_pub_lock);
    if (dry && s_gcm_kick_ev)
        SetEvent(s_gcm_kick_ev);       /* dry: ask the present thread to walk */
}

/* v221C DCOAL: coalesce the 256-index words of one DRAW_INDEX_ARRAY packet (see the GCM_GEN_FAST loop below). */
unsigned long long g_gcm_dcoal_runs = 0, g_gcm_dcoal_words = 0;   /* read + reset by the D3D12 [DCOAL] report */
static int gcm_dcoal_active(void)
{
    static int on = -1;
    extern volatile int g_rsx_dcoal_ok;   /* rsx_commands.c: set by a back end that takes counts > 256 */
    extern volatile int g_abx;
    if (__builtin_expect(on < 0, 0)) {
        const char* e = getenv("DCOAL"); on = (e && *e == '0') ? 0 : 1;
        fprintf(stderr, "[DCOAL] v221C DRAW_INDEX_ARRAY run coalescing: %s (DCOAL=0 off; ABX 0x100000 = off; back end %s)%c",
                on ? "ON" : "off", g_rsx_dcoal_ok ? "ready" : "NOT ready", 10);
    }
    return on && g_rsx_dcoal_ok && !(g_abx & 0x100000);
}
void cellGcm_rsx_process_fifo(void)
{
    AcquireSRWLockExclusive(&s_gcm_fifo_lock);
    gcm_rsx_process_fifo_unlocked();
    ReleaseSRWLockExclusive(&s_gcm_fifo_lock);
}

static void gcm_rsx_process_fifo_unlocked(void)
{
    { static unsigned _n = 0; static unsigned long long _t0 = 0;
      extern unsigned long long ps3_ms_now(void);
      _n++;
      static int _rate = -1; if (_rate < 0) _rate = getenv("GCM_RATE") ? 1 : 0;   /* v180: cached (was a getenv per drain) */
      if (_rate) {
          unsigned long long now = ps3_ms_now();
          if (_t0 == 0) _t0 = now;
          if (now - _t0 >= 2000) {
              fprintf(stderr, "[RATE] drains/sec=%.0f\n", _n * 1000.0 / (now - _t0));
              _n = 0; _t0 = now;
          }
      } }
    static rsx_state s_state;
    static int  s_inited = 0;
    static int  s_gf_fifo_fast = -1, s_gf_fifo_fast2 = -1;
    extern u32 g_rsx_last_reference;
    if (__builtin_expect(s_gf_fifo_fast < 0, 0)) {
        const char* e = getenv("GF_FIFO_FAST");
        s_gf_fifo_fast = (e && *e == '1') ? 1 : 0;
        if (s_gf_fifo_fast)
            fprintf(stderr, "[v1121G] FIFO packet decode fast paths ON (GF_FIFO_FAST=1)%c", 10);
    }
    const int gf_fifo_fast = s_gf_fifo_fast;
    if (__builtin_expect(s_gf_fifo_fast2 < 0, 0)) {
        const char* e = getenv("GF_FIFO_FAST2");
        if (e && *e == '1') {
            extern int rsx_v1129g_fast2_active(void);
            s_gf_fifo_fast2 = rsx_v1129g_fast2_active();
        } else {
            s_gf_fifo_fast2 = 0;
        }
    }
    const int gf_fifo_fast2 = s_gf_fifo_fast2;
    /* v1047A GCM_FASTWALK=1 (default off): bulk paths for whole method packets in the walker (see the two users
     * below). GCM_FASTWALK_MASK (default 0x7): 0x1 = rsx_commands.c dispatch table, 0x2 = transform-program
     * packets, 0x4 = transform-constant blocks. GCM_FASTWALK_STATS=1: a [V1047A] word/method report every 10 s. */
    static int s_v1047 = -1, s_v1047_tp = 0, s_v1047_tc = 0, s_v1047_st = 0, s_v1047_ph = 0;
    static unsigned long long s_v1047_n[8]; static unsigned long long s_v1047_t0 = 0;
    if (__builtin_expect(s_v1047 < 0, 0)) {
        const char* e = getenv("GCM_FASTWALK");
        const char* mk = getenv("GCM_FASTWALK_MASK");
        const char* st = getenv("GCM_FASTWALK_STATS");
        const unsigned long mask = mk ? strtoul(mk, NULL, 0) : 0x7ul;
        const int on = (e && *e == '1') ? 1 : 0;
        const int diag = getenv("RSX_TRACE") || getenv("IMM_MTRACE") || getenv("RSX_MHIST") ||
                         getenv("TCONST_DBG") || getenv("SEQ_DBG");
        s_v1047_tp = on && (mask & 0x2ul) && !diag;
        s_v1047_tc = on && (mask & 0x4ul);
        s_v1047_st = (st && *st == '1') ? 1 : 0;
        { const char* ph = getenv("GCM_FASTWALK_PHASE"); s_v1047_ph = (on && ph && *ph == '1') ? 1 : 0; }
        fprintf(stderr, "[V1047A] FIFO walker bulk packets: transform program %s, transform constants %s%s (GCM_FASTWALK, GCM_FASTWALK_MASK 0x%lX)%s%c",
                s_v1047_tp ? "ON" : "off", s_v1047_tc ? "ON" : "off",
#ifdef GCM_V1047_SSSE3
                s_v1047_tc ? " (SSSE3)" : "",
#else
                s_v1047_tc ? " (scalar)" : "",
#endif
                mask, s_v1047_st ? "; stats ON (GCM_FASTWALK_STATS)" : "", 10);
        if (s_v1047_ph) fprintf(stderr, "[V1047A] in-run A/B: fastwalk only in even phases of GCM_FASTWALK_PHASE_SEC (default 12) s, odd = old path (GCM_FASTWALK_PHASE=1; per-phase lines need GCM_FASTWALK_STATS=1)%c", 10);
        s_v1047 = on;
    }

    /* A statically-linked libgcm (PS3 firmware modules -- ps1_netemu) never calls
     * cellGcmSetupContext, so there is no CellGcmContextData EA to gate on: it gets
     * its ring through sys_rsx_context_allocate instead. The walk below only needs
     * `put` and the IO table, both of which the syscall layer fills; the one part
     * that does read the context (the ring-recycle heuristic) is separately guarded
     * by s_gcm_ctx_out_ea, which stays 0 on that path. */
    if (!s_gcm_context_ea && !s_gcm_syscall_mode) { s_v1129w_walk_valid = 0; return; }   /* v1129W: no walk = no skipped sleeps */
    if (!s_inited) { rsx_state_init(&s_state); s_inited = 1; }
    /* v1047A: which paths this walk takes (GCM_FASTWALK_PHASE=1: even PS3_AB phase = new, odd = old) and, in that
     * A/B mode, a per-phase [V1047A] report of the walk time (QPC around each walk). */
    int v1047_now = s_v1047, v1047_phase = 0;
    LARGE_INTEGER v1047_q0 = {0};
    if (__builtin_expect(s_v1047_ph, 0)) {   /* own phase clock (PS3_AB not needed): GCM_FASTWALK_PHASE_SEC, default 12 s */
        extern volatile int g_v1047_fw_now;
        static LARGE_INTEGER ph_qf, ph_q00; static long long ph_len = 0;
        QueryPerformanceCounter(&v1047_q0);
        if (!ph_len) { const char* e = getenv("GCM_FASTWALK_PHASE_SEC"); long sec = e ? atol(e) : 12; if (sec < 2) sec = 2;
                       QueryPerformanceFrequency(&ph_qf); ph_q00 = v1047_q0; ph_len = (long long)sec * ph_qf.QuadPart; }
        v1047_phase = (int)((v1047_q0.QuadPart - ph_q00.QuadPart) / ph_len);
        v1047_now = s_v1047 && !(v1047_phase & 1);
        g_v1047_fw_now = v1047_now;
    }
    const int v1047_tp = s_v1047_tp && v1047_now, v1047_tc = s_v1047_tc && v1047_now;

    /* Walk the FIFO exactly like the RSX: chase the guest-written `put` (an IO
     * offset in the control register), decoding method headers and following
     * JUMP/CALL/RET control words. The old linear drain copied context->current
     * onward and treated a JUMP as end-of-buffer -- fine for titles that write
     * one flat ring (cellmark), but PSL1GHT/Tiny3D immediately JUMPs into its
     * own command ring, so everything after the jump (including the reference
     * writes its waits spin on) silently never executed. */
    u32 put = vm_read32(GCM_CONTROL_GUEST_ADDR + 0);
    s_v1129w_walk_put = put; s_v1129w_walk_valid = 1;   /* v1129W GF_FIFO_KICK2: the put this walk has seen */
    { extern void rsx_ww_bump(void); rsx_ww_bump(); }   /* v219 VC_WW: data behind a new put may be newer than the last poll */
    /* The words behind `put` were stored by another thread (the title, or the
     * recycle below) before it stored `put`. On x86 the hardware keeps loads
     * in order; on arm64 it does not, so without an acquire here the ring
     * word can be fetched before `put` was and come back stale -- the walker
     * then runs off the end of the previous frame's bytes until it meets a
     * word that is not a command, and parks there forever. */
    atomic_thread_fence(memory_order_acquire);

    /* GCM_CTXDBG=1: which context the title is actually driving. Ours is written
     * once at init; if the title repoints gCellGcmCurrentContext at a buffer of
     * its own, every field below moves with it and the FIFO the runtime walks
     * stops being the one the title fills. Reported on change, not per tick. */
    static int s_ctxdbg = -1; if (s_ctxdbg < 0) s_ctxdbg = getenv("GCM_CTXDBG") ? 1 : 0;   /* v180: cached */
    if (s_ctxdbg && s_gcm_ctx_out_ea) {
        static u32 last_ptr = 0xFFFFFFFFu, last_cur = 0xFFFFFFFFu;
        u32 ptr = vm_read32(s_gcm_ctx_out_ea);
        u32 cur = ptr ? vm_read32(ptr + 0x8) : 0;
        if (ptr != last_ptr || cur != last_cur) {
            last_ptr = ptr; last_cur = cur;
            fprintf(stderr, "[CTX] gCellGcmCurrentContext@0x%08X -> 0x%08X  "
                    "begin=%08X end=%08X current=%08X callback=%08X  (ours=0x%08X)\n",
                    s_gcm_ctx_out_ea, ptr,
                    ptr ? vm_read32(ptr + 0x0) : 0, ptr ? vm_read32(ptr + 0x4) : 0,
                    cur, ptr ? vm_read32(ptr + 0xC) : 0, s_gcm_context_ea);
            fflush(stderr);
        }
    }

    u32 start_off = s_fifo_getoff;
    /* v243C RFBOUND: walker latency per guest put (the guest publishes a whole frame with one put) */
    static u32 s_v243c_put = 0xFFFFFFFFu; static int s_v243c_open = 0; static long long s_v243c_t0q = 0, s_v243c_busy = 0;
    LARGE_INTEGER v243c_q0 = {0};
    const int v243c_rb = v243c_rfbound_on();
    if (__builtin_expect(v243c_rb, 0)) {
        QueryPerformanceCounter(&v243c_q0);
        if (put != s_v243c_put && put != s_fifo_getoff) {
            if (!s_v243c_open) { s_v243c_open = 1; s_v243c_t0q = g_v243c_last_put_qpc ? g_v243c_last_put_qpc : v243c_q0.QuadPart; s_v243c_busy = 0; }
            s_v243c_put = put;
        }
    }
    static int s_draindbg = -1; if (s_draindbg < 0) s_draindbg = getenv("GCM_DRAINDBG") ? 1 : 0;   /* v180: cached */
    if (s_draindbg) {
        static int n = 0;
        if (n++ % 60 == 0)
            printf("[DRAIN] getoff=%08X put=%08X ref=%08X ctx.current=%08X\n",
                   s_fifo_getoff, put, vm_read32(GCM_CONTROL_GUEST_ADDR + 8),
                   vm_read32(s_gcm_context_ea + 0x8));
    }

    /* GCM_SCANREF=1 (one-shot): list every NV406E_SET_REFERENCE actually
     * present in 0..put. "The title waits for a fence we never published" has
     * two very different causes -- we missed the command, or it was never
     * submitted -- and only a direct scan of the ring separates them. */
    { static int scanned = 0;
      static int s_scanref = -1; if (s_scanref < 0) s_scanref = getenv("GCM_SCANREF") ? 1 : 0;   /* v180: cached */
      if (!scanned && s_scanref && put >= 0x19F0) { scanned = 1;
        unsigned found = 0;
        for (u32 io = 0; io + 8 <= put; io += 4) {
            u32 ea = gcm_io2ea(io); if (!ea) continue;
            u32 w = vm_read32(ea);
            if ((w & 0x1FFCu) == 0x0050u && (w >> 29) == 0) {
                u32 dea = gcm_io2ea(io + 4);
                fprintf(stderr, "[SCANREF] io=%08X w=%08X count=%u arg=%08X\n",
                        io, w, (w >> 18) & 0x7FFu, dea ? vm_read32(dea) : 0xDEADu);
                found++;
            }
        }
        fprintf(stderr, "[SCANREF] %u SET_REFERENCE in 0..%08X\n", found, put);
      } }

    /* GCM_FIFO_SNAP=N: dump N snapshots of the raw words at both ends of the
     * ring -- what `get` is about to decode, and what the title just wrote at
     * `put`. When get and put drift apart, the two windows say immediately
     * whether get is grinding through NOPs/garbage or the title is writing
     * somewhere get can never reach. The [FIFOSTEP] trace only ever fired on a
     * bad branch, which is exactly the case where the drift is silent. */
    { static int snap = -1;
      if (snap < 0) { const char* e = getenv("GCM_FIFO_SNAP"); snap = e ? atoi(e) : 0; }
      if (snap > 0 && put) {
          snap--;
          fprintf(stderr, "[SNAP] get=%08X put=%08X\n", s_fifo_getoff, put);
          for (int k = 0; k < 16; k++) {
              u32 gio = s_fifo_getoff + (u32)(k * 4), pio = put + (u32)(k * 4);
              u32 gea = gcm_io2ea(gio), pea = gcm_io2ea(pio);
              fprintf(stderr, "[SNAP]  get+%-3d io=%08X w=%08X   put+%-3d io=%08X w=%08X\n",
                      k * 4, gio, gea ? vm_read32(gea) : 0xDEADDEADu,
                      k * 4, pio, pea ? vm_read32(pea) : 0xDEADDEADu);
          }
          fflush(stderr);
      } }

    int budget = 0x100000;                    /* words per tick cap */
    /* GCM_DRAINDBG also reports WHY each pass stopped. A drain that ends far
     * short of `put` every tick is the signature of a FIFO that can never
     * catch up, and the reason is the whole diagnosis. */
    const char* why = "caught-up";
    u32 v1037_fol = s_v1037_follow;   /* v1037D */
    const u32 v1037_put = s_v1037_old_put, v1037_io = s_v1037_new_io;
    s_v1053_semwait = 0;   /* v1053A */
    while (s_fifo_getoff != put && budget-- > 0) {
        if (__builtin_expect(v1037_fol != 0, 0) && s_fifo_getoff == v1037_put) {
            static int n37 = 0;
            if (n37 < 16) { n37++;
                fprintf(stderr, "[V1037D] walker reached the pre-switch put 0x%08X -> continues at the default context's current 0x%08X (put 0x%08X)%c",
                        v1037_put, v1037_io, put, 10); }
            s_fifo_getoff = v1037_io; s_v1037_follow = 0; v1037_fol = 0;
            continue;
        }
        u32 ea = gcm_io2ea(s_fifo_getoff);
        if (!ea && v1053_abort_call("get on unmapped IO", s_fifo_getoff, 0)) continue;   /* v1053A */
        if (!ea) {
            /* get is sitting on an IO offset with no mapping -- the title
             * branched into a region and then unmapped it, or the offset was
             * never valid. Breaking silently (what this did) leaves get stuck
             * there FOREVER: it can never reach put, the FIFO stops draining,
             * no fences publish, and anything waiting on ctrl->ref hangs with
             * nothing logged. Resynchronise to put instead. Commands between
             * here and put are lost, which is bad, but a permanently dead FIFO
             * is worse -- and now it says so. */
            static u32 last_bad; static int n;
            if (s_fifo_getoff != last_bad || n < 4) {
                last_bad = s_fifo_getoff;
                if ((n < 16 && n++ < 16))
                    fprintf(stderr, "[cellGcmSys] FIFO get=0x%08X has no IO "
                            "mapping -- resyncing to put=0x%08X\n",
                            s_fifo_getoff, put);
            }
            s_fifo_getoff = put; s_v1037_follow = 0;   /* v1037D */
            if (v1053_on()) { s_fifo_calloff = 0; s_v1053_n[8]++; }   /* v1053A */
            why = "no-io-mapping";
            break;
        }
        u32 w = vm_read32(ea);
        if (__builtin_expect(s_v1047_st, 0)) {   /* v1047A stats: headers, method packets, their words */
            s_v1047_n[0]++;
            if ((w >> 29) == 0 || (w >> 29) == 2) { s_v1047_n[1]++; s_v1047_n[2] += (w >> 18) & 0x7FFu; }
        }
        /* Only remember offsets BELOW put -- those are positions in the
         * ring the title is writing. A park in another region is where the
         * walker gave up, not how far it got through the ring. */
        if (s_fifo_getoff < put) s_last_ring_off = s_fifo_getoff + 4;
        s_tr_off[s_tr_i % GCM_TRACE_N] = s_fifo_getoff;
        s_tr_w[s_tr_i % GCM_TRACE_N] = w;
        s_tr_i++;
        u32 type = w >> 29;

        if ((w & 0xFFFFFF00u) == GCM_FLIP_NOTE) {   /* v62: flip position marker (no-op for the RSX) */
            u32 id = w & 7u;
            if (s_display_buffer_set[id]) g_gcm_fifo_flip_off = s_display_buffers[id].offset;
            {   /* v1034F-2: count only notes this runtime wrote at exactly this offset */
                int fresh = 0, v1048_k = -1;
                for (int k = 0; k < 8; k++)
                    if (s_v1034_note_io[k] == s_fifo_getoff + 1u) { s_v1034_note_io[k] = 0; fresh = 1; v1048_k = k; break; }
                if (fresh) { g_gcm_fifo_flip_seq_fc = s_flip_request_count; g_gcm_fifo_flip_seq++; }
                else g_gcm_flip_note_stale++;
                if (fresh) {   /* v1049G GCM_FLIP_GATE: this request's commands are executed -> its flip may complete */
                    const u32 rq = s_v1048_note_req[v1048_k];
                    if ((int)(rq - s_v1049_walked_req) > 0) s_v1049_walked_req = rq;
                    if ((int)(rq - g_gcm_walked_req) > 0) g_gcm_walked_req = rq;   /* v1051P */
                    if (v1049_gate_on() && (int)(rq + s_v1049_lag_now - s_flip_request_count) >= 0 && s_flip_status == CELL_GCM_FLIP_STATUS_WAITING) {
                        s_v1049_n[0]++; GCM_PENDING_SET(2);
                    }
                }
                /* v1048D2 GCM_FLIP_ON_WALK: the walker passed the note of the flip the guest is waiting on -> done now */
                if (fresh && s_v1048_flip && s_v1048_note_req[v1048_k] == s_flip_request_count && v1048_flip_now()) {
                    if (s_flip_status == CELL_GCM_FLIP_STATUS_WAITING) s_v1048_cur.walkdone++;
                    cellGcmTickFlip();
                }
            }
            { static int once = 0; if (!once) { once = 1;
                fprintf(stderr, "[GCM] v62 ordered flips: first FIFO flip marker reached (buffer %u)%c", id, 10); } }
            s_fifo_getoff += 4;
            continue;
        }
        if ((w & 0xFFFF0000u) == 0xFEAD0000u) {
            /* lv1 driver flip: statically-linked libgcm cellGcmSetFlip writes
             * this word into the FIFO instead of calling any import. Fire the
             * flip (status/count/handler) and STOP this drain so the ticker's
             * flip-gated present shows exactly the completed frame -- draining
             * on would mix the next frame's head into the batch (wave: every
             * frame split across 4 presents, layout flashing/zooming). */
            extern s32 cellGcmSetFlipCommand(u32 bufferId);
            s_fifo_getoff += 4;
            /* Count the flip WORDS the title actually writes into the FIFO,
             * separately from the presents that result. When a title stops
             * appearing on screen the first question is which side stopped:
             * this counter still climbing while presents do not is a runtime
             * problem, both stopping together is the guest's. */
            { static unsigned long long flips = 0; static int dbg = -1;
              if (dbg < 0) dbg = getenv("GCM_FLIPCOUNT") ? 1 : 0;
              ++flips;
              if (dbg && (flips <= 8ull || (flips % 200ull) == 0))
                  fprintf(stderr, "[flipword] %llu FIFO flip words decoded%c",
                          flips, 10); }
            if ((w & 0xFFFFFF00u) == GCM_FLIP_MARKER) {
                /* Queued by _cellGcmSetFlipCommand, which already did the
                 * guest-visible half of the request. */
                s_current_display_buffer_id = w & 7u;
                if (s_display_buffer_set[w & 7u]) g_gcm_fifo_flip_off = s_display_buffers[w & 7u].offset;   /* v62 */
                s_flip_pending = 1;
                s_flip_request_count++;
                g_gcm_fifo_flip_seq_fc = s_flip_request_count; g_gcm_fifo_flip_seq++;   /* v1034F */
            } else {
                cellGcmSetFlipCommand(w & 0xFFu);
            }
            /* ...unless the FIFO is badly backlogged. One flip per drain is
             * right while `get` is keeping up with `put`; when it is megabytes
             * behind it is a deadlock, because the title's ring can only be
             * reclaimed as `get` advances. VF5 submits ~5x faster than one
             * flip per drain consumes, so its 4 MB ring fills, its own
             * buffer-full callback can never free space, and AMGL prints
             * "Command Buffer Overflow" forever. Past the threshold keep
             * draining -- the intermediate presents are frames we are too far
             * behind to show anyway. No-op for a title that is keeping up. */
            if (put < s_fifo_getoff ||
                put - s_fifo_getoff < GCM_FIFO_CATCHUP_BYTES)
                { why = "flip"; break; }
            continue;
        }

        if (type == 1) {                       /* JUMP: 0x20000000 | offset */
            { static int _rd = -1; if (_rd < 0) _rd = getenv("GCM_RECDBG") ? 1 : 0;
              if (_rd) fprintf(stderr, "[JMP] %08X -> %08X (put=%08X)\n",
                               s_fifo_getoff, w & 0x1FFFFFFCu, put); }
            { u32 tgt = w & 0x1FFFFFFCu;
              if (!gcm_io2ea(tgt) && v1053_abort_call("JUMP to unmapped IO", s_fifo_getoff, w)) continue;   /* v1053A */
              if (!gcm_io2ea(tgt)) { gcm_fifo_bad_branch("JUMP", tgt, w); gcm_fifo_dump_around(s_fifo_getoff); gcm_fifo_resync_why("unmapped-JUMP", &s_fifo_getoff, put); break; }
              /* v1053A: a list that JUMPs back into the ring window has left the subroutine for good -- a stale return slot
               * would later send an abort/RET back to an old ring position (re-walking fences). */
              if (s_fifo_calloff && tgt < s_config.ioSize && s_fifo_getoff >= s_config.ioSize && v1053_on()) s_fifo_calloff = 0;
              /* A JUMP to its own address is the "park the RSX here" idiom:
               * the title leaves it at the write head so the GPU stops if it
               * catches up, and overwrites it when the next segment is
               * appended. Taking it re-reads the same word forever -- 38.8 M
               * times in one Twisted Metal run, which is the drain burning the
               * CPU the guest needs. Stop this pass instead; the next one
               * re-reads the word, so a patched jump is still followed. */
              if (tgt == s_fifo_getoff) {
                  /* Parked. If the title has meanwhile moved `put` somewhere
                   * else, it has switched to its other segment and left this
                   * park standing -- it only patches a park when it reuses that
                   * block. Waiting for a patch that will not come freezes the
                   * FIFO for the rest of the run, so follow the write head. */
                  if (put != s_fifo_getoff) { gcm_fifo_dump_around(s_fifo_getoff);
                                              gcm_fifo_resync_why("jump-park", &s_fifo_getoff, put); }
                  break;
              }
              s_fifo_getoff = tgt; }
            continue;
        }
        if ((w & 3) == 2) {                    /* CALL: offset | 2 */
            { u32 tgt = w & 0x1FFFFFFCu;
              if (!gcm_io2ea(tgt) && v1053_abort_call("CALL to unmapped IO", s_fifo_getoff, w)) continue;   /* v1053A */
              if (!gcm_io2ea(tgt)) { gcm_fifo_bad_branch("CALL", tgt, w); gcm_fifo_resync_why("unmapped-CALL", &s_fifo_getoff, put); break; }
              if (s_fifo_calloff && v1053_on()) s_v1053_n[1]++;   /* v1053A: counted only (RSX has one return slot) */
              s_v1053_call_tgt = tgt;
              s_fifo_calloff = s_fifo_getoff + 4;
              s_fifo_getoff  = tgt; }
            continue;
        }
        if (w == 0x00020000u) {                /* RET */
            /* v1053A: a RET with no CALL used to "return" to IO 0 and walk a stale ring lap from there */
            if (!s_fifo_calloff && v1053_on()) {
                s_v1053_n[2]++;
                { static int nr = 0; if (nr < 8) { nr++; fprintf(stderr, "[V1053A] RET without a CALL at IO 0x%08X -> resync (was: continue at IO 0)%c", s_fifo_getoff, 10); } }
                gcm_fifo_resync_why("ret-without-call", &s_fifo_getoff, put);
                break;
            }
            s_fifo_getoff = s_fifo_calloff;
            s_fifo_calloff = 0;
            continue;
        }
        if (type == 0 || type == 2) {          /* method (incrementing / NI) */
            u32 count  = (w >> 18) & 0x7FF;
            u32 method = w & 0x1FFCu;
            u32 subch  = (w >> 13) & 7;
            /* v1052B GCM_GARBAGE_GUARD (default 1; user 18:04: "the screen turned completely black"): after a FIFO desync
             * the walker read plain game data as commands -- "[RSX-UNK] method 0x0000 ... 0x00F0" (floats, pointers,
             * zeros) until an unmapped JUMP stopped it -- and every later frame rendered black: those garbage writes
             * hit set-once render state. No real command stream writes more than a handful of registers starting below
             * 0x0100 (object / reference / semaphore methods), so such a header is garbage: resync BEFORE executing it.
             * Log "[V1052B]". GCM_GARBAGE_GUARD=0 = old. */
            { static int gg = -1; if (gg < 0) { const char* e = getenv("GCM_GARBAGE_GUARD"); gg = (e && *e == '0') ? 0 : 1; }
              if (gg && method < 0x0100u && count > 16u) {
                  static unsigned ng = 0;
                  if (v1053_abort_call("garbage packet", s_fifo_getoff, w)) continue;   /* v1053A: drop the list, not the frame */
                  if (ng++ < 20) fprintf(stderr, "[V1052B] garbage packet at IO 0x%08X (header 0x%08X: method 0x%04X x %u) -> resync before executing it (#%u)%c",
                                         s_fifo_getoff, w, method, count, ng, 10);
                  gcm_fifo_resync_why("garbage-packet", &s_fifo_getoff, put);
                  break;
              } }
            /* v65 FAST PATH: vertex-program constant uploads (SET_TRANSFORM_CONSTANT_LOAD +
             * SET_TRANSFORM_CONSTANT[0..31], 0x1F00-0x1F7C; 0x1F80+ are other methods, e.g. 0x1FC0/0x1FF0) are ~57% of every method in the city (~200k per
             * frame). The generic path below spends ~a dozen debug-switch checks per word and
             * then walks rsx_process_method's if-chain to reach the same 3-line store. Whole
             * incrementing block, one IO page: store straight into the state, exactly as
             * rsx_process_method does. GCM_TC_FAST=0 off (also off with TCONST_DBG/SEQ_DBG). */
            { static int tcf = -1;
              if (tcf < 0) { const char* e = getenv("GCM_TC_FAST"); tcf = (e && *e == '0') ? 0 : 1;
                             if (getenv("TCONST_DBG") || getenv("SEQ_DBG") || getenv("LOAD_DBG") || /* v180: RSX_MHIST no longer disables it (histogram = generic methods only) */
                                 getenv("RSX_TRACE") || getenv("TCONST_FIFO") || getenv("GCM_VAL_WATCH") ||
                                 rsx_live_draw_enabled() || rsx_draw_engine_enabled()) tcf = 0; }
              if (tcf && type == 0 && subch == 0 && count && method >= 0x1EFCu && method + 4u * (count - 1u) < 0x1F80u) {
                  const u32 io0 = s_fifo_getoff + 4u, io1 = s_fifo_getoff + 4u * count;
                  const u32 ea0 = gcm_io2ea(io0);
                  if (ea0 && (io0 >> 20) == (io1 >> 20)) {
                      extern uint8_t* vm_base;
                      const u8* p = vm_base + ea0;
                      if (__builtin_expect(s_v1047_st, 0)) s_v1047_n[3] += count;   /* v1047A stats: TC block words */
                      /* v1047A GCM_FASTWALK (mask 0x4): the per-word loop below as one block store. The constant words
                       * of a block land in CONSECUTIVE floats of vertex_constants (slot = load + reg/4, lane = reg%4:
                       * float index load*4 + reg), so byte-swap-copy them and widen the dirty range once to the same
                       * [first slot, last slot] the per-word loop produces; words past slot 511 are dropped as there.
                       * A load >= 512 (u32 slot wrap-around in the per-word loop) takes the per-word loop. Same maths
                       * as the AVX2 TC_VEC path above, which this build does not compile (no -mavx2 for the runtime). */
                      if (v1047_tc) {
                          u32 i0 = 0, ld = s_state.transform_constant_load;
                          if (method == 0x1EFCu) { ld = gcm_read_be32_direct(p); i0 = 1; }
                          if (ld < RSX_MAX_VERTEX_CONSTANTS) {
                              s_state.transform_constant_load = ld;
                              const u32 r0 = (method + 4u * i0 - 0x1F00u) >> 2;
                              const u32 base = ld * 4u + r0;
                              u32 n = count - i0;
                              if (base >= RSX_MAX_VERTEX_CONSTANTS * 4u) n = 0;
                              else if (base + n > RSX_MAX_VERTEX_CONSTANTS * 4u) n = RSX_MAX_VERTEX_CONSTANTS * 4u - base;
                              if (n) {
                                  u32* dst = (u32*)&s_state.vertex_constants[0][0] + base;
                                  const u8* sp = p + 4u * i0;
                                  u32 k = 0;
#ifdef GCM_V1047_SSSE3
                                  { const __m128i sw = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
                                    for (; k + 4u <= n; k += 4u)
                                        _mm_storeu_si128((__m128i*)(dst + k), _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(sp + 4u * k)), sw)); }
#endif
                                  for (; k < n; k++) dst[k] = gcm_read_be32_direct(sp + 4u * k);
                                  const u32 s0 = base >> 2, s1 = (base + n - 1u) >> 2;
                                  if (!s_state.vertex_constants_dirty) {
                                      s_state.vertex_constants_lo = s0; s_state.vertex_constants_hi = s1;
                                      s_state.vertex_constants_dirty = 1;
                                  } else {
                                      if (s0 < s_state.vertex_constants_lo) s_state.vertex_constants_lo = s0;
                                      if (s1 > s_state.vertex_constants_hi) s_state.vertex_constants_hi = s1;
                                  }
                              }
                              s_fifo_getoff += 4u + count * 4u;
                              continue;
                          }
                      }
                      /* v221C TC_VEC (TC_VEC=1 = on, default off until b029 is measured; ABX 0x100000 = off): the
                       * constant words of one block land in CONSECUTIVE floats of vertex_constants (row-major [512][4],
                       * slot = load + reg/4, lane = reg%4) -> byte-swap-copy them 8 at a time and update the dirty
                       * range once (same min/max the per-word loop produces; words past slot 511 are dropped as there). */
#ifdef GCM_TC_VEC_X86
                      { static int tv = -1; extern volatile int g_abx;
                        if (tv < 0) { const char* e = getenv("TC_VEC"); tv = (e && *e == '1') ? 1 : 0;
                            fprintf(stderr, "[TC_VEC] v221C vectorised transform-constant block store: %s (TC_VEC=1 = on; ABX 0x100000 = off)%c", tv ? "ON" : "off", 10); }
                        if (tv && !(g_abx & 0x100000)) {
                            u32 i0 = 0, ld = s_state.transform_constant_load;
                            if (method == 0x1EFCu) { u32 raw; memcpy(&raw, p, 4); ld = ps3_bswap32(raw); i0 = 1; }
                          if (ld < RSX_MAX_VERTEX_CONSTANTS) {   /* else: the per-word loop below (u32 slot wrap-around) */
                            s_state.transform_constant_load = ld;
                            const u32 r0 = (method + 4u * i0 - 0x1F00u) >> 2;   /* first register of the block */
                            const u32 base = s_state.transform_constant_load * 4u + r0;   /* float index */
                            u32 n = count - i0;
                            if (base >= RSX_MAX_VERTEX_CONSTANTS * 4u) n = 0;
                            else if (base + n > RSX_MAX_VERTEX_CONSTANTS * 4u) n = RSX_MAX_VERTEX_CONSTANTS * 4u - base;
                            if (n) {
                                u32* dst = (u32*)&s_state.vertex_constants[0][0] + base;
                                const u8* sp = p + 4u * i0;
                                const __m256i sw = _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                                                                    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
                                u32 k = 0;
                                for (; k + 8u <= n; k += 8u)
                                    _mm256_storeu_si256((__m256i*)(dst + k), _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i*)(sp + 4u * k)), sw));
                                for (; k < n; k++) { u32 raw; memcpy(&raw, sp + 4u * k, 4); dst[k] = ps3_bswap32(raw); }
                                const u32 s0 = base >> 2, s1 = (base + n - 1u) >> 2;
                                if (!s_state.vertex_constants_dirty) {
                                    s_state.vertex_constants_lo = s0; s_state.vertex_constants_hi = s1;
                                    s_state.vertex_constants_dirty = 1;
                                } else {
                                    if (s0 < s_state.vertex_constants_lo) s_state.vertex_constants_lo = s0;
                                    if (s1 > s_state.vertex_constants_hi) s_state.vertex_constants_hi = s1;
                                }
                            }
                            s_fifo_getoff += 4u + count * 4u;
                            continue;
                          }
                        } }
#endif
                      for (u32 i = 0; i < count; i++) {
                          u32 raw; memcpy(&raw, p + 4u * i, 4);
                          const u32 d = ps3_bswap32(raw);
                          const u32 m = method + 4u * i;
                          if (m == 0x1EFCu) { s_state.transform_constant_load = d; continue; }
                          const u32 reg = (m - 0x1F00u) >> 2;
                          const u32 slot = s_state.transform_constant_load + (reg >> 2), lane = reg & 3u;
                          if (slot < RSX_MAX_VERTEX_CONSTANTS) {
                              float f; memcpy(&f, &d, 4);
                              s_state.vertex_constants[slot][lane] = f;
                              if (!s_state.vertex_constants_dirty) {
                                  s_state.vertex_constants_lo = slot; s_state.vertex_constants_hi = slot;
                                  s_state.vertex_constants_dirty = 1;
                              } else {
                                  if (slot < s_state.vertex_constants_lo) s_state.vertex_constants_lo = slot;
                                  if (slot > s_state.vertex_constants_hi) s_state.vertex_constants_hi = slot;
                              }
                          }
                      }
                      s_fifo_getoff += 4u + count * 4u;
                      continue;
                  }
              } }
            /* TCONST_FIFO=1: how the title actually streams vertex constants.
             * NV4097_SET_TRANSFORM_CONSTANT is a 32-register window; if a block
             * arrives NON-INCREMENTING then every dword carries the same method
             * and any handler that derives the constant slot from the method
             * offset writes them all on top of each other. */
            { static int tf = -1;
              if (tf < 0) { const char* e = getenv("TCONST_FIFO"); tf = e ? atoi(e) : 0; }
              if (tf && subch == 0) {
                  static u32 hist[2048]; static u32 tot[2048]; static int dumped = 0;
                  u32 mi = (method >> 2) & 2047;
                  hist[mi]++; tot[mi] += count;
                  static u64 seen = 0;
                  if (++seen == 200000ull && !dumped) { dumped = 1;
                      fprintf(stderr, "[TCFIFO] method histogram (blocks, dwords):%c", 10);
                      for (u32 k = 0; k < 2048; k++) if (hist[k])
                          fprintf(stderr, "[TCFIFO]   0x%04X  %8u blocks %10u dwords%c",
                                  k << 2, hist[k], tot[k], 10);
                  }
              } }
            /* NV406E semaphore: OFFSET(0x64) / ACQUIRE(0x68) / RELEASE(0x6C).
             *
             * These were falling through to rsx_process_method as "unknown method" --
             * i.e. no-ops. A title that syncs with cellGcmSetFlipCommandWithWaitLabel and
             * cellGcmGetLabelAddress (Twisted Metal does both) then waits on a label the
             * RSX is supposed to write and never sees it move, so it never appends its
             * next FIFO segment and the GPU sits on a park forever.
             *
             * RELEASE writes the value; ACQUIRE stops this drain pass without consuming
             * the method, so the next pass re-reads it -- which is what the hardware does
             * while it waits. Semaphore offsets index the same label window
             * cellGcmGetLabelAddress hands out. */
            if (subch == 0 && (method == 0x64u || method == 0x68u || method == 0x6Cu)) {
                int sem_blocked = 0;
                for (u32 i = 0; i < count; i++) {
                    u32 dea = gcm_io2ea(s_fifo_getoff + 4 + i * 4);
                    if (!dea) break;
                    u32 m = (type == 0) ? method + i * 4 : method;
                    u32 v = vm_read32(dea);
                    u32 la = GCM_LABEL_GUEST_BASE + (s_sema_offset & 0xFFFFu);
                    { static int sn = 0; static int rd = -1; if (rd < 0) rd = getenv("GCM_RECDBG") ? 1 : 0; if (rd && (sn < 12 && sn++ < 12))
                        fprintf(stderr, "[SEMA] m=0x%02X v=0x%08X off=0x%X\n", m, v, s_sema_offset); }
                    if (m == 0x68u) { static unsigned long s_acq = 0; if (((++s_acq) & 0x3FFFu) == 1)   /* v173 diag */
                        fprintf(stderr, "[SEMA] NV406E acquire #%lu: label 0x%X = 0x%08X, want 0x%08X%c", s_acq, la, vm_read32(la), v, 10); }
                    if (m == 0x68u) { extern void rsx_ww_bump(void); rsx_ww_bump(); }   /* v219 VC_WW: SPU-released data */
                    if (m == 0x64u)      s_sema_offset = v;
                    else if (m == 0x6Cu) { extern void (*g_rsx_vtx_sync)(void); if (g_rsx_vtx_sync) g_rsx_vtx_sync(); vm_write32(la, v); }
                    else if (m == 0x68u && vm_read32(la) != v) {
            /* GCM_SEMA_ACQUIRE=1 makes ACQUIRE actually block, which is what
             * the hardware does. Off by default: a title whose label nothing
             * ever writes would wedge the drain permanently, and RELEASE on
             * its own is the half that unblocks a waiting guest. */
            /* v50: default ON with a timeout. inFamous makes the RSX wait on a label
             * that the SPU skinning jobs release; without the wait, characters were
             * drawn from half-skinned vertex buffers (missing knees/forearms, see-
             * through NPC limbs, flicker). Not blocking a thread: this drain pass just
             * stops and the next one re-checks. A label that never arrives within
             * GCM_SEMA_TIMEOUT ms (default 50) is skipped as before, with a log line.
             * GCM_SEMA_ACQUIRE=0 restores the never-wait behaviour. */
            static int blk = -1; static long tmo = 50;
            static u32 w_get = 0xFFFFFFFFu; static clock_t w_t0 = 0;
            static unsigned long n_wait = 0, n_tmo = 0;
            if (blk < 0) { const char* e = getenv("GCM_SEMA_ACQUIRE"); blk = e ? atoi(e) : 1;
                const char* t = getenv("GCM_SEMA_TIMEOUT"); if (t && *t) tmo = atol(t);
                fprintf(stderr, "[SEMA] v50 acquire waits %s (timeout %ld ms)%c", blk ? "ON" : "OFF", tmo, 10); }
            if (blk) {
                clock_t now = clock();
                if (w_get != s_fifo_getoff) { w_get = s_fifo_getoff; w_t0 = now; n_wait++;
                    if (n_wait <= 6 || (n_wait % 20000) == 0)
                        fprintf(stderr, "[SEMA] acquire #%lu waits: label 0x%X has 0x%08X, want 0x%08X (%lu timeouts)%c",
                                n_wait, la, vm_read32(la), v, n_tmo, 10); }
                /* short spin first: the SPU jobs run on their own host threads and
                 * usually release within a fraction of a ms; breaking out costs a
                 * whole present-thread idle tick */
                { LARGE_INTEGER f, a, b; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&a);
                  do { if (vm_read32(la) == v) break; SwitchToThread(); QueryPerformanceCounter(&b); }
                  while ((b.QuadPart - a.QuadPart) * 1000000LL / f.QuadPart < 1500);
                  if (__builtin_expect(v243c_rfbound_on(), 0)) {   /* v243C RFBOUND: walker time spent on SPU labels */
                      QueryPerformanceCounter(&b); const long long d = b.QuadPart - a.QuadPart;
                      g_v243c_st[4] += d; g_v243c_st[6]++;
                      if (a.QuadPart - g_v243c_st[7] < f.QuadPart / 5000) g_v243c_st[5] += d; } }
                if (vm_read32(la) == v) { w_get = 0xFFFFFFFFu; continue; }
                /* v1049S GCM_SEMA_PASSED=1 (default 0 = old; Godfather's launchers 1): a counter label that is already
                 * PAST the wanted value (by < 64k, wrap-aware) was released earlier -- the producer ran ahead of the
                 * walker. Waiting can only end in the 50 ms timeout and the same skip, so take it now. */
                { static int sp = -1; static unsigned long n_pass = 0;
                  if (sp < 0) { const char* e = getenv("GCM_SEMA_PASSED"); sp = (e && *e == '1') ? 1 : 0;
                      fprintf(stderr, "[V1049S] semaphore acquire on a label already past the value proceeds at once: %s (GCM_SEMA_PASSED)%c", sp ? "ON" : "off", 10); }
                  const u32 cur_ = vm_read32(la);
                  if (sp && (int)(cur_ - v) > 0 && cur_ - v < 0x10000u) {
                      if (((++n_pass) & 0xFFu) == 1)
                          fprintf(stderr, "[V1049S] acquire #%lu passed: label 0x%X = 0x%08X, want 0x%08X%c", n_pass, la, cur_, v, 10);
                      w_get = 0xFFFFFFFFu; continue;
                  } }
                if ((long)(now - w_t0) * 1000L / (long)CLOCKS_PER_SEC < tmo) { sem_blocked = 1; s_v1053_semwait = 1; break; }
                n_tmo++;
                if (n_tmo <= 8) fprintf(stderr, "[SEMA] acquire timed out after %ld ms: label 0x%X = 0x%08X, want 0x%08X -- skipped%c",
                                        tmo, la, vm_read32(la), v, 10);
                w_get = 0xFFFFFFFFu;
            }
        }
                }
                if (sem_blocked) break;
                s_fifo_getoff += 4 + count * 4;
                continue;
            }
            /* v180 GCM_GEN_FAST (default 1; 0 = off): ordinary subchannel-0 method blocks (~150k words/frame in
             * the city after the v65 constant fast path) skip the per-word debug switches below (GCM_SUBCH7,
             * GCM_OBJDBG, GCM_SUBCH_GATE, GCM_VAL_WATCH, live/dispatch engines, RSX_LIVE_FEED_DBG -- any of them
             * set turns this path off) and the per-word IO lookup: one lookup per block when the block's data
             * sits in one 1 MB IO page. Per word it does exactly what the subch-0 branch below does:
             * rsx_process_method, the SET_REFERENCE fence push, the GET_REPORT answer. */
            { static int gf = -1;
              if (gf < 0) { const char* e = getenv("GCM_GEN_FAST"); gf = (e && *e == '0') ? 0 : 1;
                  if (getenv("GCM_SUBCH7") || getenv("GCM_OBJDBG") || getenv("GCM_SUBCH_GATE") || getenv("GCM_VAL_WATCH") ||
                      getenv("RSX_LIVE_FEED_DBG") || rsx_live_draw_enabled() || rsx_draw_engine_enabled()) gf = 0;
                  fprintf(stderr, "[GCM] v180 generic method fast path %s (GCM_GEN_FAST)%c", gf ? "ON" : "off", 10); }
              extern volatile int g_abx;   /* ABX 0x2000 = old path (in-run A/B) */
              if (gf && !(g_abx & 0x2000) && subch == 0 && count) {
                  const u32 io0 = s_fifo_getoff + 4u, io1 = s_fifo_getoff + 4u * count;
                  const u32 ea0 = gcm_io2ea(io0);
                  if (ea0 && (io0 >> 20) == (io1 >> 20)) {
                      /* v228 GCM_FIFO_DIRECT: ea0 has already been translated and the
                       * complete packet was proved to live in that same mapped IO page.
                       * Calling vm_read32 for every argument redundantly repeats the VM
                       * bounds test, diagnostic-mode branch and raw-SPU-range test for
                       * roughly 150k FIFO words per city frame.  Read the already-mapped
                       * bytes directly, exactly like the transform-constant fast path
                       * above.  memcpy keeps the unaligned load defined; the byte swap is
                       * identical to vm_read32's fast path.  A configurable ABX bit lets
                       * a benchmark alternate this exact path by PS3_AB phase without
                       * consuming another bit in the already-crowded signed ABX mask.
                       */
                      static int direct = -1;
                      static int direct_phase = 0;
                      if (direct < 0) {
                          const char* e = getenv("GCM_FIFO_DIRECT");
                          const char* p = getenv("GCM_FIFO_DIRECT_PHASE");
                          direct = (e && *e == '0') ? 0 : 1;
                          direct_phase = (p && *p == '1') ? 1 : 0;
                          fprintf(stderr, "[GCM] v228 direct mapped FIFO argument reads %s (GCM_FIFO_DIRECT=0 off%s)%c",
                                  direct ? "ON" : "off",
                                  direct_phase ? "; GCM_FIFO_DIRECT_PHASE: even=new, odd=old" : "", 10);
                      }
                      extern uint8_t* vm_base;
                      extern volatile int g_ab_phase;
                      const u8* const fifo_args = vm_base + ea0;
                      const int direct_now = direct && (!direct_phase || !(g_ab_phase & 1));
#define GCM_FIFO_ARG(_i) \
    (direct_now ? \
        gcm_read_be32_direct(fifo_args + 4u * (_i)) : \
        vm_read32(ea0 + 4u * (_i)))
                      /* v231 GCM_INDEX_SETUP: libgcm emits SET_INDEX_ARRAY_ADDRESS
                       * and SET_INDEX_ARRAY_DMA as one two-word increasing packet.
                       * Both rsx_process_method handlers are pure assignments, but
                       * the generic loop paid two full function/diagnostic dispatches
                       * per indexed draw.  Consume that exact packet here after the
                       * same one-page proof/direct reads as the generic path. */
                      { static int idx_setup = -1, idx_setup_phase = 0;
                        if (__builtin_expect(idx_setup < 0, 0)) {
                            const char* e = getenv("GCM_INDEX_SETUP");
                            const char* p = getenv("GCM_INDEX_SETUP_PHASE");
                            idx_setup = gf_fifo_fast || (e && *e == '1');
                            idx_setup_phase = (p && *p == '1') ? 1 : 0;
                            const char* pd = getenv("RSX_PREDISPATCH");
                            if (getenv("RSX_TRACE") || getenv("IMM_MTRACE") || getenv("RSX_MHIST") ||
                                (pd && *pd == '0')) idx_setup = 0;
                            fprintf(stderr, "[GCM] v231 paired index-array setup %s (GCM_INDEX_SETUP=1 on%s)%c",
                                    idx_setup ? "ON" : "off",
                                    idx_setup_phase ? "; phase even=new, odd=old" : "", 10);
                        }
                        extern volatile int g_ab_phase;
                        const int idx_setup_now = idx_setup &&
                            (!idx_setup_phase || !(g_ab_phase & 1));
                        if (idx_setup_now && type == 0 &&
                            method == NV4097_SET_INDEX_ARRAY_ADDRESS && count == 2u) {
                            extern void rsx_v1047_sj_flush(void);
                            rsx_v1047_sj_flush();
                            s_state.index_array_offset = GCM_FIFO_ARG(0);
                            s_state.index_array_dma = GCM_FIFO_ARG(1);
                            s_fifo_getoff += 12u;
                            continue;
                        }
                      }
                      /* v245E GCM_STATE_PACKET: the title submits vertex-array and
                       * texture state as incrementing packets.  The generic loop
                       * re-enters rsx_process_method for every word, where each word
                       * repeats the method histogram/trace gates and walks the main
                       * dispatch chain before reaching one of two pure assignment
                       * helpers.  Consume a packet here after the same complete-page
                       * proof used by GCM_FIFO_DIRECT.
                       *
                       * This deliberately covers only ranges whose old handlers have
                       * no callbacks, synchronization or ordering side effects.  Dirty
                       * flags are monotonic until BEGIN_END flushes them, so setting a
                       * range's flag once after the loop is identical to setting it on
                       * every word.  All method-observation modes force the old path.
                       * GCM_STATE_PACKET=0 is an immediate fallback; PHASE=1 gives an
                       * even=new / odd=old in-run A/B without consuming an ABX bit. */
                      { static int state_packet = -1, state_packet_phase = 0;
                        if (__builtin_expect(state_packet < 0, 0)) {
                            const char* e = getenv("GCM_STATE_PACKET");
                            const char* p = getenv("GCM_STATE_PACKET_PHASE");
                            state_packet = gf_fifo_fast || (e && *e == '1');
                            state_packet_phase = (p && *p == '1') ? 1 : 0;
                            if (getenv("RSX_TRACE") || getenv("IMM_MTRACE") ||
                                getenv("RSX_MHIST") || getenv("TCONST_DBG") ||
                                getenv("SEQ_DBG")) state_packet = 0;
                            fprintf(stderr, "[GCM] v245E state-packet dispatch %s (GCM_STATE_PACKET=1 on%s)%c",
                                    state_packet ? "ON" : "off",
                                    state_packet_phase ? "; phase even=new, odd=old" : "", 10);
                        }
                        const int state_packet_now = state_packet &&
                            (!state_packet_phase || !(g_ab_phase & 1));
                        if (state_packet_now && type == 0) {
                            const u32 last_method = method + 4u * (count - 1u);
                            if (method >= NV4097_SET_VERTEX_DATA_ARRAY_OFFSET &&
                                last_method < NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + RSX_MAX_VERTEX_ATTRIBS * 4u) {
                                extern void rsx_v1047_sj_flush(void);
                                rsx_v1047_sj_flush();
                                u32 index = (method - NV4097_SET_VERTEX_DATA_ARRAY_OFFSET) >> 2;
                                for (u32 i = 0; i < count; i++)
                                    s_state.vertex_attribs[index + i].offset = GCM_FIFO_ARG(i);
                                s_state.vertex_dirty = 1;
                                s_fifo_getoff += 4u + count * 4u;
                                continue;
                            }
                            if (method >= NV4097_SET_VERTEX_DATA_ARRAY_FORMAT &&
                                last_method < NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + RSX_MAX_VERTEX_ATTRIBS * 4u) {
                                extern void rsx_v1047_sj_flush(void);
                                rsx_v1047_sj_flush();
                                u32 index = (method - NV4097_SET_VERTEX_DATA_ARRAY_FORMAT) >> 2;
                                for (u32 i = 0; i < count; i++) {
                                    const u32 d = GCM_FIFO_ARG(i);
                                    rsx_vertex_attrib* const a = &s_state.vertex_attribs[index + i];
                                    a->type = d & 0xFu;
                                    a->size = (d >> 4) & 0xFu;
                                    a->stride = (d >> 8) & 0xFFu;
                                    a->frequency = (d >> 16) & 0xFFFFu;
                                    a->enabled = (a->size != 0);
                                    a->format = d;
                                }
                                if (gf_fifo_fast2) {
                                    extern void rsx_v1129g_vertex_layout_changed(void);
                                    rsx_v1129g_vertex_layout_changed();
                                }
                                s_state.vertex_dirty = 1;
                                s_fifo_getoff += 4u + count * 4u;
                                continue;
                            }
                            if (method >= NV4097_SET_TEXTURE_OFFSET &&
                                last_method < NV4097_SET_TEXTURE_OFFSET + RSX_MAX_TEXTURES * 0x20u) {
                                extern void rsx_v1047_sj_flush(void);
                                rsx_v1047_sj_flush();
                                for (u32 i = 0; i < count; i++) {
                                    const u32 m = method + i * 4u;
                                    const u32 base = m - NV4097_SET_TEXTURE_OFFSET;
                                    const u32 unit = base >> 5;
                                    const u32 reg = base & 0x1Fu;
                                    const u32 d = GCM_FIFO_ARG(i);
                                    rsx_texture_state* const t = &s_state.textures[unit];
                                    switch (reg) {
                                    case 0x00: t->offset = d; break;
                                    case 0x04: t->format = d; break;
                                    case 0x08: t->address = d; break;
                                    case 0x0C: t->control0 = d; break;
                                    case 0x10: t->control1 = d; break;
                                    case 0x14: t->filter = d; break;
                                    case 0x18: t->image_rect = d; break;
                                    case 0x1C: t->border_color = d; break;
                                    }
                                    t->dirty = 1;
                                }
                                s_state.texture_dirty = 1;
                                s_fifo_getoff += 4u + count * 4u;
                                continue;
                            }
                            /* v246B GCM_STATE_PACKET2=1 (default off; only acts where GCM_STATE_PACKET
                             * is active, so it shares GCM_STATE_PACKET_PHASE: even=new, odd=old): two more
                             * pure-assignment ranges from the perf181 [mhist] histogram.
                             *  - SET_TRANSFORM_PROGRAM 0x0B80..0x0BFC (17.6 % of generic words): the
                             *    rsx_process_method m_transform_program body, verbatim, per word in FIFO
                             *    order (incl. the per-word v64 g_rsx_vp_gen bump and vp_dirty).
                             *  - SET_TEXTURE_CONTROL3 0x1840..0x187C (2.3 %): process_texture_control3.
                             * Packets must lie wholly inside one range; anything else falls through. */
                            { static int sp2 = -1;
                              if (__builtin_expect(sp2 < 0, 0)) {
                                  const char* e2 = getenv("GCM_STATE_PACKET2");
                                  sp2 = gf_fifo_fast || (e2 && *e2 == '1');
                                  fprintf(stderr, "[GCM] v246B state-packet ranges 2 (transform program, texture control3) %s (GCM_STATE_PACKET2=1 on)%c",
                                          sp2 ? "ON" : "off", 10);
                              }
                              if (sp2) {
                                  if (method >= NV4097_SET_TRANSFORM_PROGRAM &&
                                      last_method < NV4097_SET_TRANSFORM_PROGRAM + 32u * 4u) {
                                      extern volatile u32 g_rsx_vp_gen;
                                      extern void rsx_v1047_sj_flush(void);
                                      rsx_v1047_sj_flush();
                                      for (u32 i = 0; i < count; i++) {
                                          const u32 d = GCM_FIFO_ARG(i);
                                          const u32 w = s_state.vp_ucode_write;
                                          if (w + 4 <= sizeof(s_state.vp_ucode)) {
                                              { u32 old; memcpy(&old, &s_state.vp_ucode[w], 4);
                                                if (old != d || w + 4 > s_state.vp_ucode_bytes) g_rsx_vp_gen++; }
                                              s_state.vp_ucode[w+0] = (u8)(d);
                                              s_state.vp_ucode[w+1] = (u8)(d >> 8);
                                              s_state.vp_ucode[w+2] = (u8)(d >> 16);
                                              s_state.vp_ucode[w+3] = (u8)(d >> 24);
                                              s_state.vp_ucode_write = w + 4;
                                              if (w + 4 > s_state.vp_ucode_bytes) s_state.vp_ucode_bytes = w + 4;
                                              s_state.vp_dirty = 1;
                                          }
                                      }
                                      s_fifo_getoff += 4u + count * 4u;
                                      continue;
                                  }
                                  if (method >= NV4097_SET_TEXTURE_CONTROL3 &&
                                      last_method < NV4097_SET_TEXTURE_CONTROL3 + RSX_MAX_TEXTURES * 4u) {
                                      extern void rsx_v1047_sj_flush(void);
                                      rsx_v1047_sj_flush();
                                      const u32 unit0 = (method - NV4097_SET_TEXTURE_CONTROL3) >> 2;
                                      for (u32 i = 0; i < count; i++) {
                                          s_state.textures[unit0 + i].control3 = GCM_FIFO_ARG(i);
                                          s_state.textures[unit0 + i].dirty = 1;
                                      }
                                      s_state.texture_dirty = 1;
                                      s_fifo_getoff += 4u + count * 4u;
                                      continue;
                                  }
                              }
                            }
                        }
                      }
                      /* v1047A GCM_FASTWALK (mask 0x2): an incrementing SET_TRANSFORM_PROGRAM packet (0x0B80..0x0BFC,
                       * <= 32 words of vertex-program microcode) as one block. rsx_process_method's m_transform_program
                       * body per word, in order: g_rsx_vp_gen += 1 for each word that changes the stored word or extends
                       * vp_ucode_bytes (same final value), word stored host-endian at the write cursor, cursor/byte count
                       * advanced, vp_dirty. Only when the whole packet fits the 16 KB store (else the per-word path with
                       * its per-word bound check). A pending STRIP_JOIN run is flushed first, as rsx_process_method
                       * does for every non-DRAW_INDEX_ARRAY method. Off with RSX_TRACE/IMM_MTRACE/RSX_MHIST/TCONST_DBG/
                       * SEQ_DBG (method observers). */
                      if (v1047_tp && type == 0 && method >= NV4097_SET_TRANSFORM_PROGRAM &&
                          method + 4u * (count - 1u) < NV4097_SET_TRANSFORM_PROGRAM + 32u * 4u &&
                          s_state.vp_ucode_write <= (u32)sizeof(s_state.vp_ucode) &&
                          4u * count <= (u32)sizeof(s_state.vp_ucode) - s_state.vp_ucode_write) {
                          extern void rsx_v1047_sj_flush(void);
                          extern volatile u32 g_rsx_vp_gen;
                          rsx_v1047_sj_flush();
                          const u32 w0 = s_state.vp_ucode_write, bytes0 = s_state.vp_ucode_bytes;
                          u32 bumps = 0;
                          for (u32 i = 0; i < count; i++) {
                              const u32 d = GCM_FIFO_ARG(i), w = w0 + 4u * i;
                              u32 old; memcpy(&old, &s_state.vp_ucode[w], 4);
                              if (old != d || w + 4u > bytes0) bumps++;
                              s_state.vp_ucode[w+0] = (u8)(d);
                              s_state.vp_ucode[w+1] = (u8)(d >> 8);
                              s_state.vp_ucode[w+2] = (u8)(d >> 16);
                              s_state.vp_ucode[w+3] = (u8)(d >> 24);
                          }
                          if (bumps) g_rsx_vp_gen += bumps;
                          s_state.vp_ucode_write = w0 + 4u * count;
                          if (w0 + 4u * count > bytes0) s_state.vp_ucode_bytes = w0 + 4u * count;
                          s_state.vp_dirty = 1;
                          if (__builtin_expect(s_v1047_st, 0)) {
                              extern unsigned long long g_v1047_mhist[2048];
                              s_v1047_n[5] += count;
                              for (u32 i = 0; i < count; i++) g_v1047_mhist[((method >> 2) + i) & 2047u]++;
                          }
                          s_fifo_getoff += 4u + count * 4u;
                          continue;
                      }
                      if (__builtin_expect(s_v1047_st, 0)) s_v1047_n[4] += count;   /* v1047A stats: generic fast-path words */
                      /* v221C DCOAL (default ON; DCOAL=0 off; ABX 0x100000 = off for the in-run A/B): a NON-incrementing
                       * DRAW_INDEX_ARRAY packet carries one (first, count-1) word per 256 indices of ONE draw (libgcm
                       * splits every indexed draw that way; city: ~16.8k such words/frame vs ~9k draw records). The back
                       * end then paid the whole per-batch path (index read, VCACHE key + hash + memo + hash-table slot +
                       * entry, record merge) once per 256 indices, and 256-index windows of a mesh are often "sparse"
                       * (index span > count) = never cached (vib_build + conversion + GPU copy every frame). Contiguous
                       * arguments whose counts are all 256 (except the last) are handed over as ONE draw_indexed(first,
                       * total); the D3D12 back end cuts it into cacheable chunks itself (d3d12_draw_indexed, v221C) or
                       * re-splits it into the same 256 pieces. No state can change between the words of one packet.
                       * TRIANGLES only (strips/fans keep the per-word behaviour). */
                      const int dco = (type == 2 && method == 0x1824u && count > 1 && gcm_dcoal_active() &&
                                       s_state.primitive_type == RSX_PRIMITIVE_TRIANGLES);
                      for (u32 i = 0; i < count; i++) {
                          const u32 m = (type == 0) ? method + i * 4u : method;
                          const u32 d = GCM_FIFO_ARG(i);
                          if (dco && i + 1 < count && ((d >> 24) & 0xFFu) == 0xFFu) {
                              const u32 f0 = d & 0xFFFFFFu; u32 tot = 256u, lastc = 256u, j = i;
                              while (j + 1 < count && lastc == 256u) {
                                  const u32 d2 = GCM_FIFO_ARG(j + 1);
                                  if ((d2 & 0xFFFFFFu) != f0 + tot) break;
                                  lastc = ((d2 >> 24) & 0xFFu) + 1u; tot += lastc; j++;
                              }
                              if (j > i) {
                                  extern void rsx_draw_index_run(rsx_state* state, u32 first, u32 count);
                                  rsx_draw_index_run(&s_state, f0, tot);
                                  g_gcm_dcoal_runs++; g_gcm_dcoal_words += j - i + 1u;
                                  i = j;
                                  continue;
                              }
                          }
                           if (gf_fifo_fast2 && m == NV4097_DRAW_INDEX_ARRAY &&
                               s_state.primitive_type == RSX_PRIMITIVE_TRIANGLES) {
                               extern void rsx_v1129g_draw_index_triangles(rsx_state*, u32);
                               rsx_v1129g_draw_index_triangles(&s_state, d);
                               continue;
                           }
                          rsx_process_method(&s_state, m, d);
                          if (m == 0x50) gcm_ref_push_at(g_rsx_last_reference, s_fifo_getoff);
                          else if (m == 0x1800) {
                              const u32 idx = (d & 0xFFFFFFu) / 16u;
                              { static int rn = 0; if ((rn < 6 && rn++ < 6)) printf("[GET_REPORT] type=%u idx=%u%c", d >> 24, idx, 10); }
                              if (idx < CELL_GCM_MAX_REPORT_COUNT) {
                                  s_report_data[idx].timestamp = get_timestamp_ns();
                                  s_report_data[idx].value = (d >> 24) == 1u ? 0xFFFFu : 0u;
                              }
                          }
                      }
#undef GCM_FIFO_ARG
                      s_fifo_getoff += 4 + count * 4;
                      continue;
                  }
              } }
            for (u32 i = 0; i < count; i++) {
                u32 dea = gcm_io2ea(s_fifo_getoff + 4 + i * 4);
                if (!dea) break;
                u32 m = (type == 0) ? method + i * 4 : method;
                /* GCM_SET_USER_COMMAND is method 0xEB00, which this decode
                 * splits into subchannel 7 / method 0x0B00 -- so it was landing
                 * in the 2D engine path and being dropped as unrecognised. It
                 * is not a 2D method at all: it tells the RSX to raise a
                 * USER_CMD interrupt, which is what ps1_netemu's flip path
                 * blocks on. */
                { static int sd = -1; static unsigned long s7 = 0;
                  if (sd < 0) sd = getenv("GCM_SUBCH7") ? 1 : 0;
                  if (sd && subch == 7 && ((s7 < 16 && s7++ < 16)))
                      fprintf(stderr, "[subch7] method=0x%04X data=0x%08X\n",
                              m, vm_read32(dea)); }
                if (subch == 7 && m == 0x0B00u) {
                    extern void rsx_raise_user_cmd(u32 arg);
                    rsx_raise_user_cmd(vm_read32(dea));
                    continue;
                }
                /* GCM_SUBCH1_3D=1: treat subchannel 1 as the 3D object too.
                 * The title issues NV4097 methods on it -- 0x1A80..0x1AC0 is
                 * SET_TEXTURE_OFFSET for units 4-6 -- and the 2D path silently
                 * discards whatever it does not recognise, so that state never
                 * reaches the renderer. SET_OBJECT binds are not tracked, which
                 * is what makes the subchannel-to-engine mapping a guess. */
                /* GCM_OBJDBG=1: log every SET_OBJECT (method 0). The handle a
                 * title binds to a subchannel is what says which engine that
                 * subchannel drives; comparing subchannel 1 against subchannel 0
                 * turns the routing below from a guess into a reading. */
                { static int od = -1;
                  if (od < 0) { const char* e = getenv("GCM_OBJDBG"); od = e ? 1 : 0; }
                  if (od && m == 0) {
                      static unsigned seen[8]; static int have[8];
                      const unsigned h = vm_read32(dea);
                      if (subch < 8 && (!have[subch] || seen[subch] != h)) {
                          have[subch] = 1; seen[subch] = h;
                          printf("[GCM-OBJ] subch=%u SET_OBJECT handle=0x%08X%c",
                                 subch, h, 10); fflush(stdout);
                      } } }
                /* The subchannel is a binding slot, not an engine selector: a
                 * title may bind NV4097 to something other than subchannel 0.
                 * Twisted Metal uses subchannel 1, and caner (canersaka) hit
                 * the same thing in Yakuza Dead Souls, where SPU-built command
                 * lists bind NV4097 elsewhere -- his rsx_live_draw.c masks the
                 * subchannel out of the method for exactly this reason.
                 *
                 * gcm_2d_method only ever handles subchannels 2..7 (NV3062,
                 * NV308A/NV309E, NV3089), so anything arriving on 1 was being
                 * dropped on the floor -- for this title that included
                 * SET_SHADER_PROGRAM, leaving one stale fragment program bound
                 * for every draw in the game. Treat the subchannels the 2D path
                 * does not claim as 3D. GCM_SUBCH1_2D=1 restores the old split.
                 */
                static int s1_2d = -1;
                if (s1_2d < 0) { const char* e = getenv("GCM_SUBCH1_2D"); s1_2d = e ? atoi(e) : 0; }
                /* GCM_SUBCH_GATE=<file>: while <file> exists, log every method on a
                 * non-zero subchannel -- what each binding slot actually carries. */
                { static const char* sg = (const char*)-1; static unsigned poll; static int open_g;
                  if (sg == (const char*)-1) sg = getenv("GCM_SUBCH_GATE");
                  const u32 cm3 = m & 0x1FFCu;
                  /* GCM_VAL_WATCH=<hex>: every FIFO word carrying that value, any method. */
                  { static u32 vw = 1;
                    if (vw == 1) { const char* e = getenv("GCM_VAL_WATCH"); vw = e ? (u32)strtoul(e, 0, 16) : 0; }
                    if (vw && vm_read32(dea) == vw) {
                        static int vn; if ((vn < 64 && vn++ < 64))
                            printf("[valwatch] subch=%u method=0x%04X = 0x%08X get=0x%X%c", subch, m, vw, s_fifo_getoff, 10);
                    } }
                  /* ...and surface offsets (colour 0x210/0x218/0x288/0x28C, zeta 0x214) and texture 1 offset */
                  if (sg && !subch && (cm3 == 0x210 || cm3 == 0x214 || cm3 == 0x218 || cm3 == 0x288 || cm3 == 0x28C ||
                                       cm3 == 0x1A20)) {
                      if ((poll++ & 255u) == 0) { FILE* g = fopen(sg, "rb"); open_g = g != NULL; if (g) fclose(g); }
                      if (open_g) printf("[surfreg] 0x%04X = 0x%08X%c", cm3, vm_read32(dea), 10);
                  }
                  if (sg && subch) {
                      if ((poll++ & 255u) == 0) { FILE* g = fopen(sg, "rb"); open_g = g != NULL; if (g) fclose(g); }
                      if (open_g) printf("[subch] %u 0x%04X = 0x%08X%c", subch, m, vm_read32(dea), 10);
                  } }
                /* Mirror the whole method stream into the live NV4097->D3D12
                 * engine (caner / canersaka). Inert unless RSX_LIVE_DRAW is set.
                 * It wants the raw method with its subchannel bits -- it
                 * canonicalises with & 0x1FFC itself. */
                { static int live = -1;
                  if (live < 0) live = rsx_live_draw_enabled();
                  if (live) rsx_live_draw_method((subch << 13) | m, vm_read32(dea)); }

                /* And into the platform-neutral register-file draw engine,
                 * which is the macOS path under PS3RECOMP_RSX_ENGINE=dispatch.
                 * It wants the raw method with its subchannel bits for the
                 * same reason the live engine does, and masks them itself. */
                { static int eng = -1;
                  if (eng < 0) eng = rsx_draw_engine_enabled();
                  if (eng) rsx_draw_engine_method((subch << 13) | m, vm_read32(dea)); }

                /* RSX_LIVE_FEED_DBG=1: what the FIFO actually carries. Counted
                 * for any run, live engine or not, so the method stream and the
                 * backend's draw callbacks can be compared in one measurement. */
                { static int fdbg = -1;
                  if (fdbg < 0) fdbg = getenv("RSX_LIVE_FEED_DBG") ? 1 : 0;
                  if (fdbg) {
                      static long long n_be = 0, n_va = 0, n_ia = 0, tick = 0;
                      const u32 cm = m & 0x1FFCu;
                      if      (cm == 0x1808u) n_be++;
                      else if (cm == 0x1814u) n_va++;
                      else if (cm == 0x1824u) n_ia++;
                      if (((++tick) % 200000) == 0)
                          printf("[feed] begin_end=%lld draw_arrays=%lld draw_index=%lld\n",
                                 n_be, n_va, n_ia), fflush(stdout);
                  } }

                /* Driver methods (0xE9xx, 0xEBxx) occupy bits 2-15 of the
                 * FIFO word; the standard 11-bit extraction (w & 0x1FFC)
                 * truncates them.  Reconstruct the full address the same way
                 * the draw-engine call above does. */
                { const u32 mfull = (subch << 13) | m;
                if (mfull == 0xEB00u || mfull == 0xEB04u) {
                    cellGcmQueueUserCommand(vm_read32(dea));
                } else if (mfull == 0xE920u || mfull == 0xE924u) {
                    cellGcmSetFlipCommand(vm_read32(dea) & 7u);
                } else if (subch == 0 || (subch == 1 && !s1_2d)) {
                    rsx_process_method(&s_state, m, vm_read32(dea));
                    /* NV406E_SET_REFERENCE: queue the fence value for PACED
                     * publication (gcm_ref_publish below) instead of letting a
                     * later fence in the same batch overwrite it. */
                    if (m == 0x50) gcm_ref_push_at(g_rsx_last_reference, s_fifo_getoff);
                    /* NV4097_GET_REPORT: (type << 24) | offset into the report
                     * area. Sample counts are not measured, so a ZPASS query
                     * reports "visible" -- the conservative answer; reading 0
                     * tells the title every queried object is occluded. */
                    if (m == 0x1800) {
                        const u32 v = vm_read32(dea), idx = (v & 0xFFFFFFu) / 16u;
                        { static int rn = 0; if ((rn < 6 && rn++ < 6)) printf("[GET_REPORT] type=%u idx=%u%c", v >> 24, idx, 10); }
                        if (idx < CELL_GCM_MAX_REPORT_COUNT) {
                            s_report_data[idx].timestamp = get_timestamp_ns();
                            s_report_data[idx].value = (v >> 24) == 1u ? 0xFFFFu : 0u;
                        }
                    }
                } else
                    gcm_2d_method(subch, m, vm_read32(dea));
                }
            }
            s_fifo_getoff += 4 + count * 4;
            continue;
        }
        { static int _uw = 0; if ((_uw < 16 && _uw++ < 16) && getenv("RTT_DUMP"))
            fprintf(stderr, "[FIFOUW] unknown word 0x%08X at getoff 0x%X\n", w, s_fifo_getoff); }
        s_fifo_getoff += 4;                    /* unknown word: skip */
    }

    /* Walker watchdog. Everything above recovers from a *recognised* stall --
     * a branch we cannot take, a park the title abandoned. A walker that stops
     * for any other reason simply freezes `get` for the rest of the run and
     * nothing renders again, and which of those happens is a race, so the same
     * boot renders one time in five. If `get` has not moved across several
     * passes while `put` is somewhere else, there is work the walker will
     * never reach: resynchronise to the write head. Same trade as the
     * unmapped-get path -- commands are lost, the FIFO lives. */
    /* The stuck-detector above only fires when `get` stops MOVING. A walker
     * going in circles moves plenty -- it just never arrives. Twisted Metal
     * and flOw both write one ring and their JUMPs lead to `put`; VF5 does
     * not. AMGL leaves libgcm's default ring at IO 0 parked on its own JUMP
     * loop and drives the RSX from a second buffer it mapped at IO 0xA00000,
     * so the walker circulates the default ring forever, `get` never reaches
     * `put`, the title's ring is never reclaimed, and AMGL prints "Command
     * Buffer Overflow" until it runs off the end of its own mapping.
     *
     * A drain that burns its ENTIRE word budget without arriving is that
     * signature: no legitimate frame leaves a million words pending, and a
     * genuine burst clears within a tick or two. Circling does not, so
     * require several passes in a row before following the write head.
     *
     * ponytail: consecutive-budget-burn is a heuristic, not a proof. The real
     * fix is knowing which buffer the RSX is bound to -- that needs the
     * SetQueueHandler / context-switch path modelled, not a walker rule. */
    { static int burned = 0, passes = -1;
      if (passes < 0) { const char* e = getenv("GCM_FIFO_CIRCLE_PASSES");
                        passes = e ? atoi(e) : 4; }
      if (passes && budget <= 0 && s_fifo_getoff != put) {
          if (++burned >= passes) {
              static int n = 0;
              if ((n < 8 && n++ < 8))
                  fprintf(stderr, "[cellGcmSys] FIFO walker circling (get=0x%08X, "
                          "put=0x%08X, full budget burned x%d) -- following the "
                          "write head\n", s_fifo_getoff, put, burned);
              if (!v1053_abort_call("circling (budget burned)", s_fifo_getoff, 0))   /* v1053A */
              gcm_fifo_resync_why("circling", &s_fifo_getoff, put);
              burned = 0;
          }
      } else burned = 0; }

    { static u32 last = 0xFFFFFFFFu; static int stuck = 0;
      if (s_fifo_getoff == last && s_fifo_getoff != put) {
          /* v1053A GCM_STUCK_SEMA: a pass blocked in a semaphore acquire (still inside GCM_SEMA_TIMEOUT) is waiting, not
           * stuck -- the acquire path itself skips it after the timeout. Resyncing here dropped the rest of the frame. */
          (void)v1053_on();
          if (s_v1053_ss && s_v1053_semwait) stuck = 0;
          else if (++stuck >= 8) { gcm_fifo_dump_around(s_fifo_getoff);
                             gcm_fifo_resync_why("stuck-get", &s_fifo_getoff, put); stuck = 0; }
      } else { stuck = 0; last = s_fifo_getoff; } }
    v1053_report();   /* v1053A 10 s summary (only when something happened) */

    /* Publish progress: get chases put; ref advances at most ONE queued fence
     * per tick (gcm_ref_push/gcm_ref_publish_one above) so every SET_REFERENCE
     * value stays observable to equality-waiters. The wrap recycle path polls
     * the drained EA. */
    if (s_draindbg) {   /* v180: cached */
        static int n2 = 0;
        if (n2++ % 60 == 0)
            fprintf(stderr, "[DRAINEND] %s: %08X -> %08X (%u bytes), put=%08X\n",
                    why, start_off, s_fifo_getoff, s_fifo_getoff - start_off, put);
    }
    /* Recycle the TITLE'S OWN command buffer when its callback will not.
     *
     * gCellGcmCurrentContext points at the title's context, not the one
     * cellGcmInit handed back: Virtua Fighter 5's is a 512 KB ring at EA
     * 0x4AE00000, and the buffer-full callback AMGL installs (0x006A8030) only
     * prints "[AMGL]:[ERROR] Command Buffer Overflow!" and returns 0 -- which
     * libgcm's gcmReserve reads as "retry", so the title writes straight past
     * `end` into unmapped memory and never renders again.
     *
     * Hardware never reaches that state because the ring is recycled once the
     * RSX has consumed it. Do the same here, on the title's context, and only
     * when it is provably safe: the walker has drained everything the title
     * submitted (get == put) and `current` is within one reserve of `end`.
     * A title whose own callback recycles never reaches that state, so this is
     * inert for every port that works today.
     *
     * ponytail: append the JUMP at `current` the way the SDK's own callback
     * does, rather than tracking a second copy of the ring's geometry. */
    if (s_gcm_ctx_out_ea) {
        u32 ctx = vm_read32(s_gcm_ctx_out_ea);
        u32 begin = ctx ? vm_read32(ctx + 0x0) : 0;
        u32 end   = ctx ? vm_read32(ctx + 0x4) : 0;
        u32 cur   = ctx ? vm_read32(ctx + 0x8) : 0;
        /* `get == put` -- a fully drained FIFO -- was too strict. It holds
         * while a title is only clearing, and stops holding the moment its SPU
         * work starts producing real command volume: the walker then always has
         * a backlog, the recycle never fires, and the ring overflows again.
         * That is exactly where Virtua Fighter 5 stalls once its "SPU Delegate"
         * group starts feeding the renderer.
         *
         * Hardware does not need a drained FIFO either -- only that the bytes
         * about to be overwritten have already been read. So require the walker
         * to be a margin PAST `begin` instead: the head of the ring is consumed,
         * which is the region the guest is about to write. */
        u32 io_begin_chk = begin ? gcm_ea2io(begin) : 0xFFFFFFFFu;
        int head_consumed =
            (io_begin_chk != 0xFFFFFFFFu) &&
            (s_fifo_getoff == put ||
             (s_fifo_getoff > io_begin_chk &&
              s_fifo_getoff - io_begin_chk >= GCM_RECYCLE_MARGIN));
        /* Already overrun. gcmReserve writes past `end` when its callback
         * declines to recycle (AMGL's only prints), and once the guest is out
         * there the walker follows into memory that is not the ring: `get`
         * stops advancing, head_consumed can never become true again, and the
         * title is wedged for good -- which is exactly where Virtua Fighter 5
         * stops, at the same flip every run however long it is left.
         *
         * There is nothing left to protect at that point. Recycling loses
         * whatever was written past the end; not recycling loses the rest of
         * the run. */
        /* A pass that stopped at a flip has a backlog on purpose (the next
         * frame), and teleporting `get` to `begin` would drop it -- fence
         * writes included, so the title then waits on a reference that never
         * comes (The Simpsons Arcade Game froze at its 4th recycle). The next
         * pass drains to `put` and recycles then. */
        if (!strcmp(why, "flip") && s_fifo_getoff != put) head_consumed = 0;
        /* Only once the title has been SEEN to overrun. Near `end` is also
         * where a title whose callback does recycle sits just before calling
         * it: Tornado Outbreak's steps its context through 64 KB segments of a
         * larger ring, and recycling under it from this thread rewound put/get
         * behind its callback's bookkeeping -- which then waited forever for a
         * GET that could not arrive. A callback that declines (AMGL's) always
         * overruns first, so the first overrun is recycled by the path above
         * and near-end recycling is enabled from then on. */
        static int s_title_overran = 0;
        /* v1032H GCM_TITLE_RING_GUARD (default 1; 0 = the old test). Two fixes for
         * titles whose OWN callback manages the ring (Godfather / EA's ring manager,
         * callback 0x8A3454, installed over ours by 0x8A2E74):
         *  1. `cur == end` is NOT an overrun. Inline gcmReserve only calls the callback
         *     when current + n > end, so a command that exactly fills the segment
         *     leaves current == end legally. The old `>=` latched s_title_overran on
         *     that, and from then on recycled under the title whenever the FIFO was
         *     drained near a segment end.
         *  2. A title that moves ctx->begin itself steps its own segments through the
         *     buffer (Godfather: 32 KB segments, each opened with
         *     SET_REFERENCE(segment EA); its callback waits on ref for free space).
         *     Rewinding such a ring from this thread rewrites ctx->current under the
         *     guest: Godfather's callback then saw current == begin == ref and spun on
         *     ref forever (t=18.2 s hang: "recycled the title's own ring" then
         *     [HOTREAD] put == get, ref = 0x49FE3EA8 = the segment's begin). Once
         *     begin is seen to move on the same context, never recycle it. */
        static int s_ring_guard = -1;
        static u32 s_seen_ctx = 0, s_seen_begin = 0;
        static int s_title_segments = 0;
        if (s_ring_guard < 0) {
            const char* e = getenv("GCM_TITLE_RING_GUARD");
            s_ring_guard = (e && *e == '0') ? 0 : 1;
            fprintf(stderr, "[GCM] v1032H title ring guard %s (GCM_TITLE_RING_GUARD)%c",
                    s_ring_guard ? "ON: cur == end is not an overrun; title-stepped segments are never recycled"
                                 : "OFF (old recycle test)", 10);
        }
        if (s_ring_guard && begin) {
            if (ctx == s_seen_ctx && begin != s_seen_begin && !s_title_segments) {
                s_title_segments = 1;
                s_v1034_title_segs = 1;   /* v1034F: flip notes may use the segment tail */
                fprintf(stderr, "[GCM] v1032H ctx 0x%08X: title moves its own segment begin "
                        "(0x%08X -> 0x%08X) -- runtime ring recycle disabled for it%c",
                        ctx, s_seen_begin, begin, 10);
            }
            s_seen_ctx = ctx; s_seen_begin = begin;
        }
        const int v1032h_over = s_ring_guard ? (cur > end) : (cur >= end);
        if (s_ring_guard && begin && end > begin && cur == end) {
            static int nfull = 0;
            if (nfull < 2) { nfull++;
                fprintf(stderr, "[GCM] v1032H ring exactly full (cur == end = 0x%08X) -- not an overrun%c", end, 10); }
        }
        if (begin && end > begin && v1032h_over) { head_consumed = 1; s_title_overran = 1; }
        if (begin && end > begin && cur >= begin && cur + GCM_RECYCLE_SLACK >= end
                && head_consumed && s_title_overran && !(s_ring_guard && s_title_segments)) {
            u32 io_begin = gcm_ea2io(begin);
            if (io_begin != 0xFFFFFFFFu) {
                u32 jmp_at = (cur + 4 <= end) ? cur : end - 4;
                vm_write32(jmp_at, 0x20000000u | io_begin);   /* JUMP -> begin */
                vm_write32(ctx + 0x8, begin);                 /* current = begin */
                vm_write32(GCM_CONTROL_GUEST_ADDR + 0, io_begin);  /* put */
                s_fifo_getoff = io_begin;                          /* get */
                put = io_begin;
                static int n = 0;
                if ((n < 8 && n++ < 8))
                    fprintf(stderr, "[cellGcmSys] recycled the title's own ring "
                            "(ctx=0x%08X begin=0x%08X end=0x%08X) -- its callback "
                            "does not%c", ctx, begin, end, 10);
            }
        }
    }

    g_gcm_fifo_drained_ea = gcm_io2ea(s_fifo_getoff);
    /* v1129W GF_FIFO_KICK2: the put this walk ends at (the ring recycle above may have moved it) and whether it
     * stopped at a flip word with commands still behind it. */
    s_v1129w_walk_put = put;
    s_v1129w_flip_backlog = (s_fifo_getoff != put) && !strcmp(why, "flip");
    /* GCM_GET_EQ_PUT=1: publish `get` as having reached `put` rather than where
     * the walker actually is. A probe, not a fix -- it removes the back-pressure
     * a title uses to avoid overwriting commands the GPU has not read. It exists
     * to answer one question for a title that reports its ring full while the
     * walker says it is drained: is the title reading `get`, or something else?
     * If the overflow survives this, `get` is not what it consults. */
    { static int eq = -1;
      if (eq < 0) { const char* e = getenv("GCM_GET_EQ_PUT"); eq = e ? atoi(e) : 0; }
      vm_write32(GCM_CONTROL_GUEST_ADDR + 4, eq ? put : s_fifo_getoff); }

    AcquireSRWLockExclusive(&s_ref_pub_lock);                           /* ref */
    if (v1048_ref_now()) gcm_ref_publish_latest(); else   /* v1048D GCM_REF_LATEST */
    gcm_ref_publish_one();
    ReleaseSRWLockExclusive(&s_ref_pub_lock);
    if (__builtin_expect(v243c_rb, 0)) {   /* v243C RFBOUND */
        LARGE_INTEGER q1; QueryPerformanceCounter(&q1);
        s_v243c_busy += q1.QuadPart - v243c_q0.QuadPart;
        if (s_v243c_open && s_fifo_getoff == put) {
            g_v243c_st[8] += q1.QuadPart - s_v243c_t0q; g_v243c_st[9] += s_v243c_busy; g_v243c_st[10]++; s_v243c_open = 0;
        }
    }
    if (__builtin_expect(s_v1047_ph, 0)) {   /* v1047A GCM_FASTWALK_PHASE: per-phase walk cost, old vs new */
        extern unsigned long long g_v1047_mhist[2048];
        static int ph_cur = -1; static LARGE_INTEGER ph_t0, qf;
        static long long ph_walk = 0; static unsigned long long ph_fs0 = 0, ph_n1 = 0, ph_n2 = 0, ph_be0 = 0;
        static double acc[2][6];   /* per parity: s, flips, walk ms, packets, draws, phases */
        LARGE_INTEGER q1; QueryPerformanceCounter(&q1);
        if (!qf.QuadPart) QueryPerformanceFrequency(&qf);
        ph_walk += q1.QuadPart - v1047_q0.QuadPart;
        const int ph = v1047_phase;
        if (ph != ph_cur) {
            const unsigned long long fs = (unsigned long long)g_gcm_fifo_flip_seq, be = g_v1047_mhist[0x1808u >> 2];
            if (ph_cur >= 0) {
                const double secs = (double)(q1.QuadPart - ph_t0.QuadPart) / (double)qf.QuadPart;
                const double fl = (double)(fs - ph_fs0), wms = (double)ph_walk * 1000.0 / (double)qf.QuadPart;
                const double pk = (double)(s_v1047_n[1] - ph_n1), dr = (double)(be - ph_be0) * 0.5;
                const int par = ph_cur & 1;
                fprintf(stderr, "[V1047A] phase %d (%s): %.1f s, %.0f flips (%.1f/s), walk %.0f ms = %.2f ms/flip, %.0f packets/flip, %.0f draws/flip, %.2f us/draw, %.1f ns/packet%c",
                        ph_cur, par ? "old" : "fastwalk", secs, fl, secs > 0 ? fl / secs : 0.0, wms, fl > 0 ? wms / fl : 0.0,
                        fl > 0 ? pk / fl : 0.0, fl > 0 ? dr / fl : 0.0, dr > 0 ? wms * 1000.0 / dr : 0.0, pk > 0 ? wms * 1e6 / pk : 0.0, 10);
                if (fl > 0 && dr / fl >= 1000.0 && secs > 3.0) {   /* city phases only */
                    acc[par][0] += secs; acc[par][1] += fl; acc[par][2] += wms; acc[par][3] += pk; acc[par][4] += dr; acc[par][5] += 1;
                    if (acc[0][1] > 0 && acc[1][1] > 0)
                        fprintf(stderr, "[V1047A] A/B so far (city phases): fastwalk %.0f phases %.1f flips/s %.2f ms/flip %.2f us/draw %.1f ns/packet | old %.0f phases %.1f flips/s %.2f ms/flip %.2f us/draw %.1f ns/packet%c",
                                acc[0][5], acc[0][1] / acc[0][0], acc[0][2] / acc[0][1], acc[0][2] * 1000.0 / acc[0][4], acc[0][2] * 1e6 / acc[0][3],
                                acc[1][5], acc[1][1] / acc[1][0], acc[1][2] / acc[1][1], acc[1][2] * 1000.0 / acc[1][4], acc[1][2] * 1e6 / acc[1][3], 10);
                }
            }
            ph_cur = ph; ph_t0 = q1; ph_walk = 0; ph_fs0 = fs; ph_n1 = s_v1047_n[1]; ph_n2 = s_v1047_n[2]; ph_be0 = be;
        }
        (void)ph_n2;
    }
    else if (__builtin_expect(s_v1047_st, 0)) {   /* v1047A GCM_FASTWALK_STATS report, every 10 s */
        extern unsigned long long ps3_ms_now(void);
        extern unsigned long long g_v1047_mhist[2048];
        const unsigned long long now = ps3_ms_now();
        s_v1047_n[6]++;
        if (!s_v1047_t0) s_v1047_t0 = now;
        else if (now - s_v1047_t0 >= 10000ull) {
            static unsigned long long s_fs0 = 0;
            const unsigned long long fs = (unsigned long long)g_gcm_fifo_flip_seq, nf = fs - s_fs0; s_fs0 = fs;
            const double pf = nf ? 1.0 / (double)nf : 1.0;   /* no flips counted: 10 s totals */
            fprintf(stderr, "[V1047A] 10 s, %llu FIFO flips, %llu walks: per flip %.0f FIFO words, %.0f method packets with %.0f data words "
                    "(TC blocks %.0f, generic fast path %.0f, TP bulk %.0f); fastwalk %s%c",
                    nf, s_v1047_n[6], pf * s_v1047_n[0], pf * s_v1047_n[1], pf * s_v1047_n[2], pf * s_v1047_n[3],
                    pf * s_v1047_n[4], pf * s_v1047_n[5], s_v1047 ? "ON" : "off", 10);
            {   /* top 24 methods that reached rsx_process_method (+ TP bulk words) */
                unsigned long long tot = 0; for (u32 k = 0; k < 2048u; k++) tot += g_v1047_mhist[k];
                char line[1024]; int len = 0;
                for (int r = 0; r < 24; r++) {
                    u32 best = 0; unsigned long long bv = 0;
                    for (u32 k = 0; k < 2048u; k++) if (g_v1047_mhist[k] > bv) { bv = g_v1047_mhist[k]; best = k; }
                    if (!bv) break;
                    len += snprintf(line + len, sizeof line - (size_t)len, " %04X:%.0f", best << 2, pf * (double)bv);
                    g_v1047_mhist[best] = 0;
                    if (len > (int)sizeof line - 32) break;
                }
                fprintf(stderr, "[V1047A]   methods/flip %.0f, top:%s%c", pf * (double)tot, line, 10);
                memset(g_v1047_mhist, 0, sizeof g_v1047_mhist);
            }
            memset(s_v1047_n, 0, sizeof s_v1047_n);
            s_v1047_t0 = now;
        }
    }
}

/* FIFO command-buffer-full callback body. The title's inline gcmReserve calls
 * context->callback(context, count) (routed here via the synthetic OPD) when the
 * ring's `current` nears `end`. Real libgcm flushes, waits for the RSX `get` to
 * pass, then recycles `current` to `begin`. We mirror that: wait (bounded) for
 * the ticker drain to consume everything up to `current`, then reset current to
 * begin. Runs on the guest thread; the ticker owns all rendering, so we only
 * pointer-shuffle here (the drain's own wrap-detect handles s_get). Without this
 * the ring never recycles and the whole pipeline wedges once the FIFO wraps
 * (~200 frames at the 256KB default), which looked like a GPU/driver hang. */
static u32 gcm_ea2io(u32 ea)
{
    u32 page = ea >> 20;
    if (page >= 65536) return 0xFFFFFFFFu;
    u16 io_page = s_io_address_table[page];
    if (io_page == 0xFFFF) return 0xFFFFFFFFu;
    return ((u32)io_page << 20) | (ea & 0xFFFFFu);
}

void cellGcm_fifo_recycle(u32 ctx_ea)
{
    if (!ctx_ea) return;
    u32 begin   = vm_read32(ctx_ea + 0x0);
    u32 current = vm_read32(ctx_ea + 0x8);
    if (current <= begin) return;                       /* nothing to recycle */

    if (getenv("CELLMARK_BLINKDBG"))
        printf("[RECYCLE] ring wrap: current=0x%08X -> begin=0x%08X\n", current, begin);

    static int s_recdbg = -1;
    if (s_recdbg < 0) s_recdbg = getenv("GCM_RECDBG") ? 1 : 0;
    if (s_recdbg)
        fprintf(stderr, "[REC>] tid=%lu begin=%08X cur=%08X put=%08X get=%08X drained=%08X\n",
                GetCurrentThreadId(), begin, current,
                vm_read32(GCM_CONTROL_GUEST_ADDR + 0),
                vm_read32(GCM_CONTROL_GUEST_ADDR + 4),
                g_gcm_fifo_drained_ea);

    /* Do what the SDK's default command-buffer-full callback does: append a
     * JUMP-to-begin at the write head and move `put` to begin. The FIFO walker
     * consumes the tail, follows the jump, and idles at begin; then it's safe
     * for the guest to write from begin (that region was consumed long ago). */
    u32 io_begin = gcm_ea2io(begin);
    if (io_begin != 0xFFFFFFFFu) {
        vm_write32(current, 0x20000000u | io_begin);            /* JUMP begin  */
        /* The jump has to be in guest memory before `put` moves behind the
         * walker's `get`. Program order is not enough on arm64: the walker
         * may observe the new `put`, find `get` != `put`, and read the old
         * word at `current` before the jump lands there. */
        atomic_thread_fence(memory_order_release);
        vm_write32(GCM_CONTROL_GUEST_ADDR + 0, io_begin);        /* put = begin */
    }

    /* Wait (bounded ~2s) for the walker to consume the tail + take the jump so
     * no commands are lost; a stalled ticker degrades to dropped commands. */
    int spins = 0;
    while (g_gcm_fifo_drained_ea != begin && spins < 2000) { Sleep(1); spins++; }
    if (spins >= 2000) {
        static int warned = 0;
        if ((warned < 4 && warned++ < 4))
            printf("[cellGcmSys] fifo recycle: drain stalled (drained=0x%08X begin=0x%08X)\n",
                   g_gcm_fifo_drained_ea, begin);
    }

    vm_write32(ctx_ea + 0x8, begin);                    /* recycle ring to base */

    if (s_recdbg)
        fprintf(stderr, "[REC<] tid=%lu cur-now=%08X put=%08X get=%08X drained=%08X spins=%d\n",
                GetCurrentThreadId(), vm_read32(ctx_ea + 0x8),
                vm_read32(GCM_CONTROL_GUEST_ADDR + 0),
                vm_read32(GCM_CONTROL_GUEST_ADDR + 4),
                g_gcm_fifo_drained_ea, spins);
}

/* NID: 0xDC09357E */
s32 cellGcmSetDisplayBuffer(u32 bufferId, u32 offset, u32 pitch,
                            u32 width, u32 height)
{
    printf("[cellGcmSys] SetDisplayBuffer(id=%u, offset=0x%X, pitch=%u, %ux%u)\n",
           bufferId, offset, pitch, width, height);

    if (bufferId >= CELL_GCM_MAX_DISPLAY_BUFFER_NUM)
        return CELL_GCM_ERROR_INVALID_VALUE;

    s_display_buffers[bufferId].offset = offset;
    s_display_buffers[bufferId].pitch  = pitch;
    s_display_buffers[bufferId].width  = width;
    s_display_buffers[bufferId].height = height;
    s_display_buffer_set[bufferId] = 1;

    /* Tell the live engine too, or it has no registered scanout to present and
     * falls back to whatever surface happens to be current. Display buffers are
     * always in RSX local memory (location 0). */
    if (rsx_live_draw_enabled())
        rsx_live_draw_set_display_buffer(bufferId, 0, offset, pitch, width, height);
    if (rsx_draw_engine_enabled())
        rsx_draw_engine_set_display_buffer(bufferId, 0, offset, pitch, width, height);

    return CELL_OK;
}

/* Backend query (not a guest API): is this raw RSX offset a registered
 * display buffer? Used to split display draws (-> backbuffer) from
 * offscreen render-to-texture passes. */
int cellGcmOffsetIsDisplay(u32 offset)
{
    for (int i = 0; i < CELL_GCM_MAX_DISPLAY_BUFFER_NUM; i++)
        if (s_display_buffer_set[i] && s_display_buffers[i].offset == offset)
            return 1;
    return 0;
}

/* NID: 0xEAA52F23 */
/* How many display buffers the title has actually registered. cellResc needs
 * it to know how far to rotate its flip target; nothing else about the set is
 * anyone else's business, so this stays a count rather than exposing the array. */
u32 cellGcm_display_buffer_count(void)
{
    u32 n = 0;
    for (u32 i = 0; i < CELL_GCM_MAX_DISPLAY_BUFFER_NUM; i++)
        if (s_display_buffer_set[i]) n++;
    return n;
}

/* 2026-10-04: VRAM offset of the display buffer most recently flipped (DISPLAY_AS_RT
 * presents THIS buffer, not "the buffer of the frame's last display draw" -- the
 * UI pass lands in the other ping-pong buffer, which then never reached the
 * screen: inFamous lost its "Press Start" prompt). 0 = no flip yet. */
volatile u32 g_gcm_last_flip_off = 0;
/* v62 GCM_FLIP_ORDERED (default 1; 0 = off): g_gcm_last_flip_off is set when the PPU
 * REQUESTS the flip -- which, once the main thread runs a frame ahead of the RSX, is the
 * NEXT frame's buffer when the renderer presents: the screen then showed the ping-pong
 * buffer WITHOUT the UI pass (no 'Press Start', no HUD). cellGcmSetFlipCommand now also
 * drops a one-word marker into the guest command stream (as the real library writes its
 * flip there); the drain sets g_gcm_fifo_flip_off when it reaches it, so the backend
 * presents the buffer of the last flip the RSX has actually executed up to. */
static int gcm_flip_ordered_on(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("GCM_FLIP_ORDERED"); v = (e && *e == '0') ? 0 : 1; }
    return v;
}
static u32 s_flip_note_ctx = 0;   /* context of the current flip call (0 = default) */
static void gcm_flip_note_into_fifo(u32 bufferId)
{
    const u32 ctx = s_flip_note_ctx ? s_flip_note_ctx : s_gcm_context_ea;
    if (!ctx || !gcm_flip_ordered_on()) return;
    const u32 end = vm_read32(ctx + 4), cur = vm_read32(ctx + 8);
    /* the default ring keeps its wrap reserve; inFamous flips through small per-buffer
     * contexts (4 KB each, CALLed from the main stream) that only need room for the word */
    u32 need = (ctx == s_gcm_context_ea) ? GCM_RECYCLE_SLACK : 16u;
    /* v1034F: the 4 KB reserve only protects the runtime's own ring recycle, which v1032H disables
     * for a title that steps its own segments (Godfather: EA ring manager). There the reserve left
     * every flip issued in a segment's last 4 KB unmarked, so the walker never saw that flip and
     * two frames merged into one batch. 16 bytes still leave room for the title's JUMP. */
    if (v1034f_on() && s_v1034_title_segs && ctx == s_gcm_context_ea) need = 16u;
    if (!cur || cur + 4 + need > end || gcm_ea2io(cur) == 0xFFFFFFFFu) {
        g_gcm_flip_note_skips++;   /* v1034F */
        static int n = 0; if (n < 4) { n++;
            fprintf(stderr, "[GCM] v62 flip marker skipped: ctx=0x%08X cur=0x%08X end=0x%08X io=0x%08X%c",
                    ctx, cur, end, cur ? gcm_ea2io(cur) : 0u, 10); }
        return;
    }
    s_v1048_note_req[s_v1034_note_w & 7u] = s_flip_request_count + 1u;   /* v1048D2: the request this note belongs to */
    g_ps_flip_V[(s_flip_request_count + 1u) & 63u] = g_gf_frame_V;          /* v1051P: its frame's scheduled vblank */
    s_v1034_note_io[s_v1034_note_w++ & 7u] = gcm_ea2io(cur) + 1u;   /* v1034F-2 (+1: 0 = empty slot) */
    s_v1049_noted_req = s_flip_request_count + 1u;   /* v1049G: gcm_flip_request increments the count right after */
    vm_write32(cur, GCM_FLIP_NOTE | (bufferId & 7u));
    vm_write32(ctx + 8, cur + 4);
}
static s32 gcm_flip_request(u32 bufferId, int in_fifo)
{
    hz_guest_flip();   /* v225A */
    if (bufferId < CELL_GCM_MAX_DISPLAY_BUFFER_NUM && s_display_buffer_set[bufferId])
        g_gcm_last_flip_off = s_display_buffers[bufferId].offset;
    /* FRAME_DRAWS=1: primitive batches per frame. Prints "[world] FIRST BIG FRAME"
     * the first time a frame carries >= 500 batches (inFamous: loading screen = 1,
     * in-world ~1270), then every 50th flip. Cheap, so it is safe to leave on. */
    { static int fd = -1;
      if (fd < 0) fd = getenv("FRAME_DRAWS") ? 1 : 0;
      extern volatile unsigned g_rsx_frame_draws;
      extern volatile unsigned g_rsx_frame_verts;
      unsigned d = g_rsx_frame_draws; g_rsx_frame_draws = 0;
      v1048_on_flip(d);   /* v1048D per-phase frame stats */
      unsigned v = g_rsx_frame_verts; g_rsx_frame_verts = 0;
      extern volatile unsigned g_rsx_frame_xverts; extern volatile unsigned long long g_rsx_frame_pbytes;
      unsigned xv = g_rsx_frame_xverts; g_rsx_frame_xverts = 0;
      unsigned long long pb = g_rsx_frame_pbytes; g_rsx_frame_pbytes = 0;
      if (fd) {
          static unsigned vmax = 0; if (v > vmax) vmax = v;
          extern unsigned long long ps3_ms_now(void);
          static unsigned long long n = 0, t0 = 0; static int big = 0;
          unsigned long long now = ps3_ms_now();
          if (!t0) t0 = now;
          ++n;
          if (d >= 500 && !big) { big = 1;
              fprintf(stderr, "[world] FIRST BIG FRAME: flip %llu at %llu ms, %u draws, %u verts (%u as triangle lists, %llu MB packed)%c",
                      n, now - t0, d, v, xv, pb >> 20, 10); }
          if ((n % 50ull) == 0)
          { fprintf(stderr, "[frame-draws] flip %llu at %llu ms: %u draws, %u verts (max %u in last 50; %u as tri lists, %llu MB packed)%c",
                    n, now - t0, d, v, vmax, xv, pb >> 20, 10); vmax = 0; }
      } }
    /* GCM_FLIPCOUNT=1: every flip, with a timestamp. FLIP_DBG caps at 20 lines,
     * which answers "did it ever flip?" and not "is it still flipping, and how
     * fast?" -- the question that matters when a title stops appearing on
     * screen. Virtua Fighter 5 does not use the in-FIFO 0xFEADxxxx flip word at
     * all (that counter stays at zero for the whole run); it flips through
     * cellGcmSetFlip, so this is where its frame rate actually is. */
    { static int dbg = -1;
      if (dbg < 0) dbg = getenv("GCM_FLIPCOUNT") ? 1 : 0;
      if (dbg) {
          extern unsigned long long ps3_ms_now(void);
          static unsigned long long n = 0, t0 = 0;
          unsigned long long now = ps3_ms_now();
          if (!t0) t0 = now;
          ++n;
          /* Per-frame stderr writes perturb a 16.7 ms frame budget enough to
           * manufacture periodic 31 ms samples.  Keep dense startup coverage,
           * then sample every ten flips; stall detection still reacts within
           * roughly 167 ms at 60 Hz. */
          if (n <= 20ull || (n % 10ull) == 0)
              fprintf(stderr, "[flip] %llu at %llu ms%c", n, now - t0, 10);
      } }

    /* GCM_FLIP_BT=<n>: dump the flipping thread's guest stack for the flips
     * around n. When a title renders correctly for a while and then stops
     * submitting -- no error, no unresolved import, no overflow -- the only
     * thing separating the last good frame from the first missing one is what
     * the render path was doing on each. This is the window onto that. */
    { static long long at = -2;
      if (at == -2) { const char* e = getenv("GCM_FLIP_BT"); at = e ? atoll(e) : -1; }
      if (at >= 0) {
          static unsigned long long m = 0;
          ++m;
          if ((long long)m >= at - 5 && (long long)m <= at + 5) {
              extern PPU_THREAD_LOCAL ppu_context* g_active_ctx;
              extern void ppu_dump_guest_stack(ppu_context*, const char*);
              if (g_active_ctx) {
                  char tag[48];
                  snprintf(tag, sizeof tag, "flip#%llu", m);
                  ppu_dump_guest_stack(g_active_ctx, tag);
                  { extern void ppu_dump_bctrl_ring(uint32_t, const char*);
                    ppu_dump_bctrl_ring((uint32_t)g_active_ctx->thread_id, tag); }
              }
          }
      } }

    { static int _n=0; if (getenv("FLIP_DBG") && (_n < 20 && _n++ < 20))
        fprintf(stderr, "[FLIP] SetFlipCommand(buf=%u) set=%d\n",
                bufferId, bufferId < CELL_GCM_MAX_DISPLAY_BUFFER_NUM ? s_display_buffer_set[bufferId] : -1); }
    if (bufferId >= CELL_GCM_MAX_DISPLAY_BUFFER_NUM)
        return CELL_GCM_ERROR_INVALID_VALUE;

    if (!s_display_buffer_set[bufferId])
        return CELL_GCM_ERROR_INVALID_VALUE;


    /* in_fifo: the flip sits in the command stream as a GCM_FLIP_MARKER, and
     * the drain does this half when it reaches it (see the 0xFEAD case). */
    if (!in_fifo) {
        s_current_display_buffer_id = bufferId;
        s_flip_pending = 1;   /* ticker: present BEFORE the next drain */
        s_flip_request_count++;
    }
    /* Flip requested but not yet shown: a subsequent cellGcmSetWaitFlip blocks
     * until the present thread's cellGcmTickFlip marks it done (vsync). */
    s_flip_status = CELL_GCM_FLIP_STATUS_WAITING;
    s_last_flip_time = get_timestamp_ns();

    /* Invoke via OPD resolution, not a raw call into guest code.
     *
     * On hardware this callback fires when the flip COMPLETES, at vsync --
     * which is what cellGcmTickFlip does. Calling it again here, at request
     * time, is a compatibility wart: some titles need the early edge, but a
     * title that issues its first flip while still initialising then runs its
     * handler against half-built state. Tokyo Jungle faults that way, in
     * func_001F2D3C, dereferencing a singleton it has not constructed yet.
     * GCM_FLIPCB_ONTICK=1 leaves the callback to the tick alone. */
    if (s_flip_handler_opd && g_ps3_guest_caller && !gcm_flipcb_on_tick_only())
        g_ps3_guest_caller(s_flip_handler_opd, 0, 0, 0, 0, 0, 0, 0, 0);  /* head 0 = primary display */

    return CELL_OK;
}

s32 cellGcmSetFlipCommand(u32 bufferId)
{
    if (bufferId < CELL_GCM_MAX_DISPLAY_BUFFER_NUM && s_display_buffer_set[bufferId])
        gcm_flip_note_into_fifo(bufferId);   /* v62 */
    return gcm_flip_request(bufferId, 0);
}

/* The real _cellGcmSetFlipCommand appends the flip to the command buffer, so
 * the RSX flips once it has executed everything queued before it. Firing it
 * at call time instead presented whatever the drain had reached by then --
 * and the title has not even flushed its frame's tail yet: The Simpsons Arcade
 * Game calls it with ctx->current ~2.4 KB past put, so frames showed without
 * their last draws (a dialog, a dimming overlay) for one flip. Write a marker
 * at ctx->current instead; the drain stops at it and fires the flip there,
 * exactly as it does for the 0xFEAD words a statically-linked libgcm emits.
 * Returns 0 when there is no room or ctx is not a mapped command buffer. */
static int gcm_flip_into_fifo(u32 ctx, u32 bufferId)
{
    if (!ctx || bufferId >= CELL_GCM_MAX_DISPLAY_BUFFER_NUM ||
        !s_display_buffer_set[bufferId]) return 0;
    const u32 end = vm_read32(ctx + 4), cur = vm_read32(ctx + 8);
    /* Never near the end: those bytes are the title's reserve before its
     * ring wraps, and the recycle above keys on them. */
    if (cur + 8 + GCM_RECYCLE_SLACK > end || gcm_ea2io(cur) == 0xFFFFFFFFu) return 0;
    vm_write32(cur, GCM_FLIP_MARKER | bufferId);
    /* Wrap here, on the guest thread, once the ring is nearly used -- what the
     * SDK's buffer-full callback does. Otherwise the drain thread recycles it
     * (see "recycled the title's own ring"), rewriting ctx->current and put
     * while the guest may be between advancing current and flushing: put
     * then lands on begin, the guest's closing fence is never executed, and
     * it spins on `ref` for good. Safe when the walker has left the head of
     * the ring well behind, which is the region the guest writes next. */
    const u32 begin = vm_read32(ctx), io_begin = gcm_ea2io(begin);
    if (end - (cur + 4) < GCM_FLIP_WRAP_BYTES && io_begin != 0xFFFFFFFFu &&
        s_fifo_getoff >= io_begin + GCM_RECYCLE_MARGIN) {
        vm_write32(cur + 4, 0x20000000u | io_begin);   /* JUMP -> begin */
        vm_write32(ctx + 8, begin);
        return 1;
    }
    vm_write32(ctx + 8, cur + 4);
    return 1;
}

/* cellGcmSetFlip(context, buffer_id) — immediate flip request. PSL1GHT's
 * gcmSetFlip import (NID 0xDC09357E) lands here; without it vkcube's flip was
 * a no-op, GetFlipStatus never cleared, and init timed out into exit(-1). */
s32 cellGcmSetFlip(void* context, u32 bufferId)
{
    s_flip_note_ctx = (u32)(uintptr_t)context;   /* v62: marker goes into THIS context */
    s32 r = cellGcmSetFlipCommand(bufferId);
    s_flip_note_ctx = 0;
    return r;
}

/* NID: 0xD01B570F */
s32 cellGcmSetFlipCommandWithWaitLabel(u32 bufferId, u32 labelIndex, u32 labelValue)
{
    if (labelIndex >= CELL_GCM_MAX_LABEL_COUNT)
        return CELL_GCM_ERROR_INVALID_VALUE;

    /* Wait until the label reaches the expected value (instant in HLE). Labels
     * live in guest memory (see cellGcmGetLabelAddress) so the game can poll. */
    vm_write32(GCM_LABEL_GUEST_BASE + labelIndex * GCM_LABEL_STRIDE, labelValue);

    return cellGcmSetFlipCommand(bufferId);
}

/* NID: 0xA2478CA3 */
s32 cellGcmSetPrepareFlip(void* ctx, u32 bufferId)
{
    (void)ctx;  /* command buffer context -- not used in HLE */

    if (bufferId >= CELL_GCM_MAX_DISPLAY_BUFFER_NUM)
        return CELL_GCM_ERROR_INVALID_VALUE;

    if (!s_display_buffer_set[bufferId])
        return CELL_GCM_ERROR_INVALID_VALUE;

    printf("[cellGcmSys] SetPrepareFlip(bufferId=%u)\n", bufferId);

    s_current_display_buffer_id = bufferId;
    s_flip_status = CELL_GCM_FLIP_STATUS_DONE;
    s_flip_request_count++;
    s_last_flip_time = get_timestamp_ns();

    /* Invoke the guest flip handler via OPD resolution -- s_flip_handler holds
     * the raw guest OPD (e.g. 0x530D70); calling it as a host function pointer
     * jumps into guest code and crashes. See cellGcmSetFlipCommand for why
     * GCM_FLIPCB_ONTICK exists. */
    if (s_flip_handler_opd && g_ps3_guest_caller && !gcm_flipcb_on_tick_only())
        g_ps3_guest_caller(s_flip_handler_opd, 0, 0, 0, 0, 0, 0, 0, 0);

    return CELL_OK;
}

/* NID: 0x1BFAB6EE */
CellGcmDisplayInfo* cellGcmGetDisplayBufferByFlipIndex(u32 index)
{
    u32 id = index % CELL_GCM_MAX_DISPLAY_BUFFER_NUM;
    return &s_display_buffers[id];
}

/* NID: 0x8BADE8BE */
u32 cellGcmGetCurrentDisplayBufferId(void)
{
    return s_current_display_buffer_id;
}

/* NID: 0xD9B7653E */
void cellGcmSetFlipHandler(CellGcmFlipHandler handler)
{
    /* `handler` is a guest OPD address — store as such and remember
     * the legacy host-typed value too for any callers still using
     * the old API style. Real dispatch goes through
     * cellGcmDispatchVBlank()/DispatchFlip() via g_ps3_guest_caller. */
    printf("[cellGcmSys] SetFlipHandler(opd=0x%08X)\n", (unsigned)(size_t)handler);
    s_flip_handler_opd = (u32)(size_t)handler;
    s_flip_handler = handler;
    { u32 c = s_flip_handler_opd ? vm_read32(s_flip_handler_opd) : 0; if (c) s_flip_handler_code = c; }
    gcm_update_handler_mask();
}

/* NID: 0xA547ADDE */
void cellGcmSetVBlankHandler(CellGcmVBlankHandler handler)
{
    printf("[cellGcmSys] SetVBlankHandler(opd=0x%08X)\n", (unsigned)(size_t)handler);
    s_vblank_handler_opd = (u32)(size_t)handler;
    s_vblank_handler = handler;
    { u32 c = s_vblank_handler_opd ? vm_read32(s_vblank_handler_opd) : 0; if (c) s_vblank_handler_code = c; }
    gcm_update_handler_mask();
}

/* NID: 0xF9BFCDA3 */
void cellGcmSetSecondVHandler(CellGcmSecondVHandler handler)
{
    printf("[cellGcmSys] SetSecondVHandler(opd=0x%08X)\n", (unsigned)(size_t)handler);
    s_second_v_handler_opd = (u32)(size_t)handler;
    s_second_v_handler = handler;
}

/* NID: 0x0B4B62D5 */
void cellGcmSetUserHandler(CellGcmUserHandler handler)
{
    printf("[cellGcmSys] SetUserHandler(opd=0x%08X)\n", (unsigned)(size_t)handler);
    s_user_handler_opd = (u32)(size_t)handler;
    s_user_handler = handler;
    gcm_update_handler_mask();
}

/* NID: 0x21AC3697 */
u64 cellGcmGetLastFlipTime(void)
{
    return s_last_flip_time;
}

/* ---------------------------------------------------------------------------
 * Address translation / IO mapping
 * -----------------------------------------------------------------------*/

/* NID: 0x0E6B0DFF */
/* Mirror the host tables into the guest window the title will index. */
static void gcm_publish_offset_tables(void)
{
    for (u32 i = 0; i < GCM_OFFSET_TABLE_PAGES; i++) {
        vm_write16(GCM_OFFSET_TABLE_IO_ADDR + i * 2, s_io_address_table[i]);
        vm_write16(GCM_OFFSET_TABLE_EA_ADDR + i * 2, s_ea_address_table[i]);
    }
}

s32 cellGcmGetOffsetTable(CellGcmOffsetTable* table)
{
    u32 ea = (u32)(uintptr_t)table;
    if (!ea)
        return CELL_GCM_ERROR_INVALID_VALUE;

    gcm_publish_offset_tables();

    /* The guest struct is two 4-byte EAs, not two host pointers. */
    vm_write32(ea + 0, GCM_OFFSET_TABLE_IO_ADDR);
    vm_write32(ea + 4, GCM_OFFSET_TABLE_EA_ADDR);
    return CELL_OK;
}

/* NID: 0xDB769B32 */
static u32 gcm_io_alloc(u32 size);   /* defined below */

s32 cellGcmAddressToOffset(u32 address, u32* offset)
{
    /* `offset` is a GUEST address; write big-endian via vm_write32. */
    uint32_t off_ea = (uint32_t)(uintptr_t)offset;
    if (!off_ea)
        return CELL_GCM_ERROR_INVALID_VALUE;
    /* A2O_DBG=<N>: the EA->offset conversions a title asks for. A texture bound
     * at an offset nothing ever writes usually means the offset came from an EA
     * in a space we classified wrongly, and that is only visible here. */
    { static int cap = -1, k = 0;
      if (cap < 0) { const char* e = getenv("A2O_DBG"); cap = e ? atoi(e) : 0; }
      if (cap && k < cap) { k++;
        u32 nz = 0;
        for (u32 i = 0; i < 0x400u; i += 13) if (vm_base && vm_base[address + i]) nz++;
        printf("[A2O] ea=0x%08X  (first 1KB nonzero=%u/79)\n", address, nz); } }

    /* Local memory: offset = address - localBase. Record the offset page as
     * LOCAL-derived so cellGcmResolveOffset can disambiguate it from an IO
     * (main-memory) offset with the same page number -- the two offset spaces
     * overlap, and a title with a large IO region (gcm/cube: 16MB) otherwise
     * gets its VRAM vertex/texture/FP reads resolved into the empty command
     * buffer. */
    if (address >= s_config.localAddress &&
        address < s_config.localAddress + s_config.localSize) {
        u32 loff = address - s_config.localAddress;
        if ((loff >> 20) < sizeof(s_local_offset_page)) s_local_offset_page[loff >> 20] = 1;
        vm_write32(off_ea, loff);
        return CELL_OK;
    }

    /* IO-mapped main memory: consult offset table */
    u32 page = address >> 20;
    if (page < 65536 && s_io_address_table[page] != 0xFFFF) {
        u32 io_page = s_io_address_table[page];
        vm_write32(off_ea, (io_page << 20) | (address & 0xFFFFF));
        return CELL_OK;
    }

    /* Legacy fallback for initial IO region */
    if (s_config.ioAddress != 0 && address >= s_config.ioAddress &&
        address < s_config.ioAddress + s_config.ioSize) {
        vm_write32(off_ea, address - s_config.ioAddress);
        return CELL_OK;
    }

    /* Auto-map unmapped main memory. On real hardware PSL1GHT pre-maps its
     * whole RSX heap in gcmInitBody, so its libraries never call MapMainMemory
     * before handing an EA to the RSX (Tiny3D's command ring lives in an
     * sys_mmapper region). Mirror that by mapping the 1MB page on first use.
     *
     * The bound used to be a flat `< 0x40000000`, which is not a description of
     * anything -- RSX local memory is already handled at the top of this
     * function against its real base, and the raw-SPU window at 0x30000000 sits
     * *below* the old bound, so it was not keeping that out either. What it did
     * do was exclude titles whose RSX-visible heap lives high: Virtua Fighter 5
     * maps 0x4A900000..0x4B400000 and then hands us 0x4B400040, which fell past
     * 0x40000000, skipped the auto-map, and failed -- 976 times a boot, leaving
     * the title with no offsets to draw with (20 draws in five minutes).
     *
     * So the test is now what it always meant: anything below RSX local memory
     * is main memory. Local is checked first and returns above, so reaching
     * here with an address under localAddress means an unmapped main-memory
     * page, which is exactly the case this fallback exists for. */
    /* NULL is not a buffer. Auto-mapping it costs an IO page and, worse, hands
     * the title a real-looking offset for guest address 0 -- so an EA the title
     * computed as NULL (a failed allocation, an out-param it never filled)
     * silently becomes a valid RSX offset instead of an error it could notice.
     * The Orange Box's chooser asks for exactly this, and the page it burned was
     * the one the following cellGcmMapMainMemory would otherwise have taken. */
    if (address < 0x100000u) {
        printf("[cellGcmSys] AddressToOffset REFUSED: ea 0x%08X is not a mappable "
               "address (page 0)\n", address);
        { static int nbt = 0;   /* v1119L: who asks (first 3) */
          extern PPU_THREAD_LOCAL ppu_context* g_active_ctx; extern void ppu_dump_guest_stack(ppu_context*, const char*);
          if (nbt < 3 && g_active_ctx) { nbt++; printf("[cellGcmSys] v1119L REFUSED caller lr %08X\n", (unsigned)g_active_ctx->lr); ppu_dump_guest_stack(g_active_ctx, "addr2off-null"); } }
        vm_write32(off_ea, 0);
        return CELL_GCM_ERROR_INVALID_VALUE;
    }

    u32 auto_map_limit = s_config.localAddress ? s_config.localAddress : 0x40000000u;
    if (address < auto_map_limit) {
        u32 ea_page = address & ~0xFFFFFu;
        u32 io      = gcm_io_alloc(0x100000u);
        populate_offset_table(ea_page, io, 0x100000u);
        /* Record it in the mapping list too, not just the offset table.
         *
         * These are two different structures answering two different questions,
         * and only the offset table was being filled in. The FIFO walker
         * resolves io -> ea through the MAPPING LIST (find_io_mapping_by_io),
         * so an auto-mapped range was invisible to it: a JUMP into one reported
         * "no IO mapping", the walker resynced straight to put, and every fence
         * in the skipped range was lost -- which strands a title spinning on
         * ctrl->ref with no way to see why. Both paths have to agree about what
         * is mapped. */
        IoMapping* m = find_free_mapping();
        if (m) {
            m->ea = ea_page; m->io = io; m->size = 0x100000u; m->active = 1;
            s_io_mapping_count++;
        } else {
            printf("[cellGcmSys] WARNING: auto-map has no free IO mapping slot; "
                   "the FIFO will not be able to walk io 0x%08X\n", io);
        }
        printf("[cellGcmSys] AddressToOffset: auto-mapped ea 0x%08X -> io 0x%08X\n",
               ea_page, io);
        vm_write32(off_ea, io | (address & 0xFFFFFu));
        return CELL_OK;
    }

    printf("[cellGcmSys] WARNING: AddressToOffset failed for 0x%08X\n", address);
    vm_write32(off_ea, 0);
    { static int _d = -1; if (_d < 0) _d = getenv("A2O_FAILDBG") ? 1 : 0;
      if (_d) { static int _n = 0; if ((_n < 12 && _n++ < 12))
          fprintf(stderr, "[A2O] FAILED ea=0x%08X (not in local or IO space)%c",
                  address, 10); } }
    return CELL_GCM_ERROR_FAILURE;
}

/* IO-offset space bump allocator for main-memory mappings. Starts PAST the
 * Init command-buffer region: handing out io 0 (the old s_local_mem_allocated
 * bump, which begins at 0) remapped the FIFO's own io page -- demosaic's
 * cellGcmMapMainMemory(sys heap) got offset 0, io2ea(0) then resolved into its
 * heap (all zeros) and the RSX walker consumed every frame's commands as
 * no-ops (no draws, label fence never written, guest waited forever). */
static u32 s_io_alloc_next = 0;
static u32 gcm_io_alloc(u32 size)
{
    if (s_io_alloc_next == 0)
        s_io_alloc_next = (s_config.ioSize + 0xFFFFFu) & ~0xFFFFFu;
    if (s_io_alloc_next < 0x100000u) s_io_alloc_next = 0x100000u;
    u32 io = s_io_alloc_next;
    s_io_alloc_next += (size + 0xFFFFFu) & ~0xFFFFFu;
    return io;
}

/* NID: 0x2A6FBA9C */
s32 cellGcmMapMainMemory(u32 ea, u32 size, u32* offset)
{
    /* Log the REQUEST before validating it. Every rejection below used to
     * happen before the only printf in this function, so a title whose
     * mapping we refused looked -- in the log -- like a title that never asked.
     * Virtua Fighter 5 then fails cellGcmAddressToOffset ~976 times a boot for
     * addresses immediately past the region it did get, draws 20 times instead
     * of thousands, and never flips; the refusal that would explain it was
     * invisible. Same lesson as the savedata early returns. */
    printf("[cellGcmSys] MapMainMemory(ea=0x%08X, size=0x%X)\n", ea, size);

    if (!offset) {
        printf("[cellGcmSys] MapMainMemory REFUSED: offset out-param is NULL\n");
        return CELL_GCM_ERROR_INVALID_VALUE;
    }
    /* Size and EA must both be 1MB aligned. */
    if (size == 0 || (size & 0xFFFFF) != 0) {
        printf("[cellGcmSys] MapMainMemory REFUSED: size 0x%X is not a non-zero multiple of 1MB\n", size);
        return CELL_GCM_ERROR_INVALID_ALIGNMENT;
    }
    if ((ea & 0xFFFFF) != 0) {
        printf("[cellGcmSys] MapMainMemory REFUSED: ea 0x%08X is not 1MB aligned\n", ea);
        return CELL_GCM_ERROR_INVALID_ALIGNMENT;
    }

    /* Allocate a fresh IO region past the Init command-buffer window. */
    u32 io_offset = gcm_io_alloc(size);

    IoMapping* mapping = find_free_mapping();
    if (!mapping) {
        printf("[cellGcmSys] WARNING: no free IO mapping slots\n");
        return CELL_GCM_ERROR_FAILURE;
    }

    mapping->ea     = ea;
    mapping->io     = io_offset;
    mapping->size   = size;
    mapping->active = 1;
    s_io_mapping_count++;

    populate_offset_table(ea, io_offset, size);

    vm_write32((uint32_t)(uintptr_t)offset, io_offset);   /* guest out-param */
    return CELL_OK;
}

/* NID: 0x5A41C10F */
s32 cellGcmMapEaIoAddress(u32 ea, u32 io, u32 size)
{
    /* Logged before validation, and every failure path named -- same reason as
     * cellGcmMapMainMemory above. Two of these returned silently AFTER the log
     * line as well, so even a call that got that far could vanish. */
    printf("[cellGcmSys] MapEaIoAddress(ea=0x%08X, io=0x%08X, size=0x%X)\n", ea, io, size);

    if ((ea & 0xFFFFF) != 0 || (io & 0xFFFFF) != 0) {
        printf("[cellGcmSys] MapEaIoAddress REFUSED: ea 0x%08X / io 0x%08X not 1MB aligned\n", ea, io);
        return CELL_GCM_ERROR_INVALID_ALIGNMENT;
    }
    if (size == 0 || (size & 0xFFFFF) != 0) {
        printf("[cellGcmSys] MapEaIoAddress REFUSED: size 0x%X is not a non-zero multiple of 1MB\n", size);
        return CELL_GCM_ERROR_INVALID_ALIGNMENT;
    }
    if (find_mapping_by_ea(ea) != NULL) {
        printf("[cellGcmSys] MapEaIoAddress REFUSED: ea 0x%08X is already mapped\n", ea);
        return CELL_GCM_ERROR_ADDRESS_OVERWRAP;
    }

    IoMapping* mapping = find_free_mapping();
    if (!mapping) {
        printf("[cellGcmSys] MapEaIoAddress REFUSED: no free IO mapping slots\n");
        return CELL_GCM_ERROR_FAILURE;
    }

    mapping->ea     = ea;
    mapping->io     = io;
    mapping->size   = size;
    mapping->active = 1;
    s_io_mapping_count++;

    populate_offset_table(ea, io, size);

    return CELL_OK;
}

/* NID: 0xDB23E867 (disambiguation by context) */
s32 cellGcmUnmapEaIoAddress(u32 ea)
{
    IoMapping* mapping = find_mapping_by_ea(ea);
    if (!mapping)
        return CELL_GCM_ERROR_FAILURE;

    printf("[cellGcmSys] UnmapEaIoAddress(ea=0x%08X)\n", ea);

    clear_offset_table(mapping->ea, mapping->io, mapping->size);
    mapping->active = 0;
    s_io_mapping_count--;

    return CELL_OK;
}

/* NID: 0x3B9BD5BD */
s32 cellGcmUnmapIoAddress(u32 io)
{
    IoMapping* mapping = find_mapping_by_io(io);
    if (!mapping)
        return CELL_GCM_ERROR_FAILURE;

    printf("[cellGcmSys] UnmapIoAddress(io=0x%08X)\n", io);

    clear_offset_table(mapping->ea, mapping->io, mapping->size);
    mapping->active = 0;
    s_io_mapping_count--;

    return CELL_OK;
}

/* NID: 0xC47D0812 */
s32 cellGcmIoOffsetToAddress(u32 ioOffset, u32* ea)
{
    uint32_t ea_ea = (uint32_t)(uintptr_t)ea;   /* guest out-param */
    if (!ea_ea)
        return CELL_GCM_ERROR_INVALID_VALUE;

    u32 page = ioOffset >> 20;
    if (page < 65536 && s_ea_address_table[page] != 0xFFFF) {
        u32 ea_page = s_ea_address_table[page];
        vm_write32(ea_ea, (ea_page << 20) | (ioOffset & 0xFFFFF));
        return CELL_OK;
    }

    printf("[cellGcmSys] WARNING: IoOffsetToAddress failed for 0x%08X\n", ioOffset);
    vm_write32(ea_ea, 0);
    return CELL_GCM_ERROR_FAILURE;
}

/* Resolve an RSX FIFO offset (from a vertex-array / texture / surface method)
 * to a guest effective address the backend can read via vm_base + addr.
 *
 * An RSX offset refers to either IO-mapped *main* memory or *local* video
 * memory, distinguished on hardware by a per-object location bit that the
 * command stream no longer carries by the time the backend sees it. Resolve by
 * table: if the offset's 1MB page is IO-mapped, it's main memory; otherwise it
 * is local video memory (localAddress + offset). IO mappings are sparse (the
 * command/IO window is a few MB) while local objects sit high in the local
 * heap, so this disambiguates every real case — only offsets inside the small
 * IO window are treated as main. */
u32 cellGcmResolveOffset(u32 offset)
{
    u32 page = offset >> 20;
    /* A page the guest derived from a LOCAL EA resolves to VRAM even when the
     * IO table also covers that page number (overlapping offset spaces). */
    if (page < sizeof(s_local_offset_page) && s_local_offset_page[page])
        return s_config.localAddress + offset;
    if (page < 65536) {
        u16 ea_page = s_ea_address_table[page];
        if (ea_page != 0 && ea_page != 0xFFFF)
            return ((u32)ea_page << 20) | (offset & 0xFFFFF);
    }
    return s_config.localAddress + offset;
}

/* IO-table-first resolve: skip the local-page shortcut and ask the IO offset
 * table, falling back to VRAM only if the page is unmapped. cellGcmResolveOffset
 * prefers local for any page the guest ever derived from a local EA, which is
 * right for a title that really does keep the data in VRAM but wrong for one
 * that builds its textures in main memory on pages whose numbers happen to
 * collide. Returns 0 when the IO table has no mapping, so a caller can tell
 * "not IO-mapped" from "IO-mapped at 0". */
u32 cellGcmResolveIO(u32 offset)
{
    u32 page = offset >> 20;
    if (page < 65536) {
        u16 ea_page = s_ea_address_table[page];
        if (ea_page != 0 && ea_page != 0xFFFF)
            return ((u32)ea_page << 20) | (offset & 0xFFFFF);
    }
    return 0;
}

/* Location-aware resolve. RSX offsets are TWO overlapping number spaces --
 * LOCAL (VRAM, ea = localAddress + offset) and MAIN (IO-mapped, via the
 * offset table). cellGcmResolveOffset guesses table-first, which breaks when
 * a title's IO region is large enough to cover the same page numbers as its
 * local allocations (gcm/cube: ioSize 16MB, FP ucode at local 0xB90000 --
 * table hit returned the empty command-buffer EA instead of VRAM). Callers
 * that KNOW the location (texture format bits, SET_SHADER_PROGRAM bits) must
 * use this. `local` != 0 -> local memory. */
u32 cellGcmResolveLocated(int local, u32 offset)
{
    if (local)
        return s_config.localAddress + offset;
    return cellGcmResolveOffset(offset);
}

/* ---------------------------------------------------------------------------
 * Label / report / timestamp
 * -----------------------------------------------------------------------*/

/* NID: 0xF80196C1 */
u32* cellGcmGetLabelAddress(u8 index)
{
    if (index >= CELL_GCM_MAX_LABEL_COUNT) {
        printf("[cellGcmSys] WARNING: GetLabelAddress index %u out of range\n", index);
        return NULL;
    }
    return (u32*)(uintptr_t)(GCM_LABEL_GUEST_BASE + (u32)index * GCM_LABEL_STRIDE);
}

/* NID: 0x8572ADE4 */
CellGcmReportData* cellGcmGetReportDataAddress(u32 index)
{
    if (index >= CELL_GCM_MAX_REPORT_COUNT) {
        printf("[cellGcmSys] WARNING: GetReportDataAddress index %u out of range\n", index);
        return NULL;
    }
    return &s_report_data[index];
}

/* NID: 0x97FC4B73 */
/* NID: 0x99D397AC. The value NV4097_GET_REPORT stored at this index (the
 * walker fills it). Unregistered, every call returned a faked 0. */
u32 cellGcmGetReport(u32 type, u32 index)
{
    { static int n = 0;
      if (n < 8 && n++ < 8) printf("[cellGcmSys] GetReport(type=%u, index=%u) -> %u%c", type, index,
                          index < CELL_GCM_MAX_REPORT_COUNT ? s_report_data[index].value : 0u, 10); }
    if (index >= CELL_GCM_MAX_REPORT_COUNT) return 0;
    return s_report_data[index].value;
}

u64 cellGcmGetTimeStamp(u32 index)
{
    if (index >= CELL_GCM_MAX_REPORT_COUNT)
        return 0;

    /*
     * On real hardware this reads the RSX timestamp for a given report index.
     * We return the host timestamp in nanoseconds as a reasonable approximation.
     */
    s_report_data[index].timestamp = get_timestamp_ns();
    return s_report_data[index].timestamp;
}

/* ---------------------------------------------------------------------------
 * Tile / Zcull configuration
 * -----------------------------------------------------------------------*/

/* NID: 0x0B4B62D5 */
s32 cellGcmSetTile(u8 index, u8 location, u32 offset, u32 size,
                   u32 pitch, u8 comp, u16 base, u8 bank)
{
    if (index >= CELL_GCM_MAX_TILE_COUNT)
        return CELL_GCM_ERROR_INVALID_VALUE;

    printf("[cellGcmSys] SetTile(index=%u, loc=%u, offset=0x%X, size=0x%X, pitch=%u)\n",
           index, location, offset, size, pitch);

    s_tiles[index].offset = offset;
    s_tiles[index].size   = size;
    s_tiles[index].pitch  = pitch;
    s_tiles[index].format = comp;
    s_tiles[index].base   = base;
    s_tiles[index].bank   = bank;
    s_tiles[index].bound  = 0;

    /* Build tile register value (simplified) */
    s_tiles[index].tile  = (location << 0) | (pitch << 8);
    s_tiles[index].limit = offset + size - 1;

    return CELL_OK;
}

/* NID: 0x06EDEA25 */
s32 cellGcmSetZcull(u8 index, u32 offset, u32 width, u32 height,
                    u32 cullStart, u32 zFormat, u32 aaFormat,
                    u32 zcullDir, u32 zcullFormat,
                    u32 sFunc, u32 sRef, u32 sMask)
{
    if (index >= CELL_GCM_MAX_ZCULL_COUNT)
        return CELL_GCM_ERROR_INVALID_VALUE;

    printf("[cellGcmSys] SetZcull(index=%u, offset=0x%X, %ux%u)\n",
           index, offset, width, height);

    s_zcull[index].offset      = offset;
    s_zcull[index].width       = width;
    s_zcull[index].height      = height;
    s_zcull[index].cullStart   = cullStart;
    s_zcull[index].zFormat     = zFormat;
    s_zcull[index].aaFormat    = aaFormat;
    s_zcull[index].zcullDir    = zcullDir;
    s_zcull[index].zcullFormat = zcullFormat;
    s_zcull[index].sFunc       = sFunc;
    s_zcull[index].sRef        = sRef;
    s_zcull[index].sMask       = sMask;
    s_zcull[index].bound       = 0;

    return CELL_OK;
}

/* NID: 0x2BB1F1E5 */
s32 cellGcmBindTile(u8 index)
{
    if (index >= CELL_GCM_MAX_TILE_COUNT)
        return CELL_GCM_ERROR_INVALID_VALUE;

    printf("[cellGcmSys] BindTile(index=%u)\n", index);
    s_tiles[index].bound = 1;
    return CELL_OK;
}

/* NID: 0x1F61F9D3 */
s32 cellGcmBindZcull(u8 index)
{
    if (index >= CELL_GCM_MAX_ZCULL_COUNT)
        return CELL_GCM_ERROR_INVALID_VALUE;

    printf("[cellGcmSys] BindZcull(index=%u)\n", index);
    s_zcull[index].bound = 1;
    return CELL_OK;
}

/* NID: 0x75843042 */
s32 cellGcmUnbindTile(u8 index)
{
    if (index >= CELL_GCM_MAX_TILE_COUNT)
        return CELL_GCM_ERROR_INVALID_VALUE;

    printf("[cellGcmSys] UnbindTile(index=%u)\n", index);
    s_tiles[index].bound = 0;
    return CELL_OK;
}

/* NID: 0xA093BE1C */
s32 cellGcmUnbindZcull(u8 index)
{
    if (index >= CELL_GCM_MAX_ZCULL_COUNT)
        return CELL_GCM_ERROR_INVALID_VALUE;

    printf("[cellGcmSys] UnbindZcull(index=%u)\n", index);
    s_zcull[index].bound = 0;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Misc
 * -----------------------------------------------------------------------*/

/* NID: 0x107BF789
 * SDK prototype (cell/gcm.h): uint32_t cellGcmGetTiledPitchSize(const uint32_t size)
 * -- returns the pitch BY VALUE in r3 (no out-pointer!). The old signature here
 * took (size, u32* pitch-out) and returned a status code, so every caller got 0
 * / an error code as its "pitch" -- gcm/cube's `bne` on the result then skipped
 * its entire config+framebuffer-alloc block (GetConfiguration never ran, local
 * memory base stayed 0, AddressToOffset(0) auto-mapped a bogus IO alias, and the
 * FIFO teleported into it). Returns 0 only for size > max, like hardware. */
u32 cellGcmGetTiledPitchSize(u32 size)
{
    u32 r = 0;
    /* Smallest valid tiled pitch >= size (RSX supports only specific pitches). */
    if (size) {
        for (int i = 0; i < s_valid_pitch_count; i++) {
            if (s_valid_pitches[i] >= size) { r = s_valid_pitches[i]; break; }
        }
    }
    { static int _n=0; if ((_n < 4 && _n++ < 4)) fprintf(stderr, "[cellGcmSys] GetTiledPitchSize(0x%X) -> 0x%X\n", size, r); }
    return r;
}

/* NID: 0xBC982946 */
void cellGcmSetDebugOutputLevel(u32 level)
{
    printf("[cellGcmSys] SetDebugOutputLevel(level=%u)\n", level);
    s_debug_level = level;
}

/* ---------------------------------------------------------------------------
 * Additional functions needed by Tokyo Jungle
 * -----------------------------------------------------------------------*/

/* Notify data area — similar to report data, 16 bytes per slot */
static u8 s_notify_data[256 * 16];

void* cellGcmGetNotifyDataAddress(u32 index)
{
    if (index >= 256) return NULL;
    return &s_notify_data[index * 16];
}

/* Timestamp location — returns CELL_GCM_LOCATION_LOCAL or MAIN */
/* The Location suffix names an ARGUMENT, not an out-parameter: it is the
 * caller saying which memory the report lives in, CELL_GCM_LOCATION_LOCAL or
 * _MAIN, exactly as cellGcmGetReportDataLocation and
 * cellGcmGetReportDataAddressLocation take it three hundred lines below. This
 * one alone was declared u32* and treated the value as a guest address to
 * write the location INTO, which inverts the parameter and loses the return.
 *
 * A title passing CELL_GCM_LOCATION_MAIN therefore had a 1 taken for an
 * effective address, and vm_write32 stored four bytes at guest address 1.
 * Yakuza: Dead Souls does that during display setup and dies there.
 *
 * It returns the timestamp, like cellGcmGetTimeStamp, whose body this is with
 * a location the report table does not distinguish. */
u64 cellGcmGetTimeStampLocation(u32 index, u32 location)
{
    (void)location;

    if (index >= CELL_GCM_MAX_REPORT_COUNT)
        return 0;

    s_report_data[index].timestamp = get_timestamp_ns();
    return s_report_data[index].timestamp;
}

/* SetTileInfo — alternative to SetTile with same parameters */
s32 cellGcmSetTileInfo(u8 index, u8 location, u32 offset, u32 size,
                       u32 pitch, u8 comp, u16 base, u8 bank)
{
    /* Delegates to existing SetTile */
    return cellGcmSetTile(index, location, offset, size, pitch, comp, base, bank);
}

/* Default FIFO size — configures command buffer size before init */
static u32 s_default_fifo_size = 0x40000; /* 256KB default */

s32 cellGcmSetDefaultFifoSize(u32 size)
{
    printf("[cellGcmSys] SetDefaultFifoSize(0x%X)\n", size);
    s_default_fifo_size = size;
    return CELL_OK;
}

/* Internal flip commands — called directly by some games */
/* libgcm's internal exports pass the command context FIRST -- taking
 * bufferId as arg1 made the range check eat the context pointer, so the
 * flip counter never advanced (wave: presents free-ran 4x per frame,
 * layout flashing/zooming). */
/* NID 0x21397818 is gcmSetFlipCommand (PSL1GHT libgcm_sys exports.h) -- it
 * was misassigned to cellGcmGetLabelAddress (real NID 0xF80196C1), so every
 * import-based flip (LBP: cellGcmSetFlip -> stub 0x7482FC(ctx, id)) landed
 * in the label getter and no flip was EVER issued: the intro-movie pump ran,
 * drew, "flipped" into the wrong function, and the screen stayed black. */
/* NID: 0x21397818 */
s32 _cellGcmSetFlipCommand(void* ctx, u32 bufferId)
{
    if (gcm_flip_into_fifo((u32)(uintptr_t)ctx, bufferId))
        return gcm_flip_request(bufferId, 1);
    s_flip_note_ctx = (u32)(uintptr_t)ctx;   /* v62 */
    s32 r = cellGcmSetFlipCommand(bufferId);
    s_flip_note_ctx = 0;
    return r;
}

/* NID: 0xD8F88E1A */
s32 _cellGcmSetFlipCommandWithWaitLabel(void* ctx, u32 bufferId,
                                        u32 labelIndex, u32 labelValue)
{
    s_flip_note_ctx = (u32)(uintptr_t)ctx;   /* v62 */
    s32 r = cellGcmSetFlipCommandWithWaitLabel(bufferId, labelIndex, labelValue);
    s_flip_note_ctx = 0;
    return r;
}

/* ---------------------------------------------------------------------------
 * Additional functions (RPCS3 parity)
 * -----------------------------------------------------------------------*/

/* FIFO command buffer callback — called when put pointer wraps.
 * In recomp we don't have a real GPU consuming FIFO, so this is a no-op. */
s32 cellGcmCallback(void* context, u32 count)
{
    (void)context;
    (void)count;
    return CELL_OK;
}

/* Map RSX local memory — returns the base address and size of local VRAM */
s32 cellGcmMapLocalMemory(u32* address, u32* size)
{
    if (!address || !size)
        return CELL_GCM_ERROR_INVALID_VALUE;

    /* Guest out-params, like every other pointer parameter in this file. */
    vm_write32((uint32_t)(uintptr_t)address, s_config.localAddress);
    vm_write32((uint32_t)(uintptr_t)size,    s_config.localSize);

    printf("[cellGcmSys] MapLocalMemory(address=0x%08X, size=0x%X)\n",
           s_config.localAddress, s_config.localSize);
    return CELL_OK;
}

/* Shutdown RSX — clear all state */
void cellGcmTerminate(void)
{
    printf("[cellGcmSys] Terminate\n");

    s_gcm_initialized = 0;
    s_flip_mode   = CELL_GCM_DISPLAY_VSYNC;
    s_flip_status = CELL_GCM_FLIP_STATUS_DONE;
    s_debug_level = CELL_GCM_DEBUG_LEVEL0;
    s_vblank_count = 0;
    s_io_map_reserved = 0;
    s_default_fifo_mode = 0;
    GCM_USER_STORE(0);
    GCM_USER_PENDING_STORE(0);

    memset(s_display_buffers, 0, sizeof(s_display_buffers));
    memset(s_display_buffer_set, 0, sizeof(s_display_buffer_set));
    memset(&s_config, 0, sizeof(s_config));
    memset(&s_control, 0, sizeof(s_control));
    memset(s_io_mappings, 0, sizeof(s_io_mappings));
    memset(s_tiles, 0, sizeof(s_tiles));
    memset(s_zcull, 0, sizeof(s_zcull));
    memset(s_report_data, 0, sizeof(s_report_data));
    memset(s_labels, 0, sizeof(s_labels));
    memset(s_io_address_table, 0xFF, sizeof(s_io_address_table));
    memset(s_ea_address_table, 0xFF, sizeof(s_ea_address_table));

    s_flip_handler     = NULL;
    s_vblank_handler   = NULL;
    s_user_handler     = NULL;
    s_second_v_handler = NULL;
    s_graphics_handler = NULL;
    s_queue_handler    = NULL;
    s_local_mem_allocated = 0;
    s_io_mapping_count = 0;
    /* Unmapped = 0xFFFF. The tables are static (zero-initialized), so without
     * this every unmapped EA page silently aliased io page 0 (and vice versa),
     * producing plausible-but-wrong offsets instead of a translation failure. */
    memset(s_io_address_table, 0xFF, sizeof(s_io_address_table));
    memset(s_ea_address_table, 0xFF, sizeof(s_ea_address_table));
    s_current_display_buffer_id = 0;
    s_last_flip_time = 0;
}

/* Return remaining IO map space (256MB total IO space minus already mapped) */
u32 cellGcmGetMaxIoMapSize(void)
{
    /* RSX has a 256MB IO address window */
    u32 total_io = 256 * 1024 * 1024;
    u32 used = 0;

    for (int i = 0; i < CELL_GCM_MAX_IO_MAPPINGS; i++) {
        if (s_io_mappings[i].active)
            used += s_io_mappings[i].size;
    }

    return (used < total_io) ? (total_io - used) : 0;
}

/* Reserve IO map space (pre-allocation for future mappings) */
s32 cellGcmReserveIoMapSize(u32 size)
{
    u32 max_size = cellGcmGetMaxIoMapSize();
    if (size > max_size - s_io_map_reserved)
        return CELL_GCM_ERROR_FAILURE;

    s_io_map_reserved += size;
    return CELL_OK;
}

/* Unreserve IO map space */
s32 cellGcmUnreserveIoMapSize(u32 size)
{
    if (size > s_io_map_reserved)
        return CELL_GCM_ERROR_FAILURE;

    s_io_map_reserved -= size;
    return CELL_OK;
}

/* Return incrementing VBlank counter */
u32 cellGcmGetVBlankCount(void)
{
    /* Approximate: increment each time queried (games use this for timing) */
    return s_vblank_count++;
}

/* Return tile info for a given index */
CellGcmTileInfo* cellGcmGetTileInfo(u8 index)
{
    if (index >= CELL_GCM_MAX_TILE_COUNT) {
        printf("[cellGcmSys] WARNING: GetTileInfo index %u out of range\n", index);
        return NULL;
    }
    return &s_tiles[index];
}

/* Return zcull info for a given index */
CellGcmZcullInfo* cellGcmGetZcullInfo(u8 index)
{
    if (index >= CELL_GCM_MAX_ZCULL_COUNT) {
        printf("[cellGcmSys] WARNING: GetZcullInfo index %u out of range\n", index);
        return NULL;
    }
    return &s_zcull[index];
}

/* Return display buffer info by index */
CellGcmDisplayInfo* cellGcmGetDisplayInfo(u32 index)
{
    if (index >= CELL_GCM_MAX_DISPLAY_BUFFER_NUM) {
        printf("[cellGcmSys] WARNING: GetDisplayInfo index %u out of range\n", index);
        return NULL;
    }
    return &s_display_buffers[index];
}

/* Set default FIFO mode (before init) */
void cellGcmInitDefaultFifoMode(s32 mode)
{
    printf("[cellGcmSys] InitDefaultFifoMode(mode=%d)\n", mode);
    s_default_fifo_mode = mode;
}

/* Set command buffer to defaults (reset put/get pointers) */
void cellGcmSetDefaultCommandBuffer(void)
{
    static unsigned s_v1037_calls = 0;   /* v1037E: called 10-60x/s in gameplay -- log the first few only */
    const int v1037_log = (s_v1037_calls++ < 8);
    if (v1037_log) printf("[cellGcmSys] SetDefaultCommandBuffer\n");
    s_control.put = 0;
    s_control.get = 0;
    s_control.ref = 0;

    /* On hardware this re-points gCellGcmCurrentContext at the default context
     * cellGcmInit built. Zeroing a host-side struct is not that: a title that
     * calls it to go back to the big default buffer stays on whatever smaller
     * segment it had switched to.
     *
     * GCM_DEFAULT_CTX_REPOINT=1 does the real thing -- write our context's EA
     * into the guest variable cellGcmInit's ctx_out pointed at, and rewind that
     * context. Off by default because a title managing its own segment chain
     * may hold pointers into the old one; this exists to be measured, not
     * assumed.
     *
     * Measured once now, on Virtua Fighter 5, which is the case this was
     * written for. VF5's AMGL keeps its own command buffers (it repoints
     * gCellGcmCurrentContext at a 512 KB segment of its own inside the 4 MB it
     * maps) and calls this entry to recover when one fills. Getting a host-side
     * struct zeroed instead leaves it on the exhausted segment, so it allocates
     * another, fills that, and asks again:
     *
     *     [AMGL]:[ERROR] Command Buffer Overflow!   x91, 636 calls to here
     *
     * With the repoint: **0 overflows**, and the command packets reaching the
     * draw engine go from 12,756 to 15,097 over the same five minutes. Still
     * one title's evidence, so still opt-in -- but it is no longer unmeasured,
     * and a port whose title manages its own segments should try it. */
    if (v1037_log) {   /* v1037D: what the switch looks like (first 8 calls) */
        u32 optr = s_gcm_ctx_out_ea ? vm_read32(s_gcm_ctx_out_ea) : 0;
        u32 d = s_gcm_context_ea;
        fprintf(stderr, "[V1037D] SetDefaultCommandBuffer: current ctx 0x%08X (begin %08X end %08X current %08X), "
                "default ctx 0x%08X (begin %08X end %08X current %08X), put 0x%08X, walker get 0x%08X%c",
                optr, optr ? vm_read32(optr) : 0, optr ? vm_read32(optr + 4) : 0, optr ? vm_read32(optr + 8) : 0,
                d, d ? vm_read32(d) : 0, d ? vm_read32(d + 4) : 0, d ? vm_read32(d + 8) : 0,
                vm_read32(GCM_CONTROL_GUEST_ADDR + 0), s_fifo_getoff, 10);
    }
    /* v1037E GCM_DEFAULT_CTX_REPOINT=2: hardware/RPCS3 semantics -- only point gCellGcmCurrentContext back at the
     * default context; its begin/current stay as they are. Godfather calls this 10-60x per second in gameplay,
     * after building command lists in its own context (0x10267238, buffer 0x4991E798), while the default
     * context is mid-segment (current = begin + 0x1CC, put still at begin). Mode 1's rewind to begin threw
     * those unflushed commands away every time -> black world while moving, missing minimap roads. With no
     * rewind the walker simply reads on in order, so GCM_DEFAULT_CTX_FOLLOW has nothing to do. */
    static int s_rp = -2;
    if (s_rp == -2) { const char* e = getenv("GCM_DEFAULT_CTX_REPOINT"); s_rp = (e && *e) ? atoi(e) : 0;
        fprintf(stderr, "[V1037E] SetDefaultCommandBuffer mode %d (GCM_DEFAULT_CTX_REPOINT: 0 = no repoint, 1 = repoint + rewind current, 2 = repoint only)%c", s_rp, 10); }
    if (s_rp == 2 && s_gcm_ctx_out_ea && s_gcm_context_ea) {
        vm_write32(s_gcm_ctx_out_ea, s_gcm_context_ea);
    } else if (s_rp == 1 && s_gcm_ctx_out_ea && s_gcm_context_ea) {
        vm_write32(s_gcm_context_ea + 0x8, vm_read32(s_gcm_context_ea + 0x0));
        if (v1037_follow_on()) {   /* v1037D */
            u32 gput = vm_read32(GCM_CONTROL_GUEST_ADDR + 0);
            u32 nio  = gcm_ea2io(vm_read32(s_gcm_context_ea + 0x8));
            if (nio != 0xFFFFFFFFu && nio != gput) {
                s_v1037_old_put = gput; s_v1037_new_io = nio; s_v1037_follow = 1;
                fprintf(stderr, "[V1037D] switch armed: walker goes from put 0x%08X to 0x%08X once it gets there%c", gput, nio, 10);
            }
        }
        vm_write32(s_gcm_ctx_out_ea, s_gcm_context_ea);
        if (v1037_log) printf("[cellGcmSys] SetDefaultCommandBuffer: repointed "
               "gCellGcmCurrentContext@0x%08X -> 0x%08X\n",
               s_gcm_ctx_out_ea, s_gcm_context_ea);
    }
}

/* Debug dump — no-op in recomp */
void cellGcmDumpGraphicsError(void)
{
    printf("[cellGcmSys] DumpGraphicsError (no-op)\n");
}

/* Default command word size: 0x400 (1024) words */
u32 cellGcmGetDefaultCommandWordSize(void)
{
    return 0x400;
}

/* Default segment word size: 0x100 (256) words */
u32 cellGcmGetDefaultSegmentWordSize(void)
{
    return 0x100;
}

/* Immediate flip — perform flip right now */
s32 cellGcmSetFlipImmediate(u32 bufferId)
{
    printf("[cellGcmSys] SetFlipImmediate(bufferId=%u)\n", bufferId);
    return cellGcmSetFlipCommand(bufferId);
}

/* Set flip status directly */
void cellGcmSetFlipStatus(u32 status)
{
    s_flip_status = status;
}

/* Store graphics handler callback */
void cellGcmSetGraphicsHandler(CellGcmGraphicsHandler handler)
{
    printf("[cellGcmSys] SetGraphicsHandler(%p)\n", (void*)(size_t)handler);
    s_graphics_handler = handler;
}

/* Store queue handler callback */
void cellGcmSetQueueHandler(CellGcmQueueHandler handler)
{
    printf("[cellGcmSys] SetQueueHandler(%p)\n", (void*)(size_t)handler);
    s_queue_handler = handler;
}

/* Set second V frequency */
void cellGcmSetSecondVFrequency(u32 freq)
{
    printf("[cellGcmSys] SetSecondVFrequency(%u)\n", freq);
    s_second_v_frequency = freq;
}

/* Set VBlank frequency */
void cellGcmSetVBlankFrequency(u32 freq)
{
    printf("[cellGcmSys] SetVBlankFrequency(%u)\n", freq);
    s_vblank_frequency = freq;
}

/* Store user command value */
void cellGcmSetUserCommand(u32 cmd)
{
    GCM_USER_STORE(cmd);
}

/* Called when a FIFO consumer retires a user-interrupt method. Publishing
 * the cause and pending marker is atomic; guest code runs later in the pump.
 * Like the driver cause register, multiple pending commands coalesce. */
void cellGcmQueueUserCommand(u32 cmd)
{
    GCM_USER_STORE(cmd);
    GCM_USER_PENDING_STORE((1ULL << 32) | cmd);
}

/* Invalidate a tile region (unbind + clear) */
s32 cellGcmSetInvalidateTile(u8 index)
{
    if (index >= CELL_GCM_MAX_TILE_COUNT)
        return CELL_GCM_ERROR_INVALID_VALUE;

    printf("[cellGcmSys] SetInvalidateTile(index=%u)\n", index);

    s_tiles[index].bound = 0;
    memset(&s_tiles[index], 0, sizeof(CellGcmTileInfo));
    return CELL_OK;
}

/* EA IO address remap — stub, sorting not needed in recomp */
void cellGcmSortRemapEaIoAddress(void)
{
    /* No-op: IO address remapping not required for recomp */
}

/* Report data with location parameter */
CellGcmReportData* cellGcmGetReportDataAddressLocation(u32 index, u32 location)
{
    (void)location;  /* We only have one report area */

    if (index >= CELL_GCM_MAX_REPORT_COUNT) {
        printf("[cellGcmSys] WARNING: GetReportDataAddressLocation index %u out of range\n", index);
        return NULL;
    }
    return &s_report_data[index];
}

/* Get report value at index with location */
u32 cellGcmGetReportDataLocation(u32 index, u32 location)
{
    (void)location;

    if (index >= CELL_GCM_MAX_REPORT_COUNT)
        return 0;

    return s_report_data[index].value;
}

/* ---------------------------------------------------------------------------
 * VRAM upload registry
 *
 * A title whose texture uploads land at an address that does not match the
 * offset it later programs into SET_TEXTURE_OFFSET leaves the sampler reading
 * untouched memory. Recording what was uploaded lets the texture path find the
 * real bytes by size instead of trusting the offset. Diagnostic bridge: it is
 * a lookup, not a fix for the offset bookkeeping itself.
 * -----------------------------------------------------------------------*/
#define RSX_UPLOAD_LOG 256
static struct { u32 dst, size; int claimed; } s_uploads[RSX_UPLOAD_LOG];
static u32 s_upload_n;

void rsx_note_vram_upload(u32 dst_ea, u32 size)
{
    if (!size) return;
    s_uploads[s_upload_n % RSX_UPLOAD_LOG].dst     = dst_ea;
    s_uploads[s_upload_n % RSX_UPLOAD_LOG].size    = size;
    s_uploads[s_upload_n % RSX_UPLOAD_LOG].claimed = 0;
    s_upload_n++;
}

/* OLDEST unclaimed upload of exactly `want` bytes, claimed on return. Matching
 * by size alone hands every same-sized surface the same image; consuming them in
 * order pairs the Nth bind of a size with the Nth upload of that size, which is
 * the order a title loads and then draws its materials in. Returns 0 if none. */
u32 rsx_find_vram_upload(u32 want)
{
    u32 n = s_upload_n < RSX_UPLOAD_LOG ? s_upload_n : RSX_UPLOAD_LOG;
    u32 first = s_upload_n > RSX_UPLOAD_LOG ? s_upload_n - RSX_UPLOAD_LOG : 0;
    for (u32 i = 0; i < n; i++) {
        u32 idx = (first + i) % RSX_UPLOAD_LOG;
        if (s_uploads[idx].size == want && !s_uploads[idx].claimed) {
            s_uploads[idx].claimed = 1;
            return s_uploads[idx].dst;
        }
    }
    return 0;
}

/* Let the same pairing repeat next frame. */
void rsx_reset_upload_claims(void)
{
    for (u32 i = 0; i < RSX_UPLOAD_LOG; i++) s_uploads[i].claimed = 0;
}

/* ---------------------------------------------------------------------------
 * sys_rsx_* bridge (libs/video/sys_rsx.c)
 *
 * A PS3 firmware module links libgcm statically and drives RSX through the lv2
 * syscalls instead of importing cellGcmSys, so none of the HLE entry points above
 * ever run. Everything the FIFO walker needs is still here -- it just has to be
 * reachable from the syscall layer. These four accessors are that surface; the
 * walker, the method decoder and rsx_live_draw are untouched.
 * -----------------------------------------------------------------------*/

/* Bring the GCM state up the way cellGcmInit would, minus the IO window (the
 * driver maps that itself with sys_rsx_context_iomap). Returns the guest base of
 * RSX local memory, which is what sys_rsx_memory_allocate hands back. */
u32 cellGcm_syscall_bringup(u32 local_size)
{
    if (!s_gcm_initialized)
        cellGcmInit(0x10000, 0, 0);
    if (local_size)
        s_config.localSize = local_size;
    s_gcm_syscall_mode = 1;
    return s_config.localAddress;
}

/* Guest EA of the put/get/ref triple. The driver gets this as
 * sys_rsx_context_allocate's lpar_dma_control + 0x40 (verified in the image:
 * `lwz r3,0x18(r9)` / `addi r3,r3,64`), so the syscall returns this minus 0x40
 * and the driver's own flushes land exactly where the walker reads. */
u32 cellGcm_control_guest_addr(void) { return GCM_CONTROL_GUEST_ADDR; }

/* sys_rsx_context_iomap / iounmap: the same 1MB-page offset table
 * cellGcmMapEaIoAddress fills, which is what gcm_io2ea() and every offset
 * resolution read. */
void cellGcm_syscall_iomap(u32 ea, u32 io, u32 size)
{
    populate_offset_table(ea, io, size);
    if (s_io_mapping_count < CELL_GCM_MAX_IO_MAPPINGS) {
        s_io_mappings[s_io_mapping_count].ea     = ea;
        s_io_mappings[s_io_mapping_count].io     = io;
        s_io_mappings[s_io_mapping_count].size   = size;
        s_io_mappings[s_io_mapping_count].active = 1;
        s_io_mapping_count++;
    }
    if (!s_config.ioSize) { s_config.ioAddress = ea; s_config.ioSize = size; }
}

void cellGcm_syscall_iounmap(u32 io, u32 size)
{
    u32 page = io >> 20;
    u32 ea = (page < 65536 && s_ea_address_table[page] != 0xFFFF)
             ? ((u32)s_ea_address_table[page] << 20) : 0;
    if (ea) clear_offset_table(ea, io, size);
}

/* sys_rsx_context_attribute(0x001): the driver's explicit FIFO reset. Move the
 * walker with it -- leaving s_fifo_getoff where it was makes the next drain walk
 * from a stale offset through whatever the driver has since overwritten. */
void cellGcm_syscall_set_fifo(u32 put, u32 get)
{
    vm_write32(GCM_CONTROL_GUEST_ADDR + 0, put);
    vm_write32(GCM_CONTROL_GUEST_ADDR + 4, get);
    s_fifo_getoff  = get;
    s_fifo_calloff = 0;
}
