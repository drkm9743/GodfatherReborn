/* gf_footcam.c -- v1046L GTA-style keyboard/mouse look on foot.
 *
 * User: "the way I use the mouse to look around is way too slow and restricted, I want it to feel more like GTA".
 * The mouse used to drive the right stick. Godfather's on-foot camera is func_00055E40 (found with GF_CAMPROBE in
 * soaks Y10/Y11: ~1 call per frame in gameplay, the only camera update running on foot; tuning block obj+0x78):
 *   yaw: every frame the camera offset around the player is rotated by ONE angle f31 (func_004D2440 at 0x56D08,
 *        loc_00056CF4). f31 = (player yaw obj+0x244 - current camera yaw) -> the swing back behind the player, or the
 *        stick turn, both rate-limited by the tuning speeds (+0x94 / +0x9C via func_000587FC). That cap is the "slow",
 *        the swing-behind is the "restricted".
 *   pitch: offset obj+0x228 (rad, negative = look up; stick-driven through func_00058AA4 with velocity +0x224 and an
 *        automatic return to 0), clamped at loc_000565D0 to +-(tuning+0x70 = obj+0xE8, degrees) -- 32 deg here.
 * With the mouse as the active device (cellPad.c g_pad_look_fn) this file owns both:
 *   yaw   f31 = exactly the mouse turn of this frame (no rate cap, no swing behind the player -> the view stays where
 *         you point it while Aldo runs around, like GTA);
 *   pitch obj+0x228 = own value += mouse Y, clamped to +-GF_FOOTCAM_PITCH (default 65 deg; the game's limit obj+0xE8
 *         is raised to it while active and restored after), velocity zeroed -> no drift back.
 * A controller, free aim, lock-on, grabs and gestures keep the original behaviour.
 * Env: GF_FOOTCAM=0 off; GF_FOOTCAM_DEG (cellPad, 0.12 deg/count x F10 mouse speed/450); GF_FOOTCAM_INVX / _INVY
 *      (cellPad); GF_FOOTCAM_YAWSIGN (-1: positive game angle = turn left); GF_FOOTCAM_PITCH (65);
 *      GF_FOOTCAM_LOG (1: [footcam] lines in the out.log).
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

extern uint8_t* vm_base;
extern int (*volatile g_pad_look_fn)(float dyaw, float dpitch);
extern void (*volatile g_pad_camcycle_fn)(void);

static int ok(uint32_t ea) { return ea >= 0x10000u && ea < 0xF0000000u; }
static uint32_t rd32(uint32_t ea)
{ const uint8_t* p = vm_base + ea; return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }
static void wr32(uint32_t ea, uint32_t v)
{ uint8_t* p = vm_base + ea; p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v; }
static float rdf(uint32_t ea) { uint32_t u = rd32(ea); float f; memcpy(&f, &u, 4); return f; }
static void wrf(uint32_t ea, float f) { uint32_t u; memcpy(&u, &f, 4); wr32(ea, u); }
static int envi(const char* n, int d) { const char* e = getenv(n); return (e && *e) ? atoi(e) : d; }
static unsigned long long now_ms(void)
{
#ifdef _WIN32
    return GetTickCount64();
#else
    return 0;
#endif
}

/* mouse accumulator, micro-radians (pad thread adds, game thread drains) */
static volatile long s_acc_yaw = 0, s_acc_pitch = 0;
static volatile unsigned long long s_cam_ms = 0;   /* last on-foot camera update (game thread) */
/* v1053D: ms since the on-foot camera last ran (cellPad.c car controls: it stops while driving) */
unsigned long long gf_footcam_idle_ms(void) { const unsigned long long c = s_cam_ms; return c ? now_ms() - c : ~0ull; }
static volatile unsigned long long s_look_ms = 0;  /* last mouse-look call that was accepted (pad thread) */

/* v1107C street-dialogue camera (gf_dialogue.c sets these on the main thread, the camera update reads them): a two-shot over
 * the player's shoulder onto the NPC. view() moves the eye (blend 0.45 s in / out), gf_footcam_yaw / gf_footcam_pitch turn
 * the camera so it looks from the eye at g_gfd_cam_tgt. The sign of the game's per-frame turn (f31) and of the pitch offset
 * (+0x228) relative to the view angles (+0x1C yaw, +0x18 pitch: dir = (sin y cos p, -sin p, cos y cos p)) is learned per
 * camera object: three growing errors in a row flip it. */
volatile int g_gfd_cam_on = 0; float g_gfd_cam_eye[3], g_gfd_cam_tgt[3];
static float s_dcb = 0.0f; static unsigned long long s_dcb_t = 0;
static struct { uint32_t obj; float sgn, perr; int ngrow; float psgn, pperr, poff; int pngrow; } s_dc[8]; static int s_ndc;
static int dc_slot(uint32_t obj)
{
    for (int i = 0; i < s_ndc; i++) if (s_dc[i].obj == obj) return i;
    const int k = s_ndc < 8 ? s_ndc++ : (int)(obj % 8u);
    s_dc[k].obj = obj; s_dc[k].sgn = 1.0f; s_dc[k].perr = 1e9f; s_dc[k].ngrow = 0;
    s_dc[k].psgn = 1.0f; s_dc[k].pperr = 1e9f; s_dc[k].pngrow = 0; s_dc[k].poff = rdf(obj + 0x228);
    if (!(s_dc[k].poff == s_dc[k].poff) || fabsf(s_dc[k].poff) > 1.5f) s_dc[k].poff = 0.0f;
    return k;
}
static float dc_wrap(float a) { while (a > 3.14159265f) a -= 6.2831853f; while (a < -3.14159265f) a += 6.2831853f; return a; }
static int dc_angles(const float* eye, float* yt, float* pt)
{
    const float dx = g_gfd_cam_tgt[0] - eye[0], dy = g_gfd_cam_tgt[1] - eye[1], dz = g_gfd_cam_tgt[2] - eye[2];
    const float h = sqrtf(dx * dx + dz * dz); if (!(h > 0.05f)) return 0;
    *yt = atan2f(dx, dz); *pt = atan2f(-dy, h); return 1;
}
static int dc_active(void) { return g_gfd_cam_on || s_dcb > 0.0f; }
static int s_on = -1, s_logon = 1, s_yawsign = -1; static float s_pmax_deg = 65.0f;

static void cfg(void)
{
    if (s_on >= 0) return;
    s_on = envi("GF_FOOTCAM", 1); s_logon = envi("GF_FOOTCAM_LOG", 1); s_yawsign = envi("GF_FOOTCAM_YAWSIGN", -1);
    { const char* e = getenv("GF_FOOTCAM_PITCH"); if (e && *e) s_pmax_deg = (float)atof(e); }
    if (s_pmax_deg < 5.0f) s_pmax_deg = 5.0f;
    if (s_pmax_deg > 85.0f) s_pmax_deg = 85.0f;
    if (s_on) g_pad_look_fn = 0;   /* installed on the first camera frame (gf_look below) */
    printf("[footcam] v1046L GTA-style mouse look on foot %s (yaw sign %d, pitch +-%.0f deg)\n", s_on ? "ON" : "OFF",
           s_yawsign, s_pmax_deg);
    fflush(stdout);
}

static int s_rawlook = -1;   /* v1052M: 1 = the camera reads the raw mouse counts itself, once per camera frame */
static int gf_look(float dyaw_deg, float dpitch_deg)
{
    const unsigned long long t = now_ms();
    if (t - s_cam_ms > 150ull) return 0;          /* the on-foot camera is not running (car, cutscene, free aim ...) */
    if (s_rawlook == 1) { s_look_ms = t; return 1; }   /* v1052M: only "mouse look is active"; the amounts come from raw input */
#ifdef _WIN32
    if (dyaw_deg != 0.0f)   InterlockedExchangeAdd(&s_acc_yaw,   (long)lrintf(dyaw_deg * 17453.2925f));
    if (dpitch_deg != 0.0f) InterlockedExchangeAdd(&s_acc_pitch, (long)lrintf(dpitch_deg * 17453.2925f));
#endif
    s_look_ms = t;
    return 1;
}

/* v1046N (user: "not smooth, choppy when looking around"): the pad thread samples the mouse every ~4 ms (Sleep jitter)
 * while frames come every 5.6 ms at 180 FPS, so a frame got 0, 1 or 2 samples' worth of turn. Mouse turn now goes into a
 * pending amount that is released over time (exponential, GF_FOOTCAM_SMOOTH_MS, default 12 ms; 0 = raw), once per
 * camera frame from the top of func_00055E40, by the real frame time. */
static double s_pend_yaw = 0.0, s_pend_pitch = 0.0, s_frame_yaw = 0.0, s_frame_pitch = 0.0;
static double qpc_sec(void)
{
#ifdef _WIN32
    static LARGE_INTEGER f; LARGE_INTEGER t;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
#else
    return 0.0;
#endif
}
static void smooth_step(void)
{
    static double tau = -1.0, last = 0.0;
    if (tau < 0.0) { const char* e = getenv("GF_FOOTCAM_SMOOTH_MS"); tau = (e && *e) ? atof(e) * 0.001 : 0.012; if (tau < 0.0) tau = 0.0;
                     if (envi("GF_FOOTCAM_RAW", 1)) { const char* r = getenv("GF_FOOTCAM_RAW_SMOOTH_MS"); tau = (r && *r) ? atof(r) * 0.001 : 0.0; if (tau < 0.0) tau = 0.0; } }   /* v1052M */
    long ay = 0, ap = 0;
#ifdef _WIN32
    ay = InterlockedExchange(&s_acc_yaw, 0); ap = InterlockedExchange(&s_acc_pitch, 0);
#endif
    s_pend_yaw += (double)ay * 1e-6; s_pend_pitch += (double)ap * 1e-6;
    /* v1052M GF_FOOTCAM_RAW (default 1; user: "just fix the camera when looking around"): the mouse turn used to come
     * from the pad thread, which samples every 4-5 ms -- at ~90 FPS a frame got 2 or 3 samples' worth of turn (+-30 %),
     * and the samples themselves arrived in bursts (raw input pumped by the FIFO thread, fixed in the backend). Now the
     * camera takes ALL raw mouse counts since its last frame, here, once per camera frame -- the exact movement, like a
     * PC game reads the mouse. Same scale as before (GF_FOOTCAM_DEG x F10 mouse speed / 450, GF_FOOTCAM_INVX/INVY).
     * Smoothing GF_FOOTCAM_RAW_SMOOTH_MS (default 0 = none). GF_FOOTCAM_RAW=0 = old pad-thread path. */
#ifdef _WIN32
    { extern volatile long g_pad_raw_look_dx, g_pad_raw_look_dy; extern volatile int g_pad_raw_ok; extern volatile int g_ui_mouse_speed;
      static float deg0 = 0.12f; static int ix = 0, iy = 0;
      if (s_rawlook < 0) {
          s_rawlook = envi("GF_FOOTCAM_RAW", 1) ? 2 : 0;   /* 2 = wanted, 1 = active once raw input is seen */
          { const char* e = getenv("GF_FOOTCAM_DEG"); if (e && *e) deg0 = (float)atof(e); }
          ix = envi("GF_FOOTCAM_INVX", 0); iy = envi("GF_FOOTCAM_INVY", 0);
          printf("[footcam] v1052M camera reads the raw mouse once per frame: %s (GF_FOOTCAM_RAW)\n", s_rawlook ? "ON" : "off"); fflush(stdout);
      }
      if (s_rawlook == 2 && g_pad_raw_ok) s_rawlook = 1;
      if (s_rawlook == 1) {
          const long rx = InterlockedExchange(&g_pad_raw_look_dx, 0), ry = InterlockedExchange(&g_pad_raw_look_dy, 0);
          if (now_ms() - s_look_ms < 150ull) {   /* mouse look active (pad thread: on foot, KBM, not aiming / wheel / grab) */
              float dpc = deg0; if (g_ui_mouse_speed > 0) dpc = deg0 * (float)g_ui_mouse_speed / 450.0f;
              s_pend_yaw   += (double)rx * dpc * (ix ? -1.0 : 1.0) * 0.017453292519943295;
              s_pend_pitch += (double)ry * dpc * (iy ? -1.0 : 1.0) * 0.017453292519943295;
          }
      } }
#endif
    const double now = qpc_sec();
    double dt = last > 0.0 ? now - last : 0.0056; last = now;
    if (dt > 0.1) dt = 0.1;
    /* v1050S GF_FOOTCAM_GAMEDT (default 1): release the mouse turn by the GAME timestep (g_gf_clock_dt, published once per
     * game frame by the clock hook, smoothed to the pacer's period by GF_DT_SMOOTH) instead of the wall time between
     * camera updates -> the view turns by the same step the world moves, every frame. Further camera updates in the same
     * game frame release nothing more (their share is already pending for the hooks). Falls back to wall time when the
     * clock hook is not running (60 FPS target, cutscene skip). */
    { extern volatile double g_gf_clock_dt; extern volatile unsigned g_gf_clock_seq;
      static unsigned seq0 = 0; static double seen = 0.0; static int use = -1;
      if (use < 0) { use = envi("GF_FOOTCAM_GAMEDT", 1); printf("[footcam] v1050S mouse turn released by the game timestep: %s (GF_FOOTCAM_GAMEDT)\n", use ? "ON" : "off"); fflush(stdout); }
      const unsigned sq = g_gf_clock_seq;
      if (use && sq != 0) {
          if (sq != seq0) {
              const unsigned nt = sq - seq0; double gdt = g_gf_clock_dt;
              if (seq0 != 0 && nt > 1 && nt < 8) gdt *= (double)nt;
              seq0 = sq; seen = now; dt = gdt;
          } else if (now - seen < 0.1) dt = 0.0;
      } }
    const double k = (tau <= 0.0) ? 1.0 : 1.0 - exp(-dt / tau);
    /* v1049P (user: "the camera moves in low FPS but the game itself is not low FPS"): this runs at the top of EVERY
     * on-foot camera update and used to OVERWRITE s_frame_yaw / s_frame_pitch. The game runs several camera updates
     * per frame and not all of them reach the yaw / pitch hooks that consume it, so an update that did not consume it
     * threw that frame's mouse turn away -> the view only turned on some frames. Now the released amount ADDS UP
     * until a hook consumes it (the consumers zero it), so no mouse movement is lost. */
    { const double ry = s_pend_yaw * k, rp = s_pend_pitch * k;
      s_frame_yaw += ry;   s_pend_yaw -= ry;
      s_frame_pitch += rp; s_pend_pitch -= rp; }
    if (fabs(s_pend_yaw) < 1e-7) s_pend_yaw = 0.0;
    if (fabs(s_pend_pitch) < 1e-7) s_pend_pitch = 0.0;
}
void gf_footcam_frame(void)
{
#ifdef _WIN32
    __try { smooth_step(); } __except (EXCEPTION_EXECUTE_HANDLER) { }
#else
    smooth_step();
#endif
}

static int s_engaged = 0; static uint32_t s_obj = 0; static float s_pitch = 0.0f, s_lim_e8 = 0.0f;
static unsigned long long s_last_log = 0; static double s_yaw_total = 0.0;

/* v1049C (user: camera choppy -- "the game keeps jumping back and forth between the old camera settings"): the game
 * updates SEVERAL on-foot camera objects (the 23:35 log: 10 objects, 740 engage/release cycles; two alternating every
 * frame). The hook released and re-engaged on every camera change, so the mouse turn went to whichever camera ran
 * next and the other kept the game's own yaw -> the view jumped between them. Now every on-foot camera seen while the
 * mouse is active is kept in step: s_yaw_total is the mouse turn since engaging, each camera remembers how much of it
 * it has received (applied[]) and gets the rest whenever it runs; the pitch offset is the same value for all.
 * GF_FOOTCAM_MULTI=0 = the old single-camera behaviour. */
#define FC_N 8
static uint32_t s_fc_obj[FC_N]; static double s_fc_applied[FC_N]; static float s_fc_e8[FC_N]; static int s_fc_n = 0;
/* v1053P: when the game last updated each stored camera. User freeze 23:47: the main thread looped in the heap allocator
 * (func_00A45D50) on a corrupted free list right after new on-foot cameras appeared -- the release below restored +0xE8
 * into EVERY camera ever joined, including ones the game had freed meanwhile. Now only cameras the game updated within
 * GF_FOOTCAM_ALIVE_MS (200) are written; a stale one is skipped (if it still exists its pitch limit just stays raised). */
static unsigned long long s_fc_seen[FC_N]; static unsigned long long s_obj_seen = 0; static int s_alive_ms = -1;
static int s_multi = -1;
static int fc_find(uint32_t obj)
{
    for (int i = 0; i < s_fc_n; i++) if (s_fc_obj[i] == obj) return i;
    return -1;
}
static void fc_release_all(unsigned long long t)
{
    for (int i = 0; i < s_fc_n; i++) if (ok(s_fc_obj[i]) && t - s_fc_seen[i] < (unsigned long long)s_alive_ms) wrf(s_fc_obj[i] + 0xE8, s_fc_e8[i]);
    s_fc_n = 0;
}
static int active_for(uint32_t obj)
{
    cfg();
    if (!s_on || !ok(obj)) return 0;
    const unsigned long long t = now_ms();
    s_cam_ms = t;
    if (s_alive_ms < 0) s_alive_ms = envi("GF_FOOTCAM_ALIVE_MS", 200);
    { const int k = fc_find(obj); if (k >= 0) s_fc_seen[k] = t; if (obj == s_obj) s_obj_seen = t; }   /* v1053P */
    if (!g_pad_look_fn) g_pad_look_fn = gf_look;
    if (s_multi < 0) { s_multi = envi("GF_FOOTCAM_MULTI", 1); printf("[footcam] v1049C all on-foot cameras follow the mouse together: %s (GF_FOOTCAM_MULTI)\n", s_multi ? "ON" : "off"); fflush(stdout); }
    const int act = (t - s_look_ms) < 150ull;
    if (s_engaged && (!act || (!s_multi && obj != s_obj))) {
        if (s_multi) fc_release_all(t); else if (ok(s_obj) && t - s_obj_seen < (unsigned long long)s_alive_ms) wrf(s_obj + 0xE8, s_lim_e8);
        s_engaged = 0;
        if (s_logon) { printf("[footcam] released (mouse idle%s)\n", s_multi ? "" : " or camera changed"); fflush(stdout); }
    }
    if (!act) return 0;
    if (s_engaged && s_multi && fc_find(obj) < 0) {   /* v1049C: another camera of the same view: join it in step */
        int k = s_fc_n < FC_N ? s_fc_n++ : FC_N - 1;
        s_fc_obj[k] = obj; s_fc_applied[k] = s_yaw_total; s_fc_e8[k] = rdf(obj + 0xE8); s_fc_seen[k] = t;
        static int nlog = 0; if (s_logon && nlog < 40) { nlog++; printf("[footcam] v1049C camera 0x%08X joins (%d in step)\n", obj, s_fc_n); fflush(stdout); }
    }
    if (!s_engaged) {
        s_engaged = 1; s_obj = obj;
        s_yaw_total = 0.0; s_fc_n = 1; s_fc_obj[0] = obj; s_fc_applied[0] = 0.0; s_fc_e8[0] = rdf(obj + 0xE8); s_fc_seen[0] = t; s_obj_seen = t;
        s_lim_e8 = rdf(obj + 0xE8);
        s_pitch = rdf(obj + 0x228);
        if (!(s_pitch == s_pitch) || fabsf(s_pitch) > 2.0f) s_pitch = 0.0f;
#ifdef _WIN32
        InterlockedExchange(&s_acc_yaw, 0); InterlockedExchange(&s_acc_pitch, 0);
#endif
        s_pend_yaw = s_pend_pitch = s_frame_yaw = s_frame_pitch = 0.0;
        if (s_logon) {
            printf("[footcam] engaged camera 0x%08X: pitch offset %.3f rad, game pitch limit %.1f deg, yaw %.3f, player yaw %.3f\n",
                   obj, s_pitch, s_lim_e8, rdf(obj + 0x1B0), rdf(obj + 0x244));
            fflush(stdout);
        }
    }
    return 1;
}

/* loc_00056CF4: f31 = this frame's rotation of the camera around the player (rad) */
double gf_footcam_yaw(uint64_t obj64, double f31)
{
#ifdef _WIN32
    __try {
#endif
        const uint32_t obj = (uint32_t)obj64;
        if (g_gfd_cam_on && ok(obj)) {   /* v1107C: turn toward the dialogue target */
            const int k = dc_slot(obj); float yt, pt; const float eye[3] = { g_gfd_cam_eye[0], g_gfd_cam_eye[1], g_gfd_cam_eye[2] };
            if (!dc_angles(eye, &yt, &pt)) return 0.0;
            const float err = dc_wrap(yt - rdf(obj + 0x1C)), ae = fabsf(err);
            if (ae > s_dc[k].perr + 0.002f && ae > 0.03f) { if (++s_dc[k].ngrow >= 3) { s_dc[k].sgn = -s_dc[k].sgn; s_dc[k].ngrow = 0;
                printf("[footcam] v1107C dialogue camera %08X: yaw sign flipped (err %.3f)\n", obj, err); fflush(stdout); } }
            else s_dc[k].ngrow = 0;
            s_dc[k].perr = ae;
            float st = err * 0.22f; if (st > 0.12f) st = 0.12f; if (st < -0.12f) st = -0.12f;
            return (double)(st * s_dc[k].sgn);
        }
        if (!active_for(obj)) {
            { extern volatile int g_gf_sitting; if (g_gf_sitting) return 0.0; }   /* v1109G seated (gf_sit.c pins the player every frame): no swing back behind him -- the camera spun */
            /* GF_CAMPROBE: sum of the game's own turn per second (sign check: stick right -> ?) */
            static int pr = -1; static double sum = 0.0; static unsigned long long t0 = 0;
            if (pr < 0) pr = envi("GF_CAMPROBE", 0);
            if (pr) { const unsigned long long t = now_ms(); sum += f31; if (!t0) t0 = t;
                      if (t - t0 >= 1000ull) { printf("[footcam] game turn last second %+.4f rad\n", sum); fflush(stdout); sum = 0.0; t0 = t; } }
            return f31;
        }
        double d = s_frame_yaw * (double)s_yawsign; s_frame_yaw = 0.0;
        if (d > 3.0) d = 3.0;
        if (d < -3.0) d = -3.0;
        s_yaw_total += d;
        if (s_multi) {   /* v1049C: this camera gets all the turn it has not received yet */
            const int k = fc_find(obj);
            if (k >= 0) { d = s_yaw_total - s_fc_applied[k]; s_fc_applied[k] = s_yaw_total;
                          if (d > 3.0) d = 3.0; if (d < -3.0) d = -3.0; }
        }
        const unsigned long long t = now_ms();
        if (s_logon && t - s_last_log >= 2000ull) {
            s_last_log = t;
            printf("[footcam] mouse yaw %+.1f deg (total), pitch offset %.3f rad, camera yaw %.3f, player yaw %.3f, game wanted %+.4f\n",
                   s_yaw_total * 57.29578, s_pitch, rdf(obj + 0x1B0), rdf(obj + 0x244), f31);
            fflush(stdout);
        }
        return d;
#ifdef _WIN32
    } __except (EXCEPTION_EXECUTE_HANDLER) { return f31; }
#endif
}

/* loc_000565D0 (before the pitch clamp): obj+0x228 pitch offset (rad), obj+0x224 its velocity */
void gf_footcam_pitch(uint64_t obj64)
{
#ifdef _WIN32
    __try {
#endif
        const uint32_t obj = (uint32_t)obj64;
        if (g_gfd_cam_on && ok(obj)) {   /* v1107C: pitch toward the dialogue target */
            const int k = dc_slot(obj); float yt, pt; const float eye[3] = { g_gfd_cam_eye[0], g_gfd_cam_eye[1], g_gfd_cam_eye[2] };
            if (dc_angles(eye, &yt, &pt)) {
                const float err = pt - rdf(obj + 0x18), ae = fabsf(err);
                if (ae > s_dc[k].pperr + 0.002f && ae > 0.03f) { if (++s_dc[k].pngrow >= 3) { s_dc[k].psgn = -s_dc[k].psgn; s_dc[k].pngrow = 0; } } else s_dc[k].pngrow = 0;
                s_dc[k].pperr = ae;
                s_dc[k].poff += s_dc[k].psgn * err * 0.2f; if (s_dc[k].poff > 1.0f) s_dc[k].poff = 1.0f; if (s_dc[k].poff < -1.0f) s_dc[k].poff = -1.0f;
                wrf(obj + 0xE8, 65.0f); wrf(obj + 0x228, s_dc[k].poff); wrf(obj + 0x224, 0.0f); }
            return;
        }
        if (!s_engaged || (now_ms() - s_look_ms) >= 150ull) return;
        if (s_multi ? fc_find(obj) < 0 : obj != s_obj) return;   /* v1049C: any camera kept in step */
        const float pmax = s_pmax_deg * 0.017453293f;
        s_pitch += (float)s_frame_pitch; s_frame_pitch = 0.0;
        if (s_pitch > pmax) s_pitch = pmax;
        if (s_pitch < -pmax) s_pitch = -pmax;
        wrf(obj + 0xE8, s_pmax_deg);      /* the game's clamp right after this uses obj+0xE8 (degrees) */
        wrf(obj + 0x228, s_pitch);
        wrf(obj + 0x224, 0.0f);
#ifdef _WIN32
    } __except (EXCEPTION_EXECUTE_HANDLER) { }
#endif
}

/* v1046L GF_CAMPROBE=1 (diagnostic): which camera update runs (top hooks of func_00055E40 = 1, func_0006A7F8 = 2,
 * func_00060AE0 = 3) and, once a second, every float of the most active camera object (0x10..0x2FC) that moved by more
 * than 0.05 since the previous report. Drive the right stick (PAD_STICK) to see the yaw / pitch fields. */
static void probe(uint32_t obj, int which)
{
    static int on = -1; static unsigned calls[4]; static uint32_t objs[4];
    static unsigned long long t0 = 0; static float prev[0xC0]; static uint32_t pobj = 0;
    if (on < 0) { on = envi("GF_CAMPROBE", 0); if (on) { printf("[camprobe] ON\n"); fflush(stdout); } }
    if (!on || which < 1 || which > 3 || !ok(obj)) return;
    calls[which]++; objs[which] = obj;
    const unsigned long long t = now_ms();
    if (!t0) t0 = t;
    if (t - t0 < 1000ull) return;
    t0 = t;
    int best = 1; for (int i = 2; i <= 3; i++) if (calls[i] > calls[best]) best = i;
    printf("[camprobe] calls/s 55E40:%u 6A7F8:%u 60AE0:%u | obj %08X %08X %08X\n", calls[1], calls[2], calls[3], objs[1], objs[2], objs[3]);
    if (calls[best]) {
        const uint32_t o = objs[best];
        char line[4096]; int n = 0;
        n += snprintf(line + n, sizeof line - n, "[camprobe] fn%d obj %08X moved:", best, o);
        for (int i = 4; i < 0xC0 && n < 3900; i++) {
            const float v = rdf(o + (uint32_t)i * 4u);
            if (o == pobj && v == v && prev[i] == prev[i] && fabsf(v - prev[i]) > 0.05f && fabsf(v) < 1e6f)
                n += snprintf(line + n, sizeof line - n, " +%03X %.3f->%.3f", i * 4, prev[i], v);
            prev[i] = v;
        }
        pobj = o;
        printf("%s\n", line); fflush(stdout);
    }
    calls[1] = calls[2] = calls[3] = 0;
}
void gf_cam_probe(uint64_t obj, uint64_t which)
{
    if (which == 3) { extern volatile unsigned long long g_pad_fa_cam_ms; g_pad_fa_cam_ms = now_ms(); }   /* v1049A: free-aim camera is running */
#ifdef _WIN32
    __try { probe((uint32_t)obj, (int)which); }
    __except (EXCEPTION_EXECUTE_HANDLER) { }
#else
    probe((uint32_t)obj, (int)which);
#endif
}


/* ---------------------------------------------------------------------------------------------------------------
 * v1046V camera views (user: "add different camera angles by pressing V, make first person one of them").
 * The on-foot camera (func_00055E40) writes its result -- eye obj+0x8/+0xC/+0x10 (x, y, z; copy of the internal
 * +0x1DC..) and angles obj+0x18 pitch (rad, + = down), +0x1C yaw (rad, view = atan2(dx, dz), checked against the
 * eye -> player vector in soak Y10), +0x20, +0x24 -- and returns. At its single return (hook gf_footcam_view) the
 * selected view replaces only the OUTPUT eye; the game's internal state (+0x1B8.. / +0x1DC..) is untouched, so its own
 * follow logic, collision and the mouse look keep running and "game" view is exactly the original.
 * Views (V tap cycles, cellPad.c; V held = L3):
 *   0 game     -- unchanged
 *   1 close    -- over the shoulder: GF_CAM_CLOSE_DIST 1.9, GF_CAM_SHOULDER 0.45 (negative = other shoulder)
 *   2 far      -- GF_CAM_FAR_DIST 5.5
 *   3 first person -- eye at the head (GF_CAM_HEAD 1.62 above the feet), GF_CAM_FP_FWD 0.22 in front of the face
 * Pivot = player feet (*(player+0x68)+0x40, player = *(obj+0xA0) - 0x38) + GF_CAM_HEAD up (close/far use
 * GF_CAM_PIVOT 1.55). Switching blends over ~0.2 s. GF_CAM_VIEWS=0 = V stays plain L3. Log [camview].
 * Known limits: no wall collision for close/far; in first person the body is still drawn (looking down shows it).
 */
static volatile int s_view = 0, s_view_req = 0;
static uint32_t s_view_obj = 0;
uint32_t gf_footcam_cam_obj(void) { return s_view_obj; }   /* v1111U tailor probe */
static float s_view_blend = 1.0f, s_from[3], s_last_eye[3]; static int s_have_last = 0;
static const char* k_view_name[4] = { "game", "close (over the shoulder)", "far", "first person" };

static float envf(const char* n, float d) { const char* e = getenv(n); return (e && *e) ? (float)atof(e) : d; }

static void camcycle(void) { s_view_req++; }


void gf_footcam_setobj(uint64_t obj)
{
    static int inst = -1;
    if (inst < 0) { inst = envi("GF_CAM_VIEWS", 0);   /* v1046X: OFF by default -- the game froze ~5 frames after switching to the close view (user log 18:5x) */ if (inst) g_pad_camcycle_fn = camcycle;
                    { int v0 = envi("GF_CAM_VIEW_START", 0); if (inst && v0 > 0 && v0 < 4) s_view = v0; }   /* test aid */
                    printf("[camview] v1046V camera views on V: %s\n", inst ? "ON" : "OFF"); fflush(stdout); }
    s_view_obj = (uint32_t)obj;
}

static void view(void)
{
    static int cfg_done = 0; static float cdist, fdist, shoulder, head, pivot, fpfwd;
    if (!cfg_done) { cfg_done = 1;
        cdist = envf("GF_CAM_CLOSE_DIST", 1.9f); fdist = envf("GF_CAM_FAR_DIST", 5.5f);
        shoulder = envf("GF_CAM_SHOULDER", 0.45f); head = envf("GF_CAM_HEAD", 1.62f);
        pivot = envf("GF_CAM_PIVOT", 1.55f); fpfwd = envf("GF_CAM_FP_FWD", 0.22f); }
    const uint32_t obj = s_view_obj;
    if (!ok(obj)) return;
    while (s_view_req > 0) {
        s_view_req--;
        { static int fp = -1; if (fp < 0) fp = envi("GF_CAM_FP", 0);   /* v1046W: first person off by default (user: "horrible") */
          s_view = (s_view + 1) % (fp ? 4 : 3); }
        if (s_have_last) { memcpy(s_from, s_last_eye, sizeof s_from); s_view_blend = 0.0f; }
        printf("[camview] view %d: %s\n", s_view, k_view_name[s_view]); fflush(stdout);
    }
    const uint32_t pa = rd32(obj + 0xA0);
    if (!ok(pa) || pa < 0x38u) return;
    const uint32_t m = rd32(pa - 0x38u + 0x68);
    if (!ok(m)) return;
    const float px = rdf(m + 0x40), py = rdf(m + 0x44), pz = rdf(m + 0x48);
    float eye[3] = { rdf(obj + 0x8), rdf(obj + 0xC), rdf(obj + 0x10) };
    if (!(px == px) || !(eye[0] == eye[0])) return;
    if (s_view != 0) {
        const float yaw = rdf(obj + 0x1C), pit = rdf(obj + 0x18);
        const float cy = cosf(yaw), sy = sinf(yaw), cp = cosf(pit), sp = sinf(pit);
        const float dx = sy * cp, dy = -sp, dz = cy * cp;      /* view direction */
        const float rx = -cy, rz = sy;                           /* horizontal perpendicular (shoulder side) */
        if (s_view == 3) {
            eye[0] = px + sy * fpfwd; eye[1] = py + head; eye[2] = pz + cy * fpfwd;
        } else {
            const float dist = s_view == 1 ? cdist : fdist, sh = s_view == 1 ? shoulder : 0.0f;
            eye[0] = px - dx * dist + rx * sh; eye[1] = py + pivot - dy * dist; eye[2] = pz - dz * dist + rz * sh;
        }
    }
    if (s_view_blend < 1.0f) {
        s_view_blend += 0.03f;                       /* ~0.2 s at 150-180 FPS */
        if (s_view_blend > 1.0f) s_view_blend = 1.0f;
        const float k = s_view_blend * s_view_blend * (3.0f - 2.0f * s_view_blend);
        /* blend in player-relative space so a running player is not left behind */
        for (int i = 0; i < 3; i++) eye[i] = s_from[i] + (eye[i] - s_from[i]) * k;
    }
    { /* v1107C dialogue camera */
      const unsigned long long tn = now_ms(); float dt = s_dcb_t ? (float)(tn - s_dcb_t) / 1000.0f : 0.0f; s_dcb_t = tn; if (dt > 0.1f) dt = 0.1f;
      s_dcb += (g_gfd_cam_on ? dt : -dt) / 0.45f; if (s_dcb < 0.0f) s_dcb = 0.0f; if (s_dcb > 1.0f) s_dcb = 1.0f;
      if (s_dcb > 0.0f && g_gfd_cam_eye[0] == g_gfd_cam_eye[0]) { const float k = s_dcb * s_dcb * (3.0f - 2.0f * s_dcb);
          for (int i = 0; i < 3; i++) eye[i] = eye[i] + (g_gfd_cam_eye[i] - eye[i]) * k;
          wrf(obj + 0x8, eye[0]); wrf(obj + 0xC, eye[1]); wrf(obj + 0x10, eye[2]); } }
    if (s_view != 0 || s_view_blend < 1.0f) { wrf(obj + 0x8, eye[0]); wrf(obj + 0xC, eye[1]); wrf(obj + 0x10, eye[2]); }
    memcpy(s_last_eye, eye, sizeof eye); s_have_last = 1;
}

/* v1110J jump watch (user: "sometimes when I run, it's like it 'resets' the position of the player and camera"). Not
 * reproducible by the bot, nothing found by reading -> measure it. At the end of every on-foot camera update (per camera
 * object): player feet *(player+0x68)+0x40, output eye obj+0x8.., yaw obj+0x1C. Logged ("[camjump]", first 80 + 10 s summary):
 *   PLAYER JUMP  -- the player moved > 0.6 m in one update at > 14 m/s, or > 0.25 m AGAINST his recent motion (snap back);
 *   CAMERA YAW   -- yaw changed > 0.30 rad in one update beyond the mouse turn applied in it;
 *   CAMERA EYE   -- the eye moved > 0.8 m more than the player did (relative jump);
 *   HITCH        -- > 120 ms between updates of the same camera.
 * with the context (mouse camera engaged, camera objects in step, dialogue camera, sitting). GF_CAMJUMP=0 off. */
typedef struct { uint32_t obj; double t; float p[3], e[3], yaw, vx, vz; int ok; } jw_t;
static jw_t s_jw[FC_N]; static int s_jw_on = -1, s_jw_logged; static long st_jw[4], st_jw_n;
static double s_jw_mouse_prev;
extern int gf_player_setpos(uint32_t pl, float x, float y, float z);   /* gf_shop.c v1112I */
static long st_unlurch;
static void jw_sample(void)
{
    if (s_jw_on < 0) { s_jw_on = envi("GF_CAMJUMP", 1); printf("[camjump] v1110J player / camera jump watch %s (GF_CAMJUMP)\n", s_jw_on ? "ON" : "off"); fflush(stdout); }
    if (!s_jw_on) return;
    const uint32_t obj = s_view_obj; if (!ok(obj)) return;
    const uint32_t pa = rd32(obj + 0xA0); if (!ok(pa) || pa < 0x38u) return;
    const uint32_t m = rd32(pa - 0x38u + 0x68); if (!ok(m)) return;
    const float p[3] = { rdf(m + 0x40), rdf(m + 0x44), rdf(m + 0x48) }, e[3] = { rdf(obj + 0x8), rdf(obj + 0xC), rdf(obj + 0x10) };
    const float yaw = rdf(obj + 0x1C);
    if (!(p[0] == p[0]) || !(e[0] == e[0]) || !(yaw == yaw) || fabsf(p[0]) > 1e5f || fabsf(e[0]) > 1e5f) return;
    const double t = qpc_sec();
    float pc[3] = { p[0], p[1], p[2] };   /* v1112I: position after an unlurch */
    int k = -1; for (int i = 0; i < FC_N; i++) if (s_jw[i].ok && s_jw[i].obj == obj) { k = i; break; }
    if (k < 0) { k = 0; for (int i = 0; i < FC_N; i++) if (!s_jw[i].ok || t - s_jw[i].t > 2.0) { k = i; break; } s_jw[k].ok = 0; }
    jw_t* j = &s_jw[k];
    const double mouse = s_yaw_total; const double dm = mouse - s_jw_mouse_prev; s_jw_mouse_prev = mouse;
    st_jw_n++;
    if (j->ok && j->obj == obj) {
        const double dt = t - j->t;
        const float dx = p[0] - j->p[0], dz = p[2] - j->p[2], dy = p[1] - j->p[1], dp = sqrtf(dx * dx + dz * dz);
        const float ex = e[0] - j->e[0], ey = e[1] - j->e[1], ez = e[2] - j->e[2];
        float dyaw = yaw - j->yaw; while (dyaw > 3.14159265f) dyaw -= 6.2831853f; while (dyaw < -3.14159265f) dyaw += 6.2831853f;
        /* the eye orbits the player when the camera turns: subtract that chord (2 r sin(|dyaw|/2), r = horizontal eye distance) */
        const float orx = j->e[0] - j->p[0], orz = j->e[2] - j->p[2], orb = sqrtf(orx * orx + orz * orz);
        float rel = sqrtf((ex - dx) * (ex - dx) + (ey - dy) * (ey - dy) + (ez - dz) * (ez - dz)) - 2.0f * orb * sinf(fabsf(dyaw) * 0.5f) - 0.15f * orb * fabsf(dyaw);
        if (rel < 0.0f) rel = 0.0f;
        const float vlen = sqrtf(j->vx * j->vx + j->vz * j->vz);
        const float along = vlen > 1.0f ? (dx * j->vx + dz * j->vz) / vlen : 0.0f;
        int ev = -1; const char* what = "";
        if (dt > 0.12 && dt < 5.0) { if (vlen > 2.0f || dp > 0.5f) { ev = 3; what = "HITCH (moving)"; } else st_jw[3]++; }
        else if (dt > 0.0 && ((dp > 0.6f && dp / dt > 14.0) || (vlen > 2.0f && along < -0.25f))) { ev = 0; what = along < -0.25f ? "PLAYER SNAP BACK" : "PLAYER JUMP"; }
        else if (fabsf(fabsf(dyaw) - fabsf((float)dm)) > 0.30f && fabsf(dyaw) > 0.30f) { ev = 1; what = "CAMERA YAW"; }
        else if (rel > 0.8f && fabsf(dyaw) < 0.08f) { ev = 2; what = "CAMERA EYE"; }   /* turning near walls = camera collision, not a jump */
        /* v1112I UNLURCH (user: "the small chop when running around, extremely frequent"): the player's position jumps a constant
         * ~0.74 m forward along his motion in one update every 3-9 m while running (v1110J). Take the excess over his normal step
         * back out before the frame is drawn: SetPosition (entity + physics body) back by the excess, the eye by the same vector.
         * Only for a clean forward lurch (along > 0.5 m, < 1.3 m, running > 2 m/s, dt < 60 ms). GF_UNLURCH=0 off. */
        { static int ul = -1; if (ul < 0) { ul = envi("GF_UNLURCH", 1); printf("[camjump] v1112I unlurch %s (GF_UNLURCH)\n", ul ? "ON" : "off"); fflush(stdout); }
          /* on foot only: in a car the "lurch" is just the car's speed (16-27 m/s in user logs) and SetPosition on the
           * driver stopped the car dead at high speed. Skip in a vehicle (g_pad_incar_fn) and above running speed. */
          extern int (*volatile g_pad_incar_fn)(void);
          if (ul && ev == 0 && along > 0.5f && dp < 1.3f && vlen > 2.0f && vlen < 12.0f && dt < 0.06 && !(g_pad_incar_fn && g_pad_incar_fn())) {
              float ex = dp - vlen * (float)dt; if (ex > dp) ex = dp;
              if (ex > 0.35f) {
                  const float ux = dx / dp, uz = dz / dp, nx = p[0] - ux * ex, nz = p[2] - uz * ex;
                  if (gf_player_setpos(pa - 0x38u, nx, p[1], nz)) {
                      wrf(obj + 0x8, e[0] - ux * ex); wrf(obj + 0x10, e[2] - uz * ex);
                      pc[0] = nx; pc[2] = nz; st_unlurch++;
                      if (st_unlurch <= 40 || st_unlurch % 100 == 0) { printf("[camjump] v1112I unlurch #%ld: took %.2f m back out (moved %.2f m in %.0f ms at %.1f m/s)\n", st_unlurch, ex, dp, dt * 1000.0, vlen); fflush(stdout); }
                      ev = -1; } } } }
        if (ev >= 0) { st_jw[ev]++;
            if (s_jw_logged < 80) { s_jw_logged++; extern volatile int g_gf_sitting;
                printf("[camjump] %s: dt %.0f ms | player moved %.2f m (%.1f m/s, along his motion %+.2f, speed before %.1f m/s, dy %+.2f) "
                       "| camera yaw %+.3f rad (mouse %+.3f), eye moved %.2f m relative | cam %08X, mouse cam %s (%d in step), dialogue %d, sitting %d | player (%.1f %.1f %.1f)\n",
                       what, dt * 1000.0, dp, dt > 0 ? dp / dt : 0.0, along, vlen, dy, dyaw, dm, rel, obj, s_engaged ? "engaged" : "released", s_fc_n,
                       g_gfd_cam_on, g_gf_sitting, p[0], p[1], p[2]); fflush(stdout); } }
        if (dt > 0.0 && dt < 0.12 && pc[0] == p[0] && pc[2] == p[2]) { const float a = 0.3f;   /* v1112I: not on an unlurch frame */ j->vx = j->vx * (1 - a) + (float)(dx / dt) * a; j->vz = j->vz * (1 - a) + (float)(dz / dt) * a; }
        else { j->vx = j->vz = 0.0f; }
    } else { j->vx = j->vz = 0.0f; }
    j->obj = obj; j->t = t; memcpy(j->p, pc, sizeof pc); memcpy(j->e, e, sizeof e); j->yaw = yaw; j->ok = 1;
    { static double lt; if (t - lt > 10.0) { lt = t; if (st_jw[0] + st_jw[1] + st_jw[2] + st_jw[3])
        printf("[camjump] 10s: %ld updates | player jumps / snap backs %ld, camera yaw jumps %ld, camera eye jumps %ld, hitches %ld\n", st_jw_n, st_jw[0], st_jw[1], st_jw[2], st_jw[3]); fflush(stdout); } }
}

void gf_footcam_view(void)
{
#ifdef _WIN32
    __try { view(); jw_sample(); } __except (EXCEPTION_EXECUTE_HANDLER) { }
#else
    view();
#endif
}

/* v1112J run-lurch probe (apply-pending.ps1 block v1112J): entry of the entity position setters. which 0 = func_00233248 vt+0x30
 * SetMatrix (arg = 4 rows x 16 bytes, position row at +0x30), 1 = func_002331F8 vt+0x24 SetPosition (arg = {x,y,z,flag}),
 * 2 = func_0046E9F4 frame setter (layout unknown: +0x30 and +0x40 both tried). Player only; a step > 0.3 m (< 3 m) is logged with
 * the caller (lr) -- the writer of the 0.74 m lurch. 10 s call counts. GF_POSW=0 off. Log "[posw]". */
void gf_posw_note(int which, uint32_t ent, uint32_t arg, uint32_t lr)
{
    static int on = -1; if (on < 0) { on = envi("GF_POSW", 1); printf("[posw] v1112J player position-writer probe %s (GF_POSW)\n", on ? "ON" : "off"); fflush(stdout); }
    if (!on) return;
    const uint32_t obj = s_view_obj; if (!ok(obj)) return;
    const uint32_t pa = rd32(obj + 0xA0); if (!ok(pa) || pa < 0x38u || ent != pa - 0x38u) return;
    static long cnt[3], big[3]; static unsigned long long t10; static int nlog;
    if (which < 0 || which > 2) return; cnt[which]++;
#ifdef _WIN32
    __try {
#endif
        const uint32_t xf = rd32(ent + 0x68); if (!ok(xf) || !ok(arg)) return;
        const float cx = rdf(xf + 0x40), cy = rdf(xf + 0x44), cz = rdf(xf + 0x48);
        float best = 0.0f, nx = 0, ny = 0, nz = 0; uint32_t offs[2] = { which == 1 ? 0u : 0x30u, 0x40u }; const int no = which == 2 ? 2 : 1;
        for (int k = 0; k < no; k++) {
            const float x = rdf(arg + offs[k]), y = rdf(arg + offs[k] + 4), z = rdf(arg + offs[k] + 8);
            if (!(x == x) || !(z == z) || fabsf(x) > 1e5f || fabsf(z) > 1e5f) continue;
            const float d = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
            if (d < 3.0f && d > best) { best = d; nx = x; ny = y; nz = z; }
        }
        if (best > 0.3f) { big[which]++;
            if (nlog < 80) { nlog++; static const char* nm[3] = { "SetMatrix func_00233248", "SetPosition func_002331F8", "frame func_0046E9F4" };
                printf("[posw] %s moves the player %.2f m in one call: (%.2f %.2f %.2f) -> (%.2f %.2f %.2f) | caller lr %08X\n", nm[which], (double)best,
                       (double)cx, (double)cy, (double)cz, (double)nx, (double)ny, (double)nz, lr); fflush(stdout); } }
#ifdef _WIN32
    } __except (EXCEPTION_EXECUTE_HANDLER) { }
#endif
    const unsigned long long n = now_ms(); if (!t10) t10 = n;
    if (n - t10 > 10000) { t10 = n; printf("[posw] 10s: SetMatrix %ld (%ld big), SetPosition %ld (%ld big), frame %ld (%ld big)\n", cnt[0], big[0], cnt[1], big[1], cnt[2], big[2]); fflush(stdout); }
}
