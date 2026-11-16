/* v1129RR: title code (godfather-recomp gf_*.c) reads keys and window focus through cellPad's record / replay
 * layer, so PAD_RECORD captures them and PAD_REPLAY feeds them back (e.g. gf_smoke.c sprint = Shift read directly).
 * Without PAD_RECORD / PAD_REPLAY these are the plain Win32 calls. */
#ifndef PAD_RR_H
#define PAD_RR_H
#ifdef _WIN32
#include <windows.h>
#ifdef __cplusplus
extern "C" {
#endif
SHORT pad_rr_GetAsyncKeyState(int vk);
SHORT pad_rr_GetKeyState(int vk);
HWND  pad_rr_GetForegroundWindow(void);
#ifdef __cplusplus
}
#endif
#define GetAsyncKeyState pad_rr_GetAsyncKeyState
#define GetKeyState pad_rr_GetKeyState
#define GetForegroundWindow pad_rr_GetForegroundWindow
#endif
#endif
