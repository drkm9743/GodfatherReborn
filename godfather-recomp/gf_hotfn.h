#pragma once

#include "recompiled/ppu_recomp.h"

#ifdef __cplusplus
extern "C" {
#endif

extern int gf_hotfn_on;

int gf_hot_00696704(ppu_context* ctx);
int gf_hot_00691358(ppu_context* ctx);
int gf_hot_00696784(ppu_context* ctx);
int gf_hot_006A9400(ppu_context* ctx);
int gf_hot_0070AF14(ppu_context* ctx);
int gf_hot_0054665C(ppu_context* ctx);   /* v1129N GF_HOTFN2 bit 0 (hook in ppu_recomp_002.cpp) */

#ifdef __cplusplus
}
#endif
