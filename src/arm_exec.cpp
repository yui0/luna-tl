/*
 * Copyright © 2026 Yuichiro Nakada / Project Lunaria
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/* arm_exec.cpp — ARM 32-bit JNI execution engine for Lunaria
 * Loads an ARM ELF shared object, resolves dynamic imports to SVC trampolines,
 * and executes it under dynarmic's A32 JIT.  JNI calls from the ARM guest are
 * intercepted via SVC and forwarded to the host-side JVM stubs.
 * SVC numbering:
 * 0–228   JNINativeInterface vtable indices (JNI spec, 0-based)
 * 229–236 JVM / dl* / logging
 * 237+    libc, EGL/GLES, pthread, zlib, math, extended libc, detours */


#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <ctype.h>
#include <wchar.h>
#include <wctype.h>
#include <dirent.h>
#include <elf.h>
#include <fcntl.h>
#include <limits.h>
#include <fenv.h>
#include <map>
#include <set>
#include <memory>
#include <deque>
#include <optional>
#include <sched.h>
#include <string>
#include <sys/mman.h>
#include <unordered_set>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <poll.h>
#include <vector>
#include <zlib.h>

#include <dlfcn.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>

#include "dynarmic/interface/A32/a32.h"
#include "dynarmic/interface/A32/config.h"
#include "dynarmic/interface/A32/coprocessor.h"
#include "dynarmic/interface/exclusive_monitor.h"

extern "C" {
#include "arm_exec.h"
#include "jvm/jvm.h"
#include "jvm/jni.h"
}

/* -------------------------------------------------------------------------
 * Virtual address layout
 * ---------------------------------------------------------------------- */
static constexpr uint32_t HEAP_BASE     = 0x50000000u;
static constexpr uint32_t HEAP_SIZE     = 0xA0000000u; /* 2.5 GB — Unity Dynamic Heap for heavy titles;
 * mmap is relocated below thread stacks (64 MB). */
static constexpr uint32_t THREAD_STACK_BASE = 0x48000000u; /* co-op thread stacks (1MB each) */
static constexpr uint32_t THREAD_STACK_SIZE = 0x00100000u;
/* Guest mmap arena.  Unity's TLSF/DynamicHeap reserves *hundreds* of MB of
 * address space up front (many PROT_NONE reservations), and our munmap is a
 * no-op so nothing is ever reclaimed.  The old 64 MB window (0x44M-0x48M)
 * exhausted during the very first IL2CPP init: mmap then returned ~0u, and the
 * guest allocator — which does not check for failure — computed pool = -1 +
 * header, wrapping to low addresses inside libil2cpp's .text (≈0x913xxx).  That
 * single wrap both corrupted libil2cpp literals (→ FastMutex spin) and, once the
 * region was truly out, raised std::bad_alloc → terminate/abort loop.
 * Relocate the arena into the large unused gap between BRK_END (0x10000000) and
 * TRAMP_BASE (0x41000000): ~784 MB, plenty for a full init. */
static constexpr uint32_t MMAP_BASE      = 0x10000000u; /* just above the brk arena */
static constexpr uint32_t MMAP_END       = 0x41000000u; /* up to the trampoline base (~784 MB) */
static constexpr uint32_t MMAP_MAX_SINGLE= 0x10000000u; /* 256 MB cap per single mmap request */
static constexpr uint32_t MMAP_ALIGN     = 0x10000u;    /* 64 KB alignment (covers HBLKSIZE ≤ 64 KB) */
static uint32_t g_mmap_next = MMAP_BASE;

/* Mono rejects corlib unless the path sits under .../Managed/mono/2.0/. */
static void mono_rewrite_corlib_path(std::string &path) {
    const char *needle = "/Managed/mscorlib.dll";
    size_t pos = path.find(needle);
    if (pos != std::string::npos && path.find("/mono/") == std::string::npos)
        path.replace(pos, strlen(needle), "/Managed/mono/2.0/mscorlib.dll");
}

/* Bump-allocate a guest mmap region of `raw_len` bytes (before page-rounding).
 * Returns guest VA on success, or ~0u (MAP_FAILED) on failure.
 * - Caps requests > MMAP_MAX_SINGLE to MAP_FAILED so Boehm GC retries smaller.
 * - Returns 64 KB-aligned addresses so any HBLKSIZE ≤ 64 KB satisfies the GC. */
static uint32_t mmap_bump(uint32_t raw_len) {
    uint32_t len = (raw_len + 4095u) & ~4095u;
    if (len == 0 || len > MMAP_MAX_SINGLE) return ~0u;
    if ((uint64_t)g_mmap_next + len > MMAP_END) return ~0u;
    uint32_t addr = g_mmap_next;
    g_mmap_next = (uint32_t)(((uint64_t)addr + len + MMAP_ALIGN - 1u) & ~(uint64_t)(MMAP_ALIGN - 1u));
    return addr;
}
/* sbrk / brk arena for Boehm GC's GC_scratch_alloc().
 * Must sit in the gap between libunity.so (ends ~0x00BA0840) and
 * libmono.so (starts 0x20000000).  256 MB should be ample. */
static constexpr uint32_t BRK_BASE = 0x04000000u;
static constexpr uint32_t BRK_END  = 0x10000000u; /* 192 MB */
static uint32_t g_brk = BRK_BASE;

static constexpr uint32_t TRAMP_BASE    = 0x41000000u;
static constexpr uint32_t JNI_TBL_BASE  = 0x41010000u; /* JNINativeInterface[229] */
static constexpr uint32_t JVM_TBL_BASE  = 0x41011000u; /* JNIInvokeInterface[8] */
static constexpr uint32_t ENV_SLOT_BASE = 0x41012000u; /* holds JNI_TBL_BASE */
static constexpr uint32_t VM_SLOT_BASE  = 0x41012008u; /* holds JVM_TBL_BASE */
static constexpr uint32_t STR_SCRATCH   = 0x41013000u; /* 4KB scratch for strings */
/* Page of constants for libc *data* symbols that are normally exported as
 * variables by bionic (not functions, so they cannot be SVC trampolines).
 * Most important: __page_size.  mono_pagesize() reads *__page_size; if the
 * symbol is left pointing at an SVC trampoline, *trampoline == the `svc #0`
 * instruction word (0xEF000000) and mono mis-sizes every mmap to ~3.7 GB. */
static constexpr uint32_t LIBC_DATA     = 0x41014000u; /* 4KB libc data globals */
static constexpr uint32_t LIBC_PAGE_SIZE  = LIBC_DATA + 0x00u; /* = 4096 */
static constexpr uint32_t LIBC_PAGE_SHIFT = LIBC_DATA + 0x04u; /* = 12 */
static constexpr uint32_t LIBC_PAGE_MASK  = LIBC_DATA + 0x08u; /* = 0xfff */
static constexpr uint32_t MJIV_CACHE      = LIBC_DATA + 0x20u; /* cached mono domain */
static constexpr uint32_t MJIV_ENTRY_MARK = LIBC_DATA + 0x24u; /* diag: stub entry count */
static constexpr uint32_t MJIV_RUNTIME_VER = LIBC_DATA + 0x28u; /* jit init version string */
static constexpr uint32_t MONO_EMPTY_STR     = LIBC_DATA + 0x2cu; /* "" for g_build_filename guard */

/* tiny ARM stub that does nothing and returns r0=0.
 * Used as replacement target when blx r12 would call a heap/null address.
 * ARM: mov r0, #0  (e3a00000); bx lr  (e12fff1e) */
static constexpr uint32_t NOOP_RET0      = LIBC_DATA + 0x100u;
static constexpr uint32_t STACK_BASE    = 0x42000000u;
static constexpr uint32_t STACK_SIZE    = 0x05000000u;  /* 80MB — Unity Mono init + Boehm GC mark
 * recurse deeply; the GC conservatively
 * scans [sp, stacktop), so the stack must
 * hold the full depth or sp overflows the
 * region (top now 0x47000000 < THREAD_STACK
 * 0x48000000).. */
static constexpr uint32_t SENTINEL_ADDR = 0x43000000u;
/* Dedicated stack for guest callbacks driven from inside an SVC handler (the
 * bsearch comparator).  Sits in the gap between the main stack top (0x47000000)
 * and the co-op thread stacks (0x48000000), so it overlaps neither. */
static constexpr uint32_t CB_STACK_BASE = 0x47700000u;
static constexpr uint32_t CB_STACK_SIZE = 0x00100000u;  /* 1MB */

/* A clean return lands the guest PC on the (non-executable) SENTINEL page.
 * Dynarmic's NoExecuteFault may report the PC as SENTINEL_ADDR or, after a
 * zero-instruction fetch, SENTINEL_ADDR+4 — so any landing inside the page is
 * a normal return, not an abnormal halt.  Use this everywhere the sentinel is
 * tested so callers stay consistent (see also the thread scheduler check). */
static inline bool pc_in_sentinel(uint32_t pc) {
    uint32_t p = pc & ~1u;
    return p >= SENTINEL_ADDR && p < SENTINEL_ADDR + 0x1000u;
}


static constexpr uint32_t JNI_VTABLE_COUNT = 229u; /* indices 0–228 */
static constexpr uint32_t SVC_JVM_GETENV   = 229u;
static constexpr uint32_t SVC_JVM_ATTACH   = 230u;
static constexpr uint32_t SVC_JVM_DESTROY  = 231u;
static constexpr uint32_t SVC_LOG_PRINT    = 232u;
static constexpr uint32_t SVC_LOG_WRITE    = 233u;
static constexpr uint32_t SVC_DLOPEN       = 234u;
static constexpr uint32_t SVC_DLSYM        = 235u;
static constexpr uint32_t SVC_DLCLOSE      = 236u;

static constexpr uint32_t SVC_MALLOC       = 237u;
static constexpr uint32_t SVC_FREE         = 238u;
static constexpr uint32_t SVC_CALLOC       = 239u;
static constexpr uint32_t SVC_REALLOC      = 240u;
static constexpr uint32_t SVC_MEMCPY       = 241u;
static constexpr uint32_t SVC_MEMMOVE      = 242u;
static constexpr uint32_t SVC_MEMSET       = 243u;
static constexpr uint32_t SVC_STRLEN       = 244u;
static constexpr uint32_t SVC_STRCPY       = 245u;
static constexpr uint32_t SVC_STRNCPY      = 246u;
static constexpr uint32_t SVC_STRCMP       = 247u;
static constexpr uint32_t SVC_STRNCMP      = 248u;
static constexpr uint32_t SVC_STRDUP       = 249u;
static constexpr uint32_t SVC_STRNDUP      = 250u;
static constexpr uint32_t SVC_STRCAT       = 251u;
static constexpr uint32_t SVC_STRNCAT      = 252u;
static constexpr uint32_t SVC_ABORT          = 253u;
static constexpr uint32_t SVC_PTHREAD_KEY    = 254u; /* pthread_key_create/set/get/once/mutex/cond */
static constexpr uint32_t SVC_PTHREAD_CREATE = 255u; /* pthread_create — queues fn for deferred run */

static constexpr uint32_t SVC_ANW_FROM_SURFACE   = 256u;
static constexpr uint32_t SVC_ANW_ACQUIRE        = 257u;
static constexpr uint32_t SVC_ANW_RELEASE        = 258u;
static constexpr uint32_t SVC_ANW_GETWIDTH       = 259u;
static constexpr uint32_t SVC_ANW_GETHEIGHT      = 260u;
static constexpr uint32_t SVC_ANW_SETBUFGEO      = 261u;
static constexpr uint32_t SVC_ANW_TOSURFACE      = 262u;

static constexpr uint32_t SVC_EGL_GETDISPLAY     = 263u;
static constexpr uint32_t SVC_EGL_INITIALIZE     = 264u;
static constexpr uint32_t SVC_EGL_CHOOSECONFIG   = 265u;
static constexpr uint32_t SVC_EGL_CREATEWSURF    = 266u;
static constexpr uint32_t SVC_EGL_CREATEPBUF     = 267u;
static constexpr uint32_t SVC_EGL_CREATECTX      = 268u;
static constexpr uint32_t SVC_EGL_MAKECURRENT    = 269u;
static constexpr uint32_t SVC_EGL_SWAPBUF        = 270u;
static constexpr uint32_t SVC_EGL_DESTROYSURF    = 271u;
static constexpr uint32_t SVC_EGL_DESTROYCTX     = 272u;
static constexpr uint32_t SVC_EGL_TERMINATE      = 273u;
static constexpr uint32_t SVC_EGL_GETPROC        = 274u;
static constexpr uint32_t SVC_EGL_QUERYSURF      = 275u;
static constexpr uint32_t SVC_EGL_GETERROR       = 276u;
static constexpr uint32_t SVC_EGL_GETCFGATTRIB   = 277u;
static constexpr uint32_t SVC_EGL_QUERYSTR       = 278u;
static constexpr uint32_t SVC_EGL_SURFACEATTRIB  = 279u;
static constexpr uint32_t SVC_EGL_SWAPINTERVAL   = 280u;
static constexpr uint32_t SVC_EGL_GETCURCTX      = 281u;
static constexpr uint32_t SVC_EGL_GETCURSURF     = 282u;
static constexpr uint32_t SVC_DL_UNWIND_EXIDX   = 283u;


static constexpr uint32_t SVC_GL_BASE            = 284u;
static constexpr uint32_t SVC_GL_Viewport              = 284u;
static constexpr uint32_t SVC_GL_Clear                 = 285u;
static constexpr uint32_t SVC_GL_ClearColor            = 286u;
static constexpr uint32_t SVC_GL_ClearDepthf           = 287u;
static constexpr uint32_t SVC_GL_ClearStencil          = 288u;
static constexpr uint32_t SVC_GL_Enable                = 289u;
static constexpr uint32_t SVC_GL_Disable               = 290u;
static constexpr uint32_t SVC_GL_DepthFunc             = 291u;
static constexpr uint32_t SVC_GL_DepthMask             = 292u;
static constexpr uint32_t SVC_GL_ColorMask             = 293u;
static constexpr uint32_t SVC_GL_Scissor               = 294u;
static constexpr uint32_t SVC_GL_FrontFace             = 295u;
static constexpr uint32_t SVC_GL_CullFace              = 296u;
static constexpr uint32_t SVC_GL_BlendFuncSeparate     = 297u;
static constexpr uint32_t SVC_GL_BlendEquationSeparate = 298u;
static constexpr uint32_t SVC_GL_GetError              = 299u;
static constexpr uint32_t SVC_GL_GetString             = 300u;
static constexpr uint32_t SVC_GL_GetIntegerv           = 301u;
static constexpr uint32_t SVC_GL_PixelStorei           = 302u;
static constexpr uint32_t SVC_GL_ReadPixels            = 303u;
static constexpr uint32_t SVC_GL_Flush                 = 304u;
static constexpr uint32_t SVC_GL_Finish                = 305u;

static constexpr uint32_t SVC_GL_GenBuffers            = 306u;
static constexpr uint32_t SVC_GL_BindBuffer            = 307u;
static constexpr uint32_t SVC_GL_BufferData            = 308u;
static constexpr uint32_t SVC_GL_BufferSubData         = 309u;
static constexpr uint32_t SVC_GL_DeleteBuffers         = 310u;

static constexpr uint32_t SVC_GL_GenTextures           = 311u;
static constexpr uint32_t SVC_GL_BindTexture           = 312u;
static constexpr uint32_t SVC_GL_ActiveTexture         = 313u;
static constexpr uint32_t SVC_GL_DeleteTextures        = 314u;
static constexpr uint32_t SVC_GL_TexParameteri         = 315u;
static constexpr uint32_t SVC_GL_TexImage2D            = 316u;
static constexpr uint32_t SVC_GL_TexSubImage2D         = 317u;
static constexpr uint32_t SVC_GL_CopyTexSubImage2D     = 318u;
static constexpr uint32_t SVC_GL_CompressedTexImage2D  = 319u;
static constexpr uint32_t SVC_GL_CompressedTexSubImage2D = 320u;
static constexpr uint32_t SVC_GL_GenerateMipmap        = 321u;

static constexpr uint32_t SVC_GL_GenFramebuffers       = 322u;
static constexpr uint32_t SVC_GL_BindFramebuffer       = 323u;
static constexpr uint32_t SVC_GL_DeleteFramebuffers    = 324u;
static constexpr uint32_t SVC_GL_CheckFramebufferStatus = 325u;
static constexpr uint32_t SVC_GL_FramebufferTexture2D  = 326u;
static constexpr uint32_t SVC_GL_FramebufferRenderbuffer = 327u;
static constexpr uint32_t SVC_GL_GetFramebufferAttachmentParameteriv = 328u;

static constexpr uint32_t SVC_GL_GenRenderbuffers      = 329u;
static constexpr uint32_t SVC_GL_BindRenderbuffer      = 330u;
static constexpr uint32_t SVC_GL_DeleteRenderbuffers   = 331u;
static constexpr uint32_t SVC_GL_RenderbufferStorage   = 332u;

static constexpr uint32_t SVC_GL_CreateShader          = 333u;
static constexpr uint32_t SVC_GL_ShaderSource          = 334u;
static constexpr uint32_t SVC_GL_CompileShader         = 335u;
static constexpr uint32_t SVC_GL_DeleteShader          = 336u;
static constexpr uint32_t SVC_GL_GetShaderiv           = 337u;
static constexpr uint32_t SVC_GL_GetShaderInfoLog      = 338u;
static constexpr uint32_t SVC_GL_GetShaderSource       = 339u;

static constexpr uint32_t SVC_GL_CreateProgram         = 340u;
static constexpr uint32_t SVC_GL_AttachShader          = 341u;
static constexpr uint32_t SVC_GL_LinkProgram           = 342u;
static constexpr uint32_t SVC_GL_UseProgram            = 343u;
static constexpr uint32_t SVC_GL_DeleteProgram         = 344u;
static constexpr uint32_t SVC_GL_GetProgramiv          = 345u;
static constexpr uint32_t SVC_GL_GetProgramInfoLog     = 346u;
static constexpr uint32_t SVC_GL_GetAttribLocation     = 347u;
static constexpr uint32_t SVC_GL_GetUniformLocation    = 348u;
static constexpr uint32_t SVC_GL_GetActiveAttrib       = 349u;
static constexpr uint32_t SVC_GL_GetActiveUniform      = 350u;
static constexpr uint32_t SVC_GL_BindAttribLocation    = 351u;

static constexpr uint32_t SVC_GL_Uniform1i             = 352u;
static constexpr uint32_t SVC_GL_Uniform1iv            = 353u;
static constexpr uint32_t SVC_GL_Uniform2iv            = 354u;
static constexpr uint32_t SVC_GL_Uniform3iv            = 355u;
static constexpr uint32_t SVC_GL_Uniform4iv            = 356u;
static constexpr uint32_t SVC_GL_Uniform1fv            = 357u;
static constexpr uint32_t SVC_GL_Uniform2fv            = 358u;
static constexpr uint32_t SVC_GL_Uniform3fv            = 359u;
static constexpr uint32_t SVC_GL_Uniform4fv            = 360u;
static constexpr uint32_t SVC_GL_UniformMatrix3fv      = 361u;
static constexpr uint32_t SVC_GL_UniformMatrix4fv      = 362u;

static constexpr uint32_t SVC_GL_EnableVertexAttribArray  = 363u;
static constexpr uint32_t SVC_GL_DisableVertexAttribArray = 364u;
static constexpr uint32_t SVC_GL_VertexAttribPointer   = 365u;
static constexpr uint32_t SVC_GL_GetVertexAttribiv     = 366u;
static constexpr uint32_t SVC_GL_GetVertexAttribPointerv = 367u;

static constexpr uint32_t SVC_GL_DrawArrays            = 368u;
static constexpr uint32_t SVC_GL_DrawElements          = 369u;

static constexpr uint32_t SVC_GL_StencilFunc           = 370u;
static constexpr uint32_t SVC_GL_StencilFuncSeparate   = 371u;
static constexpr uint32_t SVC_GL_StencilMask           = 372u;
static constexpr uint32_t SVC_GL_StencilOp             = 373u;
static constexpr uint32_t SVC_GL_StencilOpSeparate     = 374u;

static constexpr uint32_t SVC_GL_BlendFunc             = 375u;
static constexpr uint32_t SVC_GL_TexParameterf         = 376u;
static constexpr uint32_t SVC_GL_DepthRangef           = 377u;
static constexpr uint32_t SVC_GL_PolygonOffset         = 378u;
static constexpr uint32_t SVC_GL_LineWidth             = 379u;
static constexpr uint32_t SVC_GL_SampleCoverage        = 380u;

static constexpr uint32_t SVC_GL_Uniform1f             = 381u;
static constexpr uint32_t SVC_GL_Uniform2f             = 382u;
static constexpr uint32_t SVC_GL_Uniform3f             = 383u;
static constexpr uint32_t SVC_GL_Uniform4f             = 384u;

static constexpr uint32_t SVC_GL_VertexAttrib1f        = 385u;
static constexpr uint32_t SVC_GL_VertexAttrib2f        = 386u;
static constexpr uint32_t SVC_GL_VertexAttrib3f        = 387u;
static constexpr uint32_t SVC_GL_VertexAttrib4f        = 388u;
static constexpr uint32_t SVC_GL_VertexAttrib4fv       = 389u;

static constexpr uint32_t SVC_GL_GetFloatv             = 390u;
static constexpr uint32_t SVC_GL_GetBooleanv           = 391u;
static constexpr uint32_t SVC_GL_IsEnabled             = 392u;
static constexpr uint32_t SVC_GL_IsProgram             = 393u;
static constexpr uint32_t SVC_GL_IsShader              = 394u;
static constexpr uint32_t SVC_GL_IsTexture             = 395u;
static constexpr uint32_t SVC_GL_IsBuffer              = 396u;
static constexpr uint32_t SVC_GL_IsFramebuffer         = 397u;
static constexpr uint32_t SVC_GL_IsRenderbuffer        = 398u;

static constexpr uint32_t SVC_GL_BlendEquation         = 399u;
static constexpr uint32_t SVC_GL_BlendColor            = 400u;
static constexpr uint32_t SVC_GL_ReleaseShaderCompiler = 401u;
static constexpr uint32_t SVC_GL_GetShaderPrecisionFormat = 402u;
static constexpr uint32_t SVC_GL_UniformMatrix2fv      = 403u;
static constexpr uint32_t SVC_GL_VertexAttrib1fv       = 404u;
static constexpr uint32_t SVC_GL_VertexAttrib2fv       = 405u;
static constexpr uint32_t SVC_GL_VertexAttrib3fv       = 406u;


static constexpr uint32_t SVC_AEABI_UIDIV       = 407u;
static constexpr uint32_t SVC_AEABI_UIDIVMOD    = 408u;
static constexpr uint32_t SVC_AEABI_IDIV        = 409u;
static constexpr uint32_t SVC_AEABI_LDIVMOD     = 410u;
static constexpr uint32_t SVC_AEABI_ULDIVMOD    = 411u; /* unsigned 64-bit division */
static constexpr uint32_t SVC_LIBC_OPEN         = 412u;
static constexpr uint32_t SVC_LIBC_CLOSE        = 413u;
static constexpr uint32_t SVC_LIBC_READ         = 414u;
static constexpr uint32_t SVC_LIBC_WRITE        = 415u;
static constexpr uint32_t SVC_LIBC_LSEEK        = 416u;
static constexpr uint32_t SVC_LIBC_FOPEN        = 417u;
static constexpr uint32_t SVC_LIBC_FCLOSE       = 418u;
static constexpr uint32_t SVC_LIBC_FREAD        = 419u;
static constexpr uint32_t SVC_LIBC_FWRITE       = 420u;
static constexpr uint32_t SVC_LIBC_FSEEK        = 421u;
static constexpr uint32_t SVC_LIBC_FTELL        = 422u;
static constexpr uint32_t SVC_LIBC_STAT         = 423u;
static constexpr uint32_t SVC_LIBC_FSTAT        = 424u;
static constexpr uint32_t SVC_LIBC_MMAP         = 425u;
static constexpr uint32_t SVC_LIBC_MUNMAP       = 426u;


static constexpr uint32_t SVC_CLOCK_GETTIME     = 427u;
static constexpr uint32_t SVC_GETTIMEOFDAY      = 428u;
static constexpr uint32_t SVC_TIME              = 429u;
static constexpr uint32_t SVC_NANOSLEEP         = 430u;
static constexpr uint32_t SVC_USLEEP            = 431u;
static constexpr uint32_t SVC_GETENV            = 432u;
static constexpr uint32_t SVC_GETPID            = 433u;
static constexpr uint32_t SVC_GETTID            = 434u;
static constexpr uint32_t SVC_SCHED_YIELD       = 435u;
static constexpr uint32_t SVC_GETPAGESIZE       = 436u;
static constexpr uint32_t SVC_SYSCONF           = 437u;
static constexpr uint32_t SVC_RET0              = 438u; /* generic success stub */
static constexpr uint32_t SVC_ERRNO_ADDR        = 439u; /* __errno */
static constexpr uint32_t SVC_SYSPROP_GET       = 440u; /* __system_property_get */

static constexpr uint32_t SVC_PTHREAD_SELF        = 441u;
static constexpr uint32_t SVC_PTHREAD_KEY_CREATE  = 442u;
static constexpr uint32_t SVC_PTHREAD_KEY_DELETE  = 443u;
static constexpr uint32_t SVC_PTHREAD_SETSPECIFIC = 444u;
static constexpr uint32_t SVC_PTHREAD_GETSPECIFIC = 445u;

static constexpr uint32_t SVC_Z_INFLATEINIT2    = 446u;
static constexpr uint32_t SVC_Z_INFLATE         = 447u;
static constexpr uint32_t SVC_Z_INFLATEEND      = 448u;
static constexpr uint32_t SVC_Z_INFLATERESET    = 449u;
static constexpr uint32_t SVC_Z_CRC32           = 450u;
static constexpr uint32_t SVC_Z_ADLER32         = 451u;

static constexpr uint32_t SVC_Z_INFLATEINIT     = 452u; /* inflateInit_(z, ver, size) */
static constexpr uint32_t SVC_Z_DEFLATEINIT2    = 453u; /* deflateInit2_(z,lvl,method,wbits,mem,strategy,ver,sz) */
static constexpr uint32_t SVC_Z_DEFLATE         = 454u; /* deflate(z, flush) */
static constexpr uint32_t SVC_Z_DEFLATEEND      = 455u; /* deflateEnd(z) */
static constexpr uint32_t SVC_Z_DEFLATERESET    = 456u; /* deflateReset(z) */

static constexpr uint32_t SVC_ATOI              = 457u;
static constexpr uint32_t SVC_ATOL              = 458u;
static constexpr uint32_t SVC_STRTOL            = 459u;
static constexpr uint32_t SVC_STRTOUL           = 460u;
static constexpr uint32_t SVC_STRTOD            = 461u;
static constexpr uint32_t SVC_STRTOF            = 462u;


static constexpr uint32_t SVC_MATH_F1_BASE      = 540u; /* float  fn(float) */
static constexpr uint32_t SVC_MATH_F1_COUNT     = 26u;
static constexpr uint32_t SVC_MATH_F2_BASE      = SVC_MATH_F1_BASE + SVC_MATH_F1_COUNT;       /* float  fn(float,float) */
static constexpr uint32_t SVC_MATH_F2_COUNT     = 8u;
static constexpr uint32_t SVC_MATH_D1_BASE      = SVC_MATH_F2_BASE + SVC_MATH_F2_COUNT;       /* double fn(double) */
static constexpr uint32_t SVC_MATH_D1_COUNT     = 25u;
static constexpr uint32_t SVC_MATH_D2_BASE      = SVC_MATH_D1_BASE + SVC_MATH_D1_COUNT;       /* double fn(double,double) */
static constexpr uint32_t SVC_MATH_D2_COUNT     = 6u;

/* Extended libc passthrough.  Previously unresolved imports fell through to
 * SVC#0 (silent zero).  mmap returning 0 was especially fatal: Unity's allocator
 * placed pools on guest VA 0 (inside libunity), causing wild jumps and black screens.
 * SVC_EXT_BASE immediately follows the math block (540..604); an older collision at
 * 524 routed math calls into libc handlers (e.g. fmodf → WMEMCPY → SIGSEGV). */

static constexpr uint32_t SVC_EXT_BASE        = SVC_MATH_D2_BASE + SVC_MATH_D2_COUNT;
static constexpr uint32_t SVC_MEMALIGN        = SVC_EXT_BASE + 0u;
static constexpr uint32_t SVC_POSIX_MEMALIGN  = SVC_EXT_BASE + 1u;
static constexpr uint32_t SVC_MEMCMP          = SVC_EXT_BASE + 2u;
static constexpr uint32_t SVC_MEMCHR          = SVC_EXT_BASE + 3u;
static constexpr uint32_t SVC_MEMRCHR         = SVC_EXT_BASE + 4u;
static constexpr uint32_t SVC_MEMMEM          = SVC_EXT_BASE + 5u;
static constexpr uint32_t SVC_STRCHR          = SVC_EXT_BASE + 6u;
static constexpr uint32_t SVC_STRRCHR         = SVC_EXT_BASE + 7u;
static constexpr uint32_t SVC_STRSTR          = SVC_EXT_BASE + 8u;
static constexpr uint32_t SVC_STRNLEN         = SVC_EXT_BASE + 9u;
static constexpr uint32_t SVC_STRCASECMP      = SVC_EXT_BASE + 10u;
static constexpr uint32_t SVC_STRCSPN         = SVC_EXT_BASE + 11u;
static constexpr uint32_t SVC_STRSPN          = SVC_EXT_BASE + 12u;
static constexpr uint32_t SVC_STRTOK_R        = SVC_EXT_BASE + 13u;
static constexpr uint32_t SVC_PTHREAD_EQUAL   = SVC_EXT_BASE + 14u;
static constexpr uint32_t SVC_SNPRINTF        = SVC_EXT_BASE + 15u;
static constexpr uint32_t SVC_SPRINTF         = SVC_EXT_BASE + 16u;
static constexpr uint32_t SVC_VSNPRINTF       = SVC_EXT_BASE + 17u;
static constexpr uint32_t SVC_VASPRINTF       = SVC_EXT_BASE + 18u;
static constexpr uint32_t SVC_PRINTF          = SVC_EXT_BASE + 19u;
static constexpr uint32_t SVC_FPRINTF         = SVC_EXT_BASE + 20u;
static constexpr uint32_t SVC_VPRINTF         = SVC_EXT_BASE + 21u;
static constexpr uint32_t SVC_VFPRINTF        = SVC_EXT_BASE + 22u;
static constexpr uint32_t SVC_PUTS            = SVC_EXT_BASE + 23u;
static constexpr uint32_t SVC_FPUTS           = SVC_EXT_BASE + 24u;
static constexpr uint32_t SVC_FPUTC           = SVC_EXT_BASE + 25u;
static constexpr uint32_t SVC_ASSERT2         = SVC_EXT_BASE + 26u;
static constexpr uint32_t SVC_LOG_VPRINT      = SVC_EXT_BASE + 27u;
static constexpr uint32_t SVC_SINCOS          = SVC_EXT_BASE + 28u;
static constexpr uint32_t SVC_SINCOSF         = SVC_EXT_BASE + 29u;
static constexpr uint32_t SVC_LDEXP           = SVC_EXT_BASE + 30u;
static constexpr uint32_t SVC_LDEXPF          = SVC_EXT_BASE + 31u;
static constexpr uint32_t SVC_MODF            = SVC_EXT_BASE + 32u;
static constexpr uint32_t SVC_MODFF           = SVC_EXT_BASE + 33u;
static constexpr uint32_t SVC_STRTOLL         = SVC_EXT_BASE + 34u;
static constexpr uint32_t SVC_STRTOULL        = SVC_EXT_BASE + 35u;
static constexpr uint32_t SVC_ACCESS          = SVC_EXT_BASE + 36u;
static constexpr uint32_t SVC_REALPATH        = SVC_EXT_BASE + 37u;
static constexpr uint32_t SVC_PREAD           = SVC_EXT_BASE + 38u;
static constexpr uint32_t SVC_PWRITE          = SVC_EXT_BASE + 39u;
static constexpr uint32_t SVC_OPENDIR         = SVC_EXT_BASE + 40u;
static constexpr uint32_t SVC_READDIR         = SVC_EXT_BASE + 41u;
static constexpr uint32_t SVC_CLOSEDIR        = SVC_EXT_BASE + 42u;
static constexpr uint32_t SVC_WCSLEN          = SVC_EXT_BASE + 43u;
static constexpr uint32_t SVC_WMEMCPY         = SVC_EXT_BASE + 44u;
static constexpr uint32_t SVC_WMEMMOVE        = SVC_EXT_BASE + 45u;
static constexpr uint32_t SVC_WMEMSET         = SVC_EXT_BASE + 46u;
static constexpr uint32_t SVC_ISSPACE         = SVC_EXT_BASE + 47u;
static constexpr uint32_t SVC_FGETS           = SVC_EXT_BASE + 48u;
static constexpr uint32_t SVC_FILENO          = SVC_EXT_BASE + 49u;
static constexpr uint32_t SVC_FEOF            = SVC_EXT_BASE + 50u;
static constexpr uint32_t SVC_BASENAME        = SVC_EXT_BASE + 51u;
static constexpr uint32_t SVC_EXIT            = SVC_EXT_BASE + 52u;
/* __aeabi_* / fortify (_chk) / additional libc stubs */
static constexpr uint32_t SVC_AEABI_MEMSET    = SVC_EXT_BASE + 53u; /* (dst, n, c) — note argument order */
static constexpr uint32_t SVC_AEABI_MEMCLR    = SVC_EXT_BASE + 54u; /* (dst, n) */
static constexpr uint32_t SVC_STRLCPY         = SVC_EXT_BASE + 55u;
static constexpr uint32_t SVC_STRNCASECMP     = SVC_EXT_BASE + 56u;
static constexpr uint32_t SVC_TOLOWER         = SVC_EXT_BASE + 57u;
static constexpr uint32_t SVC_ISALPHA         = SVC_EXT_BASE + 58u;
static constexpr uint32_t SVC_ISDIGIT         = SVC_EXT_BASE + 59u;
static constexpr uint32_t SVC_ISALNUM         = SVC_EXT_BASE + 60u;
static constexpr uint32_t SVC_ISXDIGIT        = SVC_EXT_BASE + 61u;
static constexpr uint32_t SVC_MKDIR           = SVC_EXT_BASE + 62u;
static constexpr uint32_t SVC_GETCWD          = SVC_EXT_BASE + 63u;
static constexpr uint32_t SVC_UNLINK          = SVC_EXT_BASE + 64u;
static constexpr uint32_t SVC_RENAME          = SVC_EXT_BASE + 65u;
static constexpr uint32_t SVC_FTRUNCATE       = SVC_EXT_BASE + 66u;
static constexpr uint32_t SVC_READLINK        = SVC_EXT_BASE + 67u;
static constexpr uint32_t SVC_CLOCK           = SVC_EXT_BASE + 68u;
static constexpr uint32_t SVC_LOCALTIME_R     = SVC_EXT_BASE + 69u;
static constexpr uint32_t SVC_GMTIME_R        = SVC_EXT_BASE + 70u;
static constexpr uint32_t SVC_LOCALTIME       = SVC_EXT_BASE + 71u;
static constexpr uint32_t SVC_GMTIME          = SVC_EXT_BASE + 72u;
static constexpr uint32_t SVC_MKTIME          = SVC_EXT_BASE + 73u;
static constexpr uint32_t SVC_DIFFTIME        = SVC_EXT_BASE + 74u;
static constexpr uint32_t SVC_STRFTIME        = SVC_EXT_BASE + 75u;
static constexpr uint32_t SVC_UNAME           = SVC_EXT_BASE + 76u;
static constexpr uint32_t SVC_GETRLIMIT       = SVC_EXT_BASE + 77u;
static constexpr uint32_t SVC_MREMAP          = SVC_EXT_BASE + 78u;
static constexpr uint32_t SVC_WRITEV          = SVC_EXT_BASE + 79u;
static constexpr uint32_t SVC_STRERROR        = SVC_EXT_BASE + 80u;
static constexpr uint32_t SVC_SETLOCALE       = SVC_EXT_BASE + 81u;
static constexpr uint32_t SVC_RETM1           = SVC_EXT_BASE + 82u; /* stub returning -1 on failure */
static constexpr uint32_t SVC_PTHREAD_GETATTR_NP      = SVC_EXT_BASE + 83u;
static constexpr uint32_t SVC_PTHREAD_ATTR_GETSTACK   = SVC_EXT_BASE + 84u;
static constexpr uint32_t SVC_PTHREAD_ATTR_GETSTACKSZ = SVC_EXT_BASE + 85u;
static constexpr uint32_t SVC_VSNPRINTF_CHK   = SVC_EXT_BASE + 86u;
static constexpr uint32_t SVC_VSPRINTF_CHK    = SVC_EXT_BASE + 87u;
static constexpr uint32_t SVC_ABS             = SVC_EXT_BASE + 88u;
static constexpr uint32_t SVC_SSCANF          = SVC_EXT_BASE + 89u;
static constexpr uint32_t SVC_VSSCANF         = SVC_EXT_BASE + 90u;
static constexpr uint32_t SVC_ISASCII         = SVC_EXT_BASE + 91u;
static constexpr uint32_t SVC_LIBC_MMAP2      = SVC_EXT_BASE + 92u; /* __mmap2: offset is in pages */
static constexpr uint32_t SVC_WAIT            = SVC_EXT_BASE + 93u; /* cond_wait/join: yield slice each call */
/* Semaphores (real counters) — required for Boehm GC thread registration */
static constexpr uint32_t SVC_SEM_INIT        = SVC_EXT_BASE + 94u;
static constexpr uint32_t SVC_SEM_POST        = SVC_EXT_BASE + 95u;
static constexpr uint32_t SVC_SEM_WAIT        = SVC_EXT_BASE + 96u;
static constexpr uint32_t SVC_SEM_TRYWAIT     = SVC_EXT_BASE + 97u;
static constexpr uint32_t SVC_SEM_TIMEDWAIT   = SVC_EXT_BASE + 98u;
static constexpr uint32_t SVC_SEM_DESTROY     = SVC_EXT_BASE + 99u;
static constexpr uint32_t SVC_SEM_GETVALUE    = SVC_EXT_BASE + 100u;
/* setjmp/longjmp: save/restore context in jmp_buf.  Trampolines are
 * SVC+BX LR, so restoring lr and executing BX LR completes longjmp. */
static constexpr uint32_t SVC_SETJMP          = SVC_EXT_BASE + 101u;
static constexpr uint32_t SVC_LONGJMP         = SVC_EXT_BASE + 102u;
/* ARM Linux kuser helpers (called via BLX 0xffff0fc0 / 0xffff0fe0) */
static constexpr uint32_t SVC_KUSER_CMPXCHG   = SVC_EXT_BASE + 103u;
static constexpr uint32_t SVC_KUSER_GET_TLS   = SVC_EXT_BASE + 104u;
static constexpr uint32_t SVC_SYSCALL         = SVC_EXT_BASE + 105u; /* syscall() shim (futex/gettid) */
static constexpr uint32_t SVC_SBRK           = SVC_EXT_BASE + 106u;
static constexpr uint32_t SVC_BRK            = SVC_EXT_BASE + 107u;

static constexpr uint32_t SVC_ALOOPER_FORTHREAD = SVC_EXT_BASE + 108u;
static constexpr uint32_t SVC_ALOOPER_PREPARE   = SVC_EXT_BASE + 109u;
static constexpr uint32_t SVC_ALOOPER_POLLONCE  = SVC_EXT_BASE + 110u;
static constexpr uint32_t SVC_ALOOPER_POLLALL   = SVC_EXT_BASE + 111u;
static constexpr uint32_t SVC_ALOOPER_WAKE      = SVC_EXT_BASE + 112u;
static constexpr uint32_t SVC_STATFS            = SVC_EXT_BASE + 113u;
static constexpr uint32_t SVC_STATVFS           = SVC_EXT_BASE + 114u;
static constexpr uint32_t SVC_CHDIR             = SVC_EXT_BASE + 115u;
/* reliable exception-identification logging SVCs.  Emitted by the
 * guest stubs built in build_exc_logger_stub (NOT trampolines) — they log the
 * args and then tail-call the real libmono function, so they fire on every
 * PLT/GOT call (unlike AddTicks PC sampling, which misses function entries). */
static constexpr uint32_t SVC_EXC_FROM_NAME     = SVC_EXT_BASE + 116u;
static constexpr uint32_t SVC_EXC_RAISE         = SVC_EXT_BASE + 117u;
/* getdtablesize() — required by mono's io-layer _wapi_handle_init
 * to size _wapi_fd_reserve.  Was unresolved (returned garbage), corrupting the
 * WAPI handle-table segment layout so handles read back as "unused". */
static constexpr uint32_t SVC_GETDTABLESIZE     = SVC_EXT_BASE + 118u;
/* bsearch with a guest comparator callback.  libmono imports
 * bsearch (and qsort) from libc; without this it bound to the raw-syscall
 * trampoline (svc #0) and returned garbage, so mono_metadata_typedef_from_method
 * could never find a method's owning TypeDef.  Every mono_get_method_from_token
 * then returned NULL, every JIT compile of a method containing a `call` set
 * cfg->exception_type=TYPE_LOAD, and the engine spun on a TypeLoadException
 * cascade forever (black screen). */
static constexpr uint32_t SVC_BSEARCH           = SVC_EXT_BASE + 119u;
/* inline-detour logging SVCs (LUNARIA_TRACE_EXC).  Each installed
 * detour emits svc #(SVC_DETOUR_BASE+n) as the first instruction of a patched
 * libmono function — reliably catches libmono-internal direct-bl calls that
 * neither PC sampling nor GOT/PLT redirect can see */
static constexpr uint32_t SVC_DETOUR_BASE       = SVC_EXT_BASE + 120u;
static constexpr uint32_t NUM_DETOURS           = 20u;

/* Itanium/ARM C++ ABI one-time static-initialisation guards.  Placed after the
 * detour block so they never collide with SVC_DETOUR_BASE+n.
 * Compiler-generated code checks the guard's first byte inline (ldrb; bne) and
 * only calls __cxa_guard_acquire when it is still 0; on a return of 1 it runs
 * the initialiser then calls __cxa_guard_release.  Previously these symbols were
 * unresolved → bound to the raw-syscall trampoline (svc #0), whose default arm
 * returns 0, so acquire always reported "already initialised / keep waiting".
 * Function-local statics in libunity therefore never latched and Unity re-ran
 * engine init (RegisterAllClasses → "ClassID N already registered") every
 * nativeRender frame, never reaching the play/draw loop → black screen.
 * These are pure guest-memory operations (they never call back into guest
 * code), so unlike pthread_once they can be plain SVC handlers. */
static constexpr uint32_t SVC_CXA_GUARD_ACQUIRE = SVC_DETOUR_BASE + NUM_DETOURS + 0u;
static constexpr uint32_t SVC_CXA_GUARD_RELEASE = SVC_DETOUR_BASE + NUM_DETOURS + 1u;
static constexpr uint32_t SVC_CXA_GUARD_ABORT   = SVC_DETOUR_BASE + NUM_DETOURS + 2u;
static constexpr uint32_t SVC_CXA_PURE_VIRTUAL  = SVC_DETOUR_BASE + NUM_DETOURS + 3u;
static constexpr uint32_t SVC_AEABI_ATEXIT      = SVC_DETOUR_BASE + NUM_DETOURS + 4u;


static constexpr uint32_t SVC_BTOWC             = SVC_DETOUR_BASE + NUM_DETOURS + 5u;
static constexpr uint32_t SVC_WCTOB             = SVC_DETOUR_BASE + NUM_DETOURS + 6u;
static constexpr uint32_t SVC_TOWLOWER          = SVC_DETOUR_BASE + NUM_DETOURS + 7u;
static constexpr uint32_t SVC_TOWUPPER          = SVC_DETOUR_BASE + NUM_DETOURS + 8u;
static constexpr uint32_t SVC_ISWCTYPE          = SVC_DETOUR_BASE + NUM_DETOURS + 9u;
static constexpr uint32_t SVC_WCTYPE            = SVC_DETOUR_BASE + NUM_DETOURS + 10u;
static constexpr uint32_t SVC_MBRTOWC           = SVC_DETOUR_BASE + NUM_DETOURS + 11u;
static constexpr uint32_t SVC_WCRTOMB           = SVC_DETOUR_BASE + NUM_DETOURS + 12u;
static constexpr uint32_t SVC_WMEMCHR           = SVC_DETOUR_BASE + NUM_DETOURS + 13u;
static constexpr uint32_t SVC_STRCOLL           = SVC_DETOUR_BASE + NUM_DETOURS + 14u;
static constexpr uint32_t SVC_STRXFRM           = SVC_DETOUR_BASE + NUM_DETOURS + 15u;
static constexpr uint32_t SVC_STRCASESTR        = SVC_DETOUR_BASE + NUM_DETOURS + 16u;
static constexpr uint32_t SVC_STRSEP            = SVC_DETOUR_BASE + NUM_DETOURS + 17u;

/* fp classification, fenv, wide-char classification/conversion */
static constexpr uint32_t SVC_ISNAN             = SVC_DETOUR_BASE + NUM_DETOURS + 18u;
static constexpr uint32_t SVC_ISINF             = SVC_DETOUR_BASE + NUM_DETOURS + 19u;
static constexpr uint32_t SVC_ISFINITE          = SVC_DETOUR_BASE + NUM_DETOURS + 20u;
static constexpr uint32_t SVC_SIGNBIT           = SVC_DETOUR_BASE + NUM_DETOURS + 21u;
static constexpr uint32_t SVC_FEGETROUND        = SVC_DETOUR_BASE + NUM_DETOURS + 22u;
static constexpr uint32_t SVC_FESETROUND        = SVC_DETOUR_BASE + NUM_DETOURS + 23u;
static constexpr uint32_t SVC_FECLEAREXCEPT     = SVC_DETOUR_BASE + NUM_DETOURS + 24u;
static constexpr uint32_t SVC_FERAISEEXCEPT     = SVC_DETOUR_BASE + NUM_DETOURS + 25u;
static constexpr uint32_t SVC_FETESTEXCEPT      = SVC_DETOUR_BASE + NUM_DETOURS + 26u;
static constexpr uint32_t SVC_ISWSPACE          = SVC_DETOUR_BASE + NUM_DETOURS + 27u;
static constexpr uint32_t SVC_ISWDIGIT          = SVC_DETOUR_BASE + NUM_DETOURS + 28u;
static constexpr uint32_t SVC_ISWALPHA          = SVC_DETOUR_BASE + NUM_DETOURS + 29u;
static constexpr uint32_t SVC_ISWUPPER          = SVC_DETOUR_BASE + NUM_DETOURS + 30u;
static constexpr uint32_t SVC_ISWLOWER          = SVC_DETOUR_BASE + NUM_DETOURS + 31u;
static constexpr uint32_t SVC_ISWPRINT          = SVC_DETOUR_BASE + NUM_DETOURS + 32u;
static constexpr uint32_t SVC_ISWPUNCT          = SVC_DETOUR_BASE + NUM_DETOURS + 33u;
static constexpr uint32_t SVC_ISWGRAPH          = SVC_DETOUR_BASE + NUM_DETOURS + 34u;
static constexpr uint32_t SVC_ISWALNUM          = SVC_DETOUR_BASE + NUM_DETOURS + 35u;
static constexpr uint32_t SVC_ISWBLANK          = SVC_DETOUR_BASE + NUM_DETOURS + 36u;
static constexpr uint32_t SVC_ISWCNTRL          = SVC_DETOUR_BASE + NUM_DETOURS + 37u;
static constexpr uint32_t SVC_WCTRANS           = SVC_DETOUR_BASE + NUM_DETOURS + 38u;
static constexpr uint32_t SVC_TOWCTRANS         = SVC_DETOUR_BASE + NUM_DETOURS + 39u;
static constexpr uint32_t SVC_STRTOLD           = SVC_DETOUR_BASE + NUM_DETOURS + 40u;
static constexpr uint32_t SVC_WCSTOD            = SVC_DETOUR_BASE + NUM_DETOURS + 41u;
static constexpr uint32_t SVC_WCSTOL            = SVC_DETOUR_BASE + NUM_DETOURS + 42u;
static constexpr uint32_t SVC_WCSTOUL           = SVC_DETOUR_BASE + NUM_DETOURS + 43u;
static constexpr uint32_t SVC_WCSTOLL           = SVC_DETOUR_BASE + NUM_DETOURS + 44u;
static constexpr uint32_t SVC_WCSTOULL          = SVC_DETOUR_BASE + NUM_DETOURS + 45u;
static constexpr uint32_t SVC_STRTOIMAX         = SVC_DETOUR_BASE + NUM_DETOURS + 46u;
static constexpr uint32_t SVC_STRTOUMAX         = SVC_DETOUR_BASE + NUM_DETOURS + 47u;
/* SVC_WCSLEN already defined at SVC_EXT_BASE+43 — skip duplicate */
static constexpr uint32_t SVC_WCSNCMP           = SVC_DETOUR_BASE + NUM_DETOURS + 49u;
static constexpr uint32_t SVC_WCSCMP            = SVC_DETOUR_BASE + NUM_DETOURS + 50u;
static constexpr uint32_t SVC_WCSCPY            = SVC_DETOUR_BASE + NUM_DETOURS + 51u;
static constexpr uint32_t SVC_WCSCAT            = SVC_DETOUR_BASE + NUM_DETOURS + 52u;
static constexpr uint32_t SVC_DIV               = SVC_DETOUR_BASE + NUM_DETOURS + 53u;
static constexpr uint32_t SVC_LDIV              = SVC_DETOUR_BASE + NUM_DETOURS + 54u;
/* Unresolved-symbol stubs (instead of tramp(0)) for distinct bind vs runtime logs */
static constexpr uint32_t SVC_UNKNOWN_CALL      = SVC_DETOUR_BASE + NUM_DETOURS + 55u;

static constexpr uint32_t SVC_FREXP             = SVC_DETOUR_BASE + NUM_DETOURS + 56u;
static constexpr uint32_t SVC_RINT              = SVC_DETOUR_BASE + NUM_DETOURS + 57u;
static constexpr uint32_t SVC_LRAND48           = SVC_DETOUR_BASE + NUM_DETOURS + 58u;
static constexpr uint32_t SVC_SRAND48           = SVC_DETOUR_BASE + NUM_DETOURS + 59u;
static constexpr uint32_t SVC_STRPBRK           = SVC_DETOUR_BASE + NUM_DETOURS + 60u;
static constexpr uint32_t SVC_STRTOK            = SVC_DETOUR_BASE + NUM_DETOURS + 61u;
static constexpr uint32_t SVC_DUP2              = SVC_DETOUR_BASE + NUM_DETOURS + 62u;
static constexpr uint32_t SVC_CLOCK_GETRES      = SVC_DETOUR_BASE + NUM_DETOURS + 63u;
static constexpr uint32_t SVC_GETHOSTNAME       = SVC_DETOUR_BASE + NUM_DETOURS + 64u;
static constexpr uint32_t SVC_GETRUSAGE         = SVC_DETOUR_BASE + NUM_DETOURS + 65u;
static constexpr uint32_t SVC_VSPRINTF2         = SVC_DETOUR_BASE + NUM_DETOURS + 66u;
static constexpr uint32_t SVC_GETC              = SVC_DETOUR_BASE + NUM_DETOURS + 67u;
static constexpr uint32_t SVC_PUTCHAR           = SVC_DETOUR_BASE + NUM_DETOURS + 68u;
static constexpr uint32_t SVC_FPCLASSIFYF       = SVC_DETOUR_BASE + NUM_DETOURS + 69u;
static constexpr uint32_t SVC_INET_ADDR         = SVC_DETOUR_BASE + NUM_DETOURS + 70u;
static constexpr uint32_t SVC_FCNTL2            = SVC_DETOUR_BASE + NUM_DETOURS + 71u;
static constexpr uint32_t SVC_MONO_PATH_NORM    = SVC_DETOUR_BASE + NUM_DETOURS + 72u;
static constexpr uint32_t SVC_G_FILENAME_URI    = SVC_DETOUR_BASE + NUM_DETOURS + 73u;
static constexpr uint32_t SVC_MONO_FILE_MAP_OPEN = SVC_DETOUR_BASE + NUM_DETOURS + 102u;
static constexpr uint32_t SVC_MONO_FILE_MAP_SIZE = SVC_DETOUR_BASE + NUM_DETOURS + 103u;
static constexpr uint32_t SVC_MONO_FILE_MAP_FD   = SVC_DETOUR_BASE + NUM_DETOURS + 104u;
static constexpr uint32_t SVC_MONO_FILE_MAP      = SVC_DETOUR_BASE + NUM_DETOURS + 105u;
static constexpr uint32_t SVC_G_FILENAME_FROM_URI = SVC_DETOUR_BASE + NUM_DETOURS + 106u;
static constexpr uint32_t SVC_MONO_FILE_MAP_CLOSE = SVC_DETOUR_BASE + NUM_DETOURS + 107u;

static constexpr uint32_t SVC_LIBC_LSEEK64      = SVC_DETOUR_BASE + NUM_DETOURS + 74u; /* lseek64(fd, r1:r2, whence_r3) */

static constexpr uint32_t SVC_PTHREAD_MUTEX_INIT    = SVC_DETOUR_BASE + NUM_DETOURS + 75u;
static constexpr uint32_t SVC_PTHREAD_MUTEX_LOCK    = SVC_DETOUR_BASE + NUM_DETOURS + 76u;
static constexpr uint32_t SVC_PTHREAD_MUTEX_TRYLOCK = SVC_DETOUR_BASE + NUM_DETOURS + 77u;
static constexpr uint32_t SVC_PTHREAD_MUTEX_UNLOCK  = SVC_DETOUR_BASE + NUM_DETOURS + 78u;
static constexpr uint32_t SVC_PTHREAD_MUTEX_DESTROY = SVC_DETOUR_BASE + NUM_DETOURS + 79u;
static constexpr uint32_t SVC_PTHREAD_COND_INIT     = SVC_DETOUR_BASE + NUM_DETOURS + 80u;
static constexpr uint32_t SVC_PTHREAD_COND_DESTROY  = SVC_DETOUR_BASE + NUM_DETOURS + 81u;
static constexpr uint32_t SVC_PTHREAD_COND_SIGNAL   = SVC_DETOUR_BASE + NUM_DETOURS + 82u;
static constexpr uint32_t SVC_PTHREAD_COND_BROADCAST= SVC_DETOUR_BASE + NUM_DETOURS + 83u;
static constexpr uint32_t SVC_PTHREAD_EXIT          = SVC_DETOUR_BASE + NUM_DETOURS + 84u;
static constexpr uint32_t SVC_PTHREAD_ATTR_NOOP     = SVC_DETOUR_BASE + NUM_DETOURS + 85u; /* init/destroy/setstacksize etc */
static constexpr uint32_t SVC_PTHREAD_MUTEXATTR_NOOP= SVC_DETOUR_BASE + NUM_DETOURS + 86u; /* mutexattr_init/settype/destroy */
static constexpr uint32_t SVC_PTHREAD_CONDATTR_NOOP = SVC_DETOUR_BASE + NUM_DETOURS + 87u; /* condattr_init/setclock/destroy */
/* pthread_rwlock: no real blocking under cooperative scheduling; track state for EBUSY */
static constexpr uint32_t SVC_PTHREAD_RWLOCK_INIT     = SVC_DETOUR_BASE + NUM_DETOURS + 88u;
static constexpr uint32_t SVC_PTHREAD_RWLOCK_RDLOCK   = SVC_DETOUR_BASE + NUM_DETOURS + 89u;
static constexpr uint32_t SVC_PTHREAD_RWLOCK_WRLOCK   = SVC_DETOUR_BASE + NUM_DETOURS + 90u;
static constexpr uint32_t SVC_PTHREAD_RWLOCK_UNLOCK   = SVC_DETOUR_BASE + NUM_DETOURS + 91u;
static constexpr uint32_t SVC_PTHREAD_RWLOCK_DESTROY  = SVC_DETOUR_BASE + NUM_DETOURS + 92u;
static constexpr uint32_t SVC_PTHREAD_RWLOCK_TRYRDLOCK= SVC_DETOUR_BASE + NUM_DETOURS + 93u;
static constexpr uint32_t SVC_PTHREAD_RWLOCK_TRYWRLOCK= SVC_DETOUR_BASE + NUM_DETOURS + 94u;
/* pthread_join: wait for target thread finished flag */
static constexpr uint32_t SVC_PTHREAD_JOIN            = SVC_DETOUR_BASE + NUM_DETOURS + 95u;
/* pthread_detach: mark thread detached */
static constexpr uint32_t SVC_PTHREAD_DETACH          = SVC_DETOUR_BASE + NUM_DETOURS + 96u;
/* pthread_cond_wait/timedwait: check cond and schedule */
static constexpr uint32_t SVC_PTHREAD_COND_WAIT       = SVC_DETOUR_BASE + NUM_DETOURS + 97u;
static constexpr uint32_t SVC_PTHREAD_COND_TIMEDWAIT  = SVC_DETOUR_BASE + NUM_DETOURS + 98u;
/* qsort: invoke guest comparator via call_guest_cb */
static constexpr uint32_t SVC_QSORT                   = SVC_DETOUR_BASE + NUM_DETOURS + 99u;
/* fdopen: register fd in g_file_tab, return guest shim */
static constexpr uint32_t SVC_FDOPEN                  = SVC_DETOUR_BASE + NUM_DETOURS + 100u;
/* strerror_r: write error string into buffer */
static constexpr uint32_t SVC_STRERROR_R              = SVC_DETOUR_BASE + NUM_DETOURS + 101u;
static constexpr uint32_t SVC_TOTAL              = SVC_DETOUR_BASE + NUM_DETOURS + 109u;


static constexpr uint32_t SVC29_BASE             = SVC_TOTAL;

static constexpr uint32_t SVC_FMA                = SVC29_BASE + 0u;  /* fma(double,double,double) */
static constexpr uint32_t SVC_FMAF               = SVC29_BASE + 1u;  /* fmaf(float,float,float) */
static constexpr uint32_t SVC_SCALBN             = SVC29_BASE + 2u;  /* scalbn(double,int) */
static constexpr uint32_t SVC_SCALBNF            = SVC29_BASE + 3u;  /* scalbnf(float,int) */
static constexpr uint32_t SVC_ILOGB              = SVC29_BASE + 4u;  /* ilogb(double)->int */
static constexpr uint32_t SVC_ILOGBF             = SVC29_BASE + 5u;  /* ilogbf(float)->int */

static constexpr uint32_t SVC_DUP                = SVC29_BASE + 6u;  /* dup(fd) */
static constexpr uint32_t SVC_FERROR             = SVC29_BASE + 7u;  /* ferror(FILE*) */
static constexpr uint32_t SVC_REWINDDIR          = SVC29_BASE + 8u;  /* rewinddir(DIR*) */
static constexpr uint32_t SVC_MBTOWC             = SVC29_BASE + 9u;  /* mbtowc(pwc,s,n) */
static constexpr uint32_t SVC_MBRLEN             = SVC29_BASE + 10u; /* mbrlen(s,n,ps) */
static constexpr uint32_t SVC_MBSRTOWCS          = SVC29_BASE + 11u; /* mbsrtowcs(dst,src,n,ps) */

static constexpr uint32_t SVC_LOGB               = SVC29_BASE + 12u; /* logb(double)->double */
static constexpr uint32_t SVC_LRINTF             = SVC29_BASE + 13u; /* lrintf(float)->int */
static constexpr uint32_t SVC_EXPM1F             = SVC29_BASE + 14u; /* expm1f(float)->float */
static constexpr uint32_t SVC_NANF               = SVC29_BASE + 15u; /* nanf(const char*)->float */
static constexpr uint32_t SVC_WMEMCMP            = SVC29_BASE + 16u; /* wmemcmp(s1,s2,n)->int */
static constexpr uint32_t SVC_SWPRINTF           = SVC29_BASE + 17u; /* swprintf(buf,n,fmt,...)->int */
static constexpr uint32_t SVC_LOCALECONV         = SVC29_BASE + 18u; /* localeconv()->struct lconv* */
static constexpr uint32_t SVC_SOCKETPAIR         = SVC29_BASE + 19u; /* socketpair(dom,type,prot,sv) */
static constexpr uint32_t SVC_ACFG_SDKVER        = SVC29_BASE + 20u; /* AConfiguration_getSdkVersion */
static constexpr uint32_t SVC_ACHOREOGRAPHER_GET = SVC29_BASE + 21u; /* AChoreographer_getInstance */
static constexpr uint32_t SVC_AASSETMGR_FROMJAVA = SVC29_BASE + 22u; /* AAssetManager_fromJava */
static constexpr uint32_t SVC_AASSETMGR_OPEN     = SVC29_BASE + 23u; /* AAssetManager_open */
static constexpr uint32_t SVC_AASSET_GETBUFFER   = SVC29_BASE + 24u; /* AAsset_getBuffer */
static constexpr uint32_t SVC_AASSET_GETLENGTH   = SVC29_BASE + 25u; /* AAsset_getLength */
static constexpr uint32_t SVC_ACHOREOGRAPHER_POST      = SVC29_BASE + 26u; /* postFrameCallback */
static constexpr uint32_t SVC_ACHOREOGRAPHER_POST64    = SVC29_BASE + 27u; /* postFrameCallback64 */
static constexpr uint32_t SVC_ACHOREOGRAPHER_POSTDELAY = SVC29_BASE + 28u; /* postFrameCallbackDelayed */
static constexpr uint32_t SVC_MONO_PREP            = SVC29_BASE + 29u; /* mono config before jit init */
static constexpr uint32_t SVC29_TOTAL            = SVC29_BASE + 30u;

/* ---- GC signal / sigaction ---- */
static constexpr uint32_t SVC31_BASE             = SVC29_TOTAL;
static constexpr uint32_t SVC_SIGACTION          = SVC31_BASE + 0u; /* sigaction(signum,new,old) */
static constexpr uint32_t SVC_PTHREAD_KILL       = SVC31_BASE + 1u; /* pthread_kill/tkill/kill(tid,sig) */
static constexpr uint32_t SVC_BSD_SIGNAL         = SVC31_BASE + 2u; /* bsd_signal(signum,handler) */
static constexpr uint32_t SVC_EGL_SYSTIME_FREQ   = SVC31_BASE + 3u; /* eglGetSystemTimeFrequencyNV() → u64 ticks/s */
static constexpr uint32_t SVC_EGL_SYSTIME        = SVC31_BASE + 4u; /* eglGetSystemTimeNV() → u64 ticks */
static constexpr uint32_t SVC_SIGSUSPEND         = SVC31_BASE + 5u; /* sigsuspend(mask): GC suspend loop */

/* pipe/pipe2: host-backed pipes (fds live in the same table as open() fds) */
static constexpr uint32_t SVC_PIPE               = SVC31_BASE + 6u;
static constexpr uint32_t SVC_PIPE2              = SVC31_BASE + 7u;
static constexpr uint32_t SVC_ALOOPER_ADDFD      = SVC31_BASE + 8u; /* ALooper_addFd → 1 on success */

/* ---- AAudio (FMOD output/recorder path; API 26+) ----
 * Minimal stubs: streams open successfully with plausible parameters but no
 * host audio I/O.  Setters/start/stop/close map to SVC_RET0 (AAUDIO_OK). */
static constexpr uint32_t SVC_AAUDIO_CREATE_BUILDER = SVC31_BASE + 9u;  /* AAudio_createStreamBuilder(**b) */
static constexpr uint32_t SVC_AAUDIO_OPEN_STREAM    = SVC31_BASE + 10u; /* AAudioStreamBuilder_openStream(b,**s) */
static constexpr uint32_t SVC_AAUDIO_GET_FPB        = SVC31_BASE + 11u; /* AAudioStream_getFramesPerBurst */
static constexpr uint32_t SVC_AAUDIO_GET_BUFSIZE    = SVC31_BASE + 12u; /* AAudioStream_getBufferSizeInFrames */
static constexpr uint32_t SVC_AAUDIO_SET_BUFSIZE    = SVC31_BASE + 13u; /* AAudioStream_setBufferSizeInFrames */
static constexpr uint32_t SVC_AAUDIO_GET_BUFCAP     = SVC31_BASE + 14u; /* AAudioStream_getBufferCapacityInFrames */
static constexpr uint32_t SVC_AAUDIO_WAIT_STATE     = SVC31_BASE + 15u; /* AAudioStream_waitForStateChange */

/* getauxval(type): FMOD dlopen()s libc.so and dlsym()s this to read AT_HWCAP
 * for NEON detection; failure makes it assume no SIMD hardware. */
static constexpr uint32_t SVC_GETAUXVAL             = SVC31_BASE + 16u;

/* AAudio builder setters that must record state for callback pumping */
static constexpr uint32_t SVC_AAUDIO_SET_DIRECTION  = SVC31_BASE + 17u;
static constexpr uint32_t SVC_AAUDIO_SET_DATA_CB    = SVC31_BASE + 18u;
static constexpr uint32_t SVC_AAUDIO_SET_FORMAT     = SVC31_BASE + 19u;
static constexpr uint32_t SVC_AAUDIO_SET_CHANNELS   = SVC31_BASE + 20u;
static constexpr uint32_t SVC_AAUDIO_SET_RATE       = SVC31_BASE + 21u;
static constexpr uint32_t SVC_AAUDIO_START          = SVC31_BASE + 22u;
static constexpr uint32_t SVC_AAUDIO_STOP           = SVC31_BASE + 23u; /* stop + close */

/* Shared SVC behind per-symbol stub trampolines for dlsym'd-but-unimplemented
 * functions.  Each unknown symbol gets its own trampoline slot past
 * SVC_TRAMP_TOTAL, so the handler can recover the symbol name from the PC
 * and log which unimplemented function the guest is actually calling. */
static constexpr uint32_t SVC_UNKNOWN_SYM           = SVC31_BASE + 24u;

/* ---- GLES 3.x entry points (Unity's GLES3 GfxDevice resolves these via
 * dlsym; the 284-406 GL block is full, so they live here) ---- */
static constexpr uint32_t SVC_GL3_GetStringi             = SVC31_BASE + 25u;
static constexpr uint32_t SVC_GL3_GetIntegeri_v          = SVC31_BASE + 26u;
static constexpr uint32_t SVC_GL3_GetInternalformativ    = SVC31_BASE + 27u;
static constexpr uint32_t SVC_GL3_GetProgramInterfaceiv  = SVC31_BASE + 28u;
static constexpr uint32_t SVC_GL3_GetProgramResourceiv   = SVC31_BASE + 29u;
static constexpr uint32_t SVC_GL3_GetProgramResourceName = SVC31_BASE + 30u;
static constexpr uint32_t SVC_GL3_GenVertexArrays        = SVC31_BASE + 31u;
static constexpr uint32_t SVC_GL3_BindVertexArray        = SVC31_BASE + 32u;
static constexpr uint32_t SVC_GL3_DeleteVertexArrays     = SVC31_BASE + 33u;
static constexpr uint32_t SVC_GL3_IsVertexArray          = SVC31_BASE + 34u;
static constexpr uint32_t SVC_GL3_BindSampler            = SVC31_BASE + 35u;
static constexpr uint32_t SVC_GL3_BindBufferBase         = SVC31_BASE + 36u;
static constexpr uint32_t SVC_GL3_BindBufferRange        = SVC31_BASE + 37u;
static constexpr uint32_t SVC_GL3_MapBufferRange         = SVC31_BASE + 38u;
static constexpr uint32_t SVC_GL3_UnmapBuffer            = SVC31_BASE + 39u;
static constexpr uint32_t SVC_GL3_FlushMappedBufferRange = SVC31_BASE + 40u;
static constexpr uint32_t SVC_GL3_TexStorage2D           = SVC31_BASE + 41u;
static constexpr uint32_t SVC_GL3_TexStorage3D           = SVC31_BASE + 42u;
static constexpr uint32_t SVC_GL3_TexSubImage3D          = SVC31_BASE + 43u;
static constexpr uint32_t SVC_GL3_ProgramParameteri      = SVC31_BASE + 44u;
static constexpr uint32_t SVC_GL3_GetProgramBinary       = SVC31_BASE + 45u;
static constexpr uint32_t SVC_GL3_ProgramBinary          = SVC31_BASE + 46u;
static constexpr uint32_t SVC_GL3_FenceSync              = SVC31_BASE + 47u;
static constexpr uint32_t SVC_GL3_ClientWaitSync         = SVC31_BASE + 48u;
static constexpr uint32_t SVC_GL3_DeleteSync             = SVC31_BASE + 49u;
static constexpr uint32_t SVC_GL3_InvalidateFramebuffer  = SVC31_BASE + 50u;
static constexpr uint32_t SVC_GL3_DetachShader           = SVC31_BASE + 51u;
static constexpr uint32_t SVC_GL3_DrawBuffers            = SVC31_BASE + 52u;

static constexpr uint32_t SVC_TRAMP_TOTAL        = SVC_GL3_DrawBuffers + 1u;

static constexpr uint32_t TRAMP_STRIDE     = 8u; /* ARM32: SVC #n + BX LR */

/* dlsym'd-but-unimplemented symbols: slot i lives at trampoline index
 * SVC_TRAMP_TOTAL + i and executes svc #SVC_UNKNOWN_SYM. */
static std::vector<std::string> g_unknown_sym_names;
static std::map<std::string, uint32_t> g_unknown_sym_slot;


static constexpr uint32_t JVM_SLOT_RESERVED0  = 0;
static constexpr uint32_t JVM_SLOT_RESERVED1  = 1;
static constexpr uint32_t JVM_SLOT_RESERVED2  = 2;
static constexpr uint32_t JVM_SLOT_DESTROY    = 3;
static constexpr uint32_t JVM_SLOT_ATTACH     = 4;
static constexpr uint32_t JVM_SLOT_DETACH     = 5;
static constexpr uint32_t JVM_SLOT_GETENV     = 6;
static constexpr uint32_t JVM_SLOT_ATTACH_DA  = 7;
static constexpr uint32_t JVM_SLOT_COUNT      = 8;

static const std::pair<const char *, uint32_t> kSymbolSvcMap[] = {
    {"__android_log_print",  SVC_LOG_PRINT},
    {"__android_log_write",  SVC_LOG_WRITE},
    {"dlopen",               SVC_DLOPEN},
    {"dlsym",                SVC_DLSYM},
    {"dlclose",              SVC_DLCLOSE},
    {"malloc",               SVC_MALLOC},
    {"free",                 SVC_FREE},
    {"calloc",               SVC_CALLOC},
    {"realloc",              SVC_REALLOC},
    {"memcpy",               SVC_MEMCPY},
    {"memmove",              SVC_MEMMOVE},
    {"memset",               SVC_MEMSET},
    {"strlen",               SVC_STRLEN},
    {"strcpy",               SVC_STRCPY},
    {"strncpy",              SVC_STRNCPY},
    {"strcmp",               SVC_STRCMP},
    {"strncmp",              SVC_STRNCMP},
    {"strdup",               SVC_STRDUP},
    {"strndup",              SVC_STRNDUP},
    {"strcat",               SVC_STRCAT},
    {"strncat",              SVC_STRNCAT},
    {"abort",                SVC_ABORT},
    {"pthread_create",              SVC_PTHREAD_CREATE},
    {"pthread_join",                SVC_PTHREAD_JOIN},
    {"pthread_exit",                SVC_PTHREAD_EXIT},
    {"pthread_mutex_init",          SVC_PTHREAD_MUTEX_INIT},
    {"pthread_mutex_lock",          SVC_PTHREAD_MUTEX_LOCK},
    {"pthread_mutex_trylock",       SVC_PTHREAD_MUTEX_TRYLOCK},
    {"pthread_mutex_unlock",        SVC_PTHREAD_MUTEX_UNLOCK},
    {"pthread_mutex_destroy",       SVC_PTHREAD_MUTEX_DESTROY},
    {"pthread_mutexattr_init",      SVC_PTHREAD_MUTEXATTR_NOOP},
    {"pthread_mutexattr_settype",   SVC_PTHREAD_MUTEXATTR_NOOP},
    {"pthread_mutexattr_destroy",   SVC_PTHREAD_MUTEXATTR_NOOP},
    {"pthread_cond_init",           SVC_PTHREAD_COND_INIT},
    {"pthread_cond_wait",           SVC_PTHREAD_COND_WAIT},
    {"pthread_cond_timedwait",      SVC_PTHREAD_COND_TIMEDWAIT},
    {"pthread_cond_signal",         SVC_PTHREAD_COND_SIGNAL},
    {"pthread_cond_broadcast",      SVC_PTHREAD_COND_BROADCAST},
    {"pthread_cond_destroy",        SVC_PTHREAD_COND_DESTROY},
    {"pthread_condattr_init",       SVC_PTHREAD_CONDATTR_NOOP},
    {"pthread_condattr_setclock",   SVC_PTHREAD_CONDATTR_NOOP},
    {"pthread_condattr_destroy",    SVC_PTHREAD_CONDATTR_NOOP},
    {"pthread_attr_init",           SVC_PTHREAD_ATTR_NOOP},
    {"pthread_attr_setstacksize",   SVC_PTHREAD_ATTR_NOOP},
    {"pthread_attr_destroy",        SVC_PTHREAD_ATTR_NOOP},
    {"sem_init",                    SVC_SEM_INIT},
    {"sem_wait",                    SVC_SEM_WAIT},
    {"sem_trywait",                 SVC_SEM_TRYWAIT},
    {"sem_post",                    SVC_SEM_POST},
    {"sem_destroy",                 SVC_SEM_DESTROY},
    
    {"ANativeWindow_fromSurface",        SVC_ANW_FROM_SURFACE},
    {"ANativeWindow_acquire",            SVC_ANW_ACQUIRE},
    {"ANativeWindow_release",            SVC_ANW_RELEASE},
    {"ANativeWindow_getWidth",           SVC_ANW_GETWIDTH},
    {"ANativeWindow_getHeight",          SVC_ANW_GETHEIGHT},
    {"ANativeWindow_setBuffersGeometry", SVC_ANW_SETBUFGEO},
    {"ANativeWindow_toSurface",          SVC_ANW_TOSURFACE},
    
    {"eglGetDisplay",         SVC_EGL_GETDISPLAY},
    {"eglInitialize",         SVC_EGL_INITIALIZE},
    {"eglChooseConfig",       SVC_EGL_CHOOSECONFIG},
    {"eglCreateWindowSurface",SVC_EGL_CREATEWSURF},
    {"eglCreatePbufferSurface",SVC_EGL_CREATEPBUF},
    {"eglCreateContext",      SVC_EGL_CREATECTX},
    {"eglMakeCurrent",        SVC_EGL_MAKECURRENT},
    {"eglSwapBuffers",        SVC_EGL_SWAPBUF},
    {"eglDestroySurface",     SVC_EGL_DESTROYSURF},
    {"eglDestroyContext",     SVC_EGL_DESTROYCTX},
    {"eglTerminate",          SVC_EGL_TERMINATE},
    {"eglGetProcAddress",     SVC_EGL_GETPROC},
    {"eglQuerySurface",       SVC_EGL_QUERYSURF},
    {"eglGetError",           SVC_EGL_GETERROR},
    {"eglGetConfigAttrib",    SVC_EGL_GETCFGATTRIB},
    {"eglQueryString",        SVC_EGL_QUERYSTR},
    {"eglSurfaceAttrib",      SVC_EGL_SURFACEATTRIB},
    {"eglSwapInterval",           SVC_EGL_SWAPINTERVAL},
    {"eglGetCurrentContext",      SVC_EGL_GETCURCTX},
    {"eglGetCurrentSurface",      SVC_EGL_GETCURSURF},
    
    {"eglGetSystemTimeFrequencyNV", SVC_EGL_SYSTIME_FREQ},
    {"eglGetSystemTimeNV",          SVC_EGL_SYSTIME},
    
    {"dl_unwind_find_exidx",  SVC_DL_UNWIND_EXIDX},
    
    {"__aeabi_uidiv",         SVC_AEABI_UIDIV},
    {"__aeabi_uidivmod",      SVC_AEABI_UIDIVMOD},
    {"__aeabi_idiv",          SVC_AEABI_IDIV},
    {"__aeabi_idivmod",       SVC_AEABI_IDIV}, /* shared with SVC_AEABI_IDIV handler (sets quotient r0 and remainder r1) */
    {"__aeabi_ldivmod",       SVC_AEABI_LDIVMOD},
    {"__aeabi_uldivmod",      SVC_AEABI_ULDIVMOD},
    
    {"open",                  SVC_LIBC_OPEN},
    {"open64",                SVC_LIBC_OPEN},
    {"close",                 SVC_LIBC_CLOSE},
    {"read",                  SVC_LIBC_READ},
    {"write",                 SVC_LIBC_WRITE},
    {"lseek",                 SVC_LIBC_LSEEK},
    {"lseek64",               SVC_LIBC_LSEEK64},
    {"fopen",                 SVC_LIBC_FOPEN},
    {"fopen64",               SVC_LIBC_FOPEN},
    {"fclose",                SVC_LIBC_FCLOSE},
    {"fread",                 SVC_LIBC_FREAD},
    {"fwrite",                SVC_LIBC_FWRITE},
    {"fseek",                 SVC_LIBC_FSEEK},
    {"ftell",                 SVC_LIBC_FTELL},
    {"stat",                  SVC_LIBC_STAT},
    {"fstat",                 SVC_LIBC_FSTAT},
    {"stat64",                SVC_LIBC_STAT},
    {"fstat64",               SVC_LIBC_FSTAT},
    {"statfs",                SVC_STATFS},
    {"statfs64",              SVC_STATFS},
    {"fstatfs",               SVC_STATFS},
    {"fstatfs64",             SVC_STATFS},
    {"statvfs",               SVC_STATVFS},
    {"statvfs64",             SVC_STATVFS},
    {"fstatvfs",              SVC_STATVFS},
    {"fstatvfs64",            SVC_STATVFS},
    
    {"glViewport",              SVC_GL_Viewport},
    {"glClear",                 SVC_GL_Clear},
    {"glClearColor",            SVC_GL_ClearColor},
    {"glClearDepthf",           SVC_GL_ClearDepthf},
    {"glClearStencil",          SVC_GL_ClearStencil},
    {"glEnable",                SVC_GL_Enable},
    {"glDisable",               SVC_GL_Disable},
    {"glDepthFunc",             SVC_GL_DepthFunc},
    {"glDepthMask",             SVC_GL_DepthMask},
    {"glColorMask",             SVC_GL_ColorMask},
    {"glScissor",               SVC_GL_Scissor},
    {"glFrontFace",             SVC_GL_FrontFace},
    {"glCullFace",              SVC_GL_CullFace},
    {"glBlendFuncSeparate",     SVC_GL_BlendFuncSeparate},
    {"glBlendEquationSeparate", SVC_GL_BlendEquationSeparate},
    {"glGetError",              SVC_GL_GetError},
    {"glGetString",             SVC_GL_GetString},
    {"glGetIntegerv",           SVC_GL_GetIntegerv},
    {"glPixelStorei",           SVC_GL_PixelStorei},
    {"glReadPixels",            SVC_GL_ReadPixels},
    {"glFlush",                 SVC_GL_Flush},
    {"glFinish",                SVC_GL_Finish},
    {"glGenBuffers",            SVC_GL_GenBuffers},
    {"glBindBuffer",            SVC_GL_BindBuffer},
    {"glBufferData",            SVC_GL_BufferData},
    {"glBufferSubData",         SVC_GL_BufferSubData},
    {"glDeleteBuffers",         SVC_GL_DeleteBuffers},
    {"glGenTextures",           SVC_GL_GenTextures},
    {"glBindTexture",           SVC_GL_BindTexture},
    {"glActiveTexture",         SVC_GL_ActiveTexture},
    {"glDeleteTextures",        SVC_GL_DeleteTextures},
    {"glTexParameteri",         SVC_GL_TexParameteri},
    {"glTexImage2D",            SVC_GL_TexImage2D},
    {"glTexSubImage2D",         SVC_GL_TexSubImage2D},
    {"glCopyTexSubImage2D",     SVC_GL_CopyTexSubImage2D},
    {"glCompressedTexImage2D",  SVC_GL_CompressedTexImage2D},
    {"glCompressedTexSubImage2D", SVC_GL_CompressedTexSubImage2D},
    {"glGenerateMipmap",        SVC_GL_GenerateMipmap},
    {"glGenFramebuffers",       SVC_GL_GenFramebuffers},
    {"glBindFramebuffer",       SVC_GL_BindFramebuffer},
    {"glDeleteFramebuffers",    SVC_GL_DeleteFramebuffers},
    {"glCheckFramebufferStatus",SVC_GL_CheckFramebufferStatus},
    {"glFramebufferTexture2D",  SVC_GL_FramebufferTexture2D},
    {"glFramebufferRenderbuffer",SVC_GL_FramebufferRenderbuffer},
    {"glGetFramebufferAttachmentParameteriv", SVC_GL_GetFramebufferAttachmentParameteriv},
    {"glGenRenderbuffers",      SVC_GL_GenRenderbuffers},
    {"glBindRenderbuffer",      SVC_GL_BindRenderbuffer},
    {"glDeleteRenderbuffers",   SVC_GL_DeleteRenderbuffers},
    {"glRenderbufferStorage",   SVC_GL_RenderbufferStorage},
    {"glCreateShader",          SVC_GL_CreateShader},
    {"glShaderSource",          SVC_GL_ShaderSource},
    {"glCompileShader",         SVC_GL_CompileShader},
    {"glDeleteShader",          SVC_GL_DeleteShader},
    {"glGetShaderiv",           SVC_GL_GetShaderiv},
    {"glGetShaderInfoLog",      SVC_GL_GetShaderInfoLog},
    {"glGetShaderSource",       SVC_GL_GetShaderSource},
    {"glCreateProgram",         SVC_GL_CreateProgram},
    {"glAttachShader",          SVC_GL_AttachShader},
    {"glLinkProgram",           SVC_GL_LinkProgram},
    {"glUseProgram",            SVC_GL_UseProgram},
    {"glDeleteProgram",         SVC_GL_DeleteProgram},
    {"glGetProgramiv",          SVC_GL_GetProgramiv},
    {"glGetProgramInfoLog",     SVC_GL_GetProgramInfoLog},
    {"glGetAttribLocation",     SVC_GL_GetAttribLocation},
    {"glGetUniformLocation",    SVC_GL_GetUniformLocation},
    {"glGetActiveAttrib",       SVC_GL_GetActiveAttrib},
    {"glGetActiveUniform",      SVC_GL_GetActiveUniform},
    {"glBindAttribLocation",    SVC_GL_BindAttribLocation},
    {"glUniform1i",             SVC_GL_Uniform1i},
    {"glUniform1iv",            SVC_GL_Uniform1iv},
    {"glUniform2iv",            SVC_GL_Uniform2iv},
    {"glUniform3iv",            SVC_GL_Uniform3iv},
    {"glUniform4iv",            SVC_GL_Uniform4iv},
    {"glUniform1fv",            SVC_GL_Uniform1fv},
    {"glUniform2fv",            SVC_GL_Uniform2fv},
    {"glUniform3fv",            SVC_GL_Uniform3fv},
    {"glUniform4fv",            SVC_GL_Uniform4fv},
    {"glUniformMatrix3fv",      SVC_GL_UniformMatrix3fv},
    {"glUniformMatrix4fv",      SVC_GL_UniformMatrix4fv},
    {"glEnableVertexAttribArray", SVC_GL_EnableVertexAttribArray},
    {"glDisableVertexAttribArray",SVC_GL_DisableVertexAttribArray},
    {"glVertexAttribPointer",   SVC_GL_VertexAttribPointer},
    {"glGetVertexAttribiv",     SVC_GL_GetVertexAttribiv},
    {"glGetVertexAttribPointerv",SVC_GL_GetVertexAttribPointerv},
    {"glDrawArrays",            SVC_GL_DrawArrays},
    {"glDrawElements",          SVC_GL_DrawElements},
    {"glStencilFunc",           SVC_GL_StencilFunc},
    {"glStencilFuncSeparate",   SVC_GL_StencilFuncSeparate},
    {"glStencilMask",           SVC_GL_StencilMask},
    {"glStencilOp",             SVC_GL_StencilOp},
    {"glStencilOpSeparate",     SVC_GL_StencilOpSeparate},
    {"glBlendFunc",             SVC_GL_BlendFunc},
    {"glTexParameterf",         SVC_GL_TexParameterf},
    {"glDepthRangef",           SVC_GL_DepthRangef},
    {"glPolygonOffset",         SVC_GL_PolygonOffset},
    {"glLineWidth",             SVC_GL_LineWidth},
    {"glSampleCoverage",        SVC_GL_SampleCoverage},
    
    {"glUniform1f",             SVC_GL_Uniform1f},
    {"glUniform2f",             SVC_GL_Uniform2f},
    {"glUniform3f",             SVC_GL_Uniform3f},
    {"glUniform4f",             SVC_GL_Uniform4f},
    
    {"glVertexAttrib1f",        SVC_GL_VertexAttrib1f},
    {"glVertexAttrib2f",        SVC_GL_VertexAttrib2f},
    {"glVertexAttrib3f",        SVC_GL_VertexAttrib3f},
    {"glVertexAttrib4f",        SVC_GL_VertexAttrib4f},
    {"glVertexAttrib4fv",       SVC_GL_VertexAttrib4fv},
    
    {"glGetFloatv",             SVC_GL_GetFloatv},
    {"glGetBooleanv",           SVC_GL_GetBooleanv},
    {"glIsEnabled",             SVC_GL_IsEnabled},
    {"glIsProgram",             SVC_GL_IsProgram},
    {"glIsShader",              SVC_GL_IsShader},
    {"glIsTexture",             SVC_GL_IsTexture},
    {"glIsBuffer",              SVC_GL_IsBuffer},
    {"glIsFramebuffer",         SVC_GL_IsFramebuffer},
    {"glIsRenderbuffer",        SVC_GL_IsRenderbuffer},
    
    {"glBlendEquation",         SVC_GL_BlendEquation},
    {"glBlendColor",            SVC_GL_BlendColor},
    {"glReleaseShaderCompiler", SVC_GL_ReleaseShaderCompiler},
    {"glGetShaderPrecisionFormat", SVC_GL_GetShaderPrecisionFormat},
    {"glUniformMatrix2fv",      SVC_GL_UniformMatrix2fv},
    
    {"glVertexAttrib1fv",       SVC_GL_VertexAttrib1fv},
    {"glVertexAttrib2fv",       SVC_GL_VertexAttrib2fv},
    {"glVertexAttrib3fv",       SVC_GL_VertexAttrib3fv},

    {"glGetStringi",             SVC_GL3_GetStringi},
    {"glGetIntegeri_v",          SVC_GL3_GetIntegeri_v},
    {"glGetInternalformativ",    SVC_GL3_GetInternalformativ},
    {"glGetProgramInterfaceiv",  SVC_GL3_GetProgramInterfaceiv},
    {"glGetProgramResourceiv",   SVC_GL3_GetProgramResourceiv},
    {"glGetProgramResourceName", SVC_GL3_GetProgramResourceName},
    {"glGenVertexArrays",        SVC_GL3_GenVertexArrays},
    {"glBindVertexArray",        SVC_GL3_BindVertexArray},
    {"glDeleteVertexArrays",     SVC_GL3_DeleteVertexArrays},
    {"glIsVertexArray",          SVC_GL3_IsVertexArray},
    {"glBindSampler",            SVC_GL3_BindSampler},
    {"glBindBufferBase",         SVC_GL3_BindBufferBase},
    {"glBindBufferRange",        SVC_GL3_BindBufferRange},
    {"glMapBufferRange",         SVC_GL3_MapBufferRange},
    {"glUnmapBuffer",            SVC_GL3_UnmapBuffer},
    {"glFlushMappedBufferRange", SVC_GL3_FlushMappedBufferRange},
    {"glTexStorage2D",           SVC_GL3_TexStorage2D},
    {"glTexStorage3D",           SVC_GL3_TexStorage3D},
    {"glTexSubImage3D",          SVC_GL3_TexSubImage3D},
    {"glProgramParameteri",      SVC_GL3_ProgramParameteri},
    {"glGetProgramBinary",       SVC_GL3_GetProgramBinary},
    {"glProgramBinary",          SVC_GL3_ProgramBinary},
    {"glFenceSync",              SVC_GL3_FenceSync},
    {"glClientWaitSync",         SVC_GL3_ClientWaitSync},
    {"glDeleteSync",             SVC_GL3_DeleteSync},
    {"glInvalidateFramebuffer",  SVC_GL3_InvalidateFramebuffer},
    {"glDetachShader",           SVC_GL3_DetachShader},
    {"glDrawBuffers",            SVC_GL3_DrawBuffers},

    {"clock_gettime",           SVC_CLOCK_GETTIME},
    {"gettimeofday",            SVC_GETTIMEOFDAY},
    {"time",                    SVC_TIME},
    {"nanosleep",               SVC_NANOSLEEP},
    {"usleep",                  SVC_USLEEP},
    {"getenv",                  SVC_GETENV},
    {"getpid",                  SVC_GETPID},
    {"gettid",                  SVC_GETTID},
    {"sched_yield",             SVC_SCHED_YIELD},
    {"getpagesize",             SVC_GETPAGESIZE},
    {"sysconf",                 SVC_SYSCONF},
    {"mprotect",                SVC_RET0},
    {"madvise",                 SVC_RET0},
    {"msync",                   SVC_RET0},
    {"mlock",                   SVC_RET0},
    {"munlock",                 SVC_RET0},
    {"prctl",                   SVC_RET0},
    {"setenv",                  SVC_RET0},
    {"sigaction",               SVC_SIGACTION},
    {"sigemptyset",             SVC_RET0},
    {"sigaddset",               SVC_RET0},
    {"sigprocmask",             SVC_RET0},
    {"pthread_sigmask",         SVC_RET0},
    {"setpriority",             SVC_RET0},
    {"getpriority",             SVC_RET0},
    {"sched_setaffinity",       SVC_RET0},
    {"sched_getaffinity",       SVC_RET0},
    {"__errno",                 SVC_ERRNO_ADDR},
    {"__system_property_get",   SVC_SYSPROP_GET},
    {"__system_property_find",  SVC_RET0},
    /* pthread TLS (real) */
    {"pthread_self",            SVC_PTHREAD_SELF},
    {"pthread_key_create",      SVC_PTHREAD_KEY_CREATE},
    {"pthread_key_delete",      SVC_PTHREAD_KEY_DELETE},
    {"pthread_setspecific",     SVC_PTHREAD_SETSPECIFIC},
    {"pthread_getspecific",     SVC_PTHREAD_GETSPECIFIC},
    
    {"inflateInit2_",           SVC_Z_INFLATEINIT2},
    {"inflate",                 SVC_Z_INFLATE},
    {"inflateEnd",              SVC_Z_INFLATEEND},
    {"inflateReset",            SVC_Z_INFLATERESET},
    {"crc32",                   SVC_Z_CRC32},
    {"adler32",                 SVC_Z_ADLER32},
    
    {"atoi",                    SVC_ATOI},
    {"atol",                    SVC_ATOL},
    {"strtol",                  SVC_STRTOL},
    {"strtoul",                 SVC_STRTOUL},
    {"strtod",                  SVC_STRTOD},
    {"strtof",                  SVC_STRTOF},
    {"strtoll",                 SVC_STRTOLL},
    {"strtoull",                SVC_STRTOULL},
    
    {"mmap",                    SVC_LIBC_MMAP},
    {"munmap",                  SVC_LIBC_MUNMAP},
    {"memalign",                SVC_MEMALIGN},
    {"posix_memalign",          SVC_POSIX_MEMALIGN},
    
    {"memcmp",                  SVC_MEMCMP},
    {"memchr",                  SVC_MEMCHR},
    {"memrchr",                 SVC_MEMRCHR},
    {"memmem",                  SVC_MEMMEM},
    {"strchr",                  SVC_STRCHR},
    {"strrchr",                 SVC_STRRCHR},
    {"strstr",                  SVC_STRSTR},
    {"strnlen",                 SVC_STRNLEN},
    {"strcasecmp",              SVC_STRCASECMP},
    {"strcspn",                 SVC_STRCSPN},
    {"strspn",                  SVC_STRSPN},
    {"strtok_r",                SVC_STRTOK_R},
    {"basename",                SVC_BASENAME},
    {"isspace",                 SVC_ISSPACE},
    {"wcslen",                  SVC_WCSLEN},
    {"wmemcpy",                 SVC_WMEMCPY},
    {"wmemmove",                SVC_WMEMMOVE},
    {"wmemset",                 SVC_WMEMSET},
    
    {"snprintf",                SVC_SNPRINTF},
    {"sprintf",                 SVC_SPRINTF},
    {"vsnprintf",               SVC_VSNPRINTF},
    {"vasprintf",               SVC_VASPRINTF},
    {"printf",                  SVC_PRINTF},
    {"fprintf",                 SVC_FPRINTF},
    {"vprintf",                 SVC_VPRINTF},
    {"vfprintf",                SVC_VFPRINTF},
    {"puts",                    SVC_PUTS},
    {"fputs",                   SVC_FPUTS},
    {"fputc",                   SVC_FPUTC},
    {"__assert2",               SVC_ASSERT2},
    {"__android_log_vprint",    SVC_LOG_VPRINT},
    {"__android_log_buf_write", SVC_LOG_WRITE},
    
    {"sincos",                  SVC_SINCOS},
    {"sincosf",                 SVC_SINCOSF},
    {"ldexp",                   SVC_LDEXP},
    {"ldexpf",                  SVC_LDEXPF},
    {"modf",                    SVC_MODF},
    {"modff",                   SVC_MODFF},
    
    {"access",                  SVC_ACCESS},
    {"realpath",                SVC_REALPATH},
    {"pread",                   SVC_PREAD},
    {"pwrite",                  SVC_PWRITE},
    {"lstat",                   SVC_LIBC_STAT},
    {"fseeko",                  SVC_LIBC_FSEEK},
    {"ftello",                  SVC_LIBC_FTELL},
    {"fgets",                   SVC_FGETS},
    {"fileno",                  SVC_FILENO},
    {"feof",                    SVC_FEOF},
    {"opendir",                 SVC_OPENDIR},
    {"readdir",                 SVC_READDIR},
    {"closedir",                SVC_CLOSEDIR},
    
    {"pthread_equal",                SVC_PTHREAD_EQUAL},
    {"pthread_detach",               SVC_PTHREAD_DETACH},
    {"pthread_attr_setdetachstate",  SVC_PTHREAD_ATTR_NOOP},
    {"pthread_setname_np",           SVC_PTHREAD_ATTR_NOOP},
    {"pthread_rwlock_init",          SVC_PTHREAD_RWLOCK_INIT},
    {"pthread_rwlock_rdlock",        SVC_PTHREAD_RWLOCK_RDLOCK},
    {"pthread_rwlock_tryrdlock",     SVC_PTHREAD_RWLOCK_TRYRDLOCK},
    {"pthread_rwlock_wrlock",        SVC_PTHREAD_RWLOCK_WRLOCK},
    {"pthread_rwlock_trywrlock",     SVC_PTHREAD_RWLOCK_TRYWRLOCK},
    {"pthread_rwlock_unlock",        SVC_PTHREAD_RWLOCK_UNLOCK},
    {"pthread_rwlock_destroy",       SVC_PTHREAD_RWLOCK_DESTROY},
    
    {"fflush",                  SVC_RET0},
    {"clearerr",                SVC_RET0},
    {"setbuf",                  SVC_RET0},
    {"setvbuf",                 SVC_RET0},
    {"fsync",                   SVC_RET0},
    {"flock",                   SVC_RET0},
    {"exit",                    SVC_EXIT},
    {"_exit",                   SVC_EXIT},
    
    {"__aeabi_memcpy",          SVC_MEMCPY},
    {"__aeabi_memcpy4",         SVC_MEMCPY},
    {"__aeabi_memcpy8",         SVC_MEMCPY},
    {"__aeabi_memmove",         SVC_MEMMOVE},
    {"__aeabi_memset",          SVC_AEABI_MEMSET},
    {"__aeabi_memset4",         SVC_AEABI_MEMSET},
    {"__aeabi_memset8",         SVC_AEABI_MEMSET},
    {"__aeabi_memclr",          SVC_AEABI_MEMCLR},
    {"__aeabi_memclr4",         SVC_AEABI_MEMCLR},
    {"__aeabi_memclr8",         SVC_AEABI_MEMCLR},
    
    {"__memcpy_chk",            SVC_MEMCPY},
    {"__memmove_chk",           SVC_MEMMOVE},
    {"__memset_chk",            SVC_MEMSET},
    {"__strcpy_chk",            SVC_STRCPY},
    {"__strncpy_chk",           SVC_STRNCPY},
    {"__strcat_chk",            SVC_STRCAT},
    {"__strncat_chk",           SVC_STRNCAT},
    {"__strlen_chk",            SVC_STRLEN},
    {"__strchr_chk",            SVC_STRCHR},
    {"__strrchr_chk",           SVC_STRRCHR},
    {"__strlcpy_chk",           SVC_STRLCPY},
    {"__vsnprintf_chk",         SVC_VSNPRINTF_CHK},
    {"__vsprintf_chk",          SVC_VSPRINTF_CHK},
    {"__open_2",                SVC_LIBC_OPEN},
    {"__mmap2",                 SVC_LIBC_MMAP2},
    {"__FD_SET_chk",            SVC_RET0},
    {"__FD_ISSET_chk",          SVC_RET0},
    {"__pthread_cleanup_push",  SVC_RET0},
    {"__pthread_cleanup_pop",   SVC_RET0},
    {"__stack_chk_fail",        SVC_ABORT},
    {"__cxa_atexit",            SVC_RET0},
    {"__cxa_finalize",          SVC_RET0},
    {"__system_property_read",  SVC_RET0},
    {"__gnu_Unwind_Find_exidx", SVC_DL_UNWIND_EXIDX},
    
    {"tolower",                 SVC_TOLOWER},
    {"isalpha",                 SVC_ISALPHA},
    {"isdigit",                 SVC_ISDIGIT},
    {"isalnum",                 SVC_ISALNUM},
    {"isxdigit",                SVC_ISXDIGIT},
    {"isascii",                 SVC_ISASCII},
    {"strlcpy",                 SVC_STRLCPY},
    {"strncasecmp",             SVC_STRNCASECMP},
    {"strerror",                SVC_STRERROR},
    {"strerror_r",              SVC_STRERROR_R},
    {"__xpg_strerror_r",        SVC_STRERROR_R},
    {"setlocale",               SVC_SETLOCALE},
    {"newlocale",               SVC_RET0},
    {"uselocale",               SVC_RET0},
    {"freelocale",              SVC_RET0},
    {"strtoll_l",               SVC_STRTOLL},
    {"strtoull_l",              SVC_STRTOULL},
    {"abs",                     SVC_ABS},
    {"sscanf",                  SVC_SSCANF},
    {"vsscanf",                 SVC_VSSCANF},
    
    {"mkdir",                   SVC_MKDIR},
    {"getcwd",                  SVC_GETCWD},
    {"unlink",                  SVC_UNLINK},
    {"remove",                  SVC_UNLINK},
    {"rename",                  SVC_RENAME},
    {"ftruncate",               SVC_FTRUNCATE},
    {"ftruncate64",             SVC_FTRUNCATE},
    {"readlink",                SVC_READLINK},
    {"writev",                  SVC_WRITEV},
    {"rmdir",                   SVC_RETM1},
    {"chmod",                   SVC_RET0},
    {"fchmod",                  SVC_RET0},
    {"chdir",                   SVC_CHDIR},
    {"isatty",                  SVC_RET0},
    {"dladdr",                  SVC_RET0},
    {"dlerror",                 SVC_RET0},
    
    {"clock",                   SVC_CLOCK},
    {"localtime_r",             SVC_LOCALTIME_R},
    {"gmtime_r",                SVC_GMTIME_R},
    {"localtime",               SVC_LOCALTIME},
    {"gmtime",                  SVC_GMTIME},
    {"mktime",                  SVC_MKTIME},
    {"difftime",                SVC_DIFFTIME},
    {"strftime",                SVC_STRFTIME},
    
    {"uname",                   SVC_UNAME},
    {"getrlimit",               SVC_GETRLIMIT},
    {"getdtablesize",           SVC_GETDTABLESIZE},
    {"mremap",                  SVC_MREMAP},
    {"pthread_getattr_np",      SVC_PTHREAD_GETATTR_NP},
    {"pthread_attr_getstack",   SVC_PTHREAD_ATTR_GETSTACK},
    {"pthread_attr_getstacksize", SVC_PTHREAD_ATTR_GETSTACKSZ},
    {"pthread_attr_getdetachstate", SVC_RET0},
    {"pthread_getschedparam",   SVC_RET0},
    {"pthread_setschedparam",   SVC_RET0},
    {"pthread_kill",            SVC_PTHREAD_KILL},
    {"sched_get_priority_max",  SVC_RET0},
    {"sched_get_priority_min",  SVC_RET0},
    {"sched_getcpu",            SVC_RET0},
    {"sem_getvalue",            SVC_SEM_GETVALUE},
    {"sem_timedwait",           SVC_SEM_TIMEDWAIT},
    {"sleep",                   SVC_USLEEP},
    {"getuid",                  SVC_RET0},
    {"geteuid",                 SVC_RET0},
    {"getegid",                 SVC_RET0},
    {"raise",                   SVC_RET0},
    {"kill",                    SVC_PTHREAD_KILL},
    {"bsd_signal",              SVC_BSD_SIGNAL},
    {"sigaltstack",             SVC_RET0},
    {"syscall",                 SVC_SYSCALL},
    {"sbrk",                   SVC_SBRK},
    {"brk",                    SVC_BRK},
    
    {"ALooper_forThread",           SVC_ALOOPER_FORTHREAD},
    {"ALooper_prepare",             SVC_ALOOPER_PREPARE},
    {"ALooper_pollOnce",            SVC_ALOOPER_POLLONCE},
    {"ALooper_pollAll",             SVC_ALOOPER_POLLALL},
    {"ALooper_wake",                SVC_ALOOPER_WAKE},
    {"ALooper_addFd",               SVC_ALOOPER_ADDFD}, /* returns 1 on success */
    {"ALooper_removeFd",            SVC_ALOOPER_ADDFD}, /* same contract: 1 on success */
    /* AAudio: stateful stubs; the data callback is pumped from idle points */
    {"AAudio_createStreamBuilder",              SVC_AAUDIO_CREATE_BUILDER},
    {"AAudioStreamBuilder_setDirection",        SVC_AAUDIO_SET_DIRECTION},
    {"AAudioStreamBuilder_setSampleRate",       SVC_AAUDIO_SET_RATE},
    {"AAudioStreamBuilder_setChannelCount",     SVC_AAUDIO_SET_CHANNELS},
    {"AAudioStreamBuilder_setFormat",           SVC_AAUDIO_SET_FORMAT},
    {"AAudioStreamBuilder_setSharingMode",      SVC_RET0},
    {"AAudioStreamBuilder_setPerformanceMode",  SVC_RET0},
    {"AAudioStreamBuilder_setBufferCapacityInFrames", SVC_RET0},
    {"AAudioStreamBuilder_setFramesPerDataCallback",  SVC_RET0},
    {"AAudioStreamBuilder_setDataCallback",     SVC_AAUDIO_SET_DATA_CB},
    {"AAudioStreamBuilder_setErrorCallback",    SVC_RET0},
    {"AAudioStreamBuilder_setDeviceId",         SVC_RET0},
    {"AAudioStreamBuilder_openStream",          SVC_AAUDIO_OPEN_STREAM},
    {"AAudioStreamBuilder_delete",              SVC_RET0},
    {"AAudioStream_requestStart",               SVC_AAUDIO_START},
    {"AAudioStream_requestStop",                SVC_AAUDIO_STOP},
    {"AAudioStream_close",                      SVC_AAUDIO_STOP},
    {"AAudioStream_getFramesPerBurst",          SVC_AAUDIO_GET_FPB},
    {"AAudioStream_getBufferSizeInFrames",      SVC_AAUDIO_GET_BUFSIZE},
    {"AAudioStream_setBufferSizeInFrames",      SVC_AAUDIO_SET_BUFSIZE},
    {"AAudioStream_getBufferCapacityInFrames",  SVC_AAUDIO_GET_BUFCAP},
    {"AAudioStream_getXRunCount",               SVC_RET0},
    {"AAudioStream_getDeviceId",                SVC_RET0},
    {"AAudioStream_isMMapUsed",                 SVC_RET0},
    {"AAudioStream_waitForStateChange",         SVC_AAUDIO_WAIT_STATE},
    {"getauxval",                               SVC_GETAUXVAL},
    /* Swappy presentation timing: accept (EGL_TRUE) without acting on it */
    {"eglPresentationTimeANDROID",              SVC_ALOOPER_ADDFD},
    {"ALooper_acquire",             SVC_RET0},
    {"ALooper_release",             SVC_RET0},
    {"setjmp",                  SVC_SETJMP},
    {"_setjmp",                 SVC_SETJMP},
    {"sigsetjmp",               SVC_SETJMP},
    {"longjmp",                 SVC_LONGJMP},
    {"_longjmp",                SVC_LONGJMP},
    {"siglongjmp",              SVC_LONGJMP},
    {"perror",                  SVC_RET0},
    {"openlog",                 SVC_RET0},
    {"closelog",                SVC_RET0},
    {"syslog",                  SVC_RET0},
    {"qsort",                   SVC_QSORT},
    {"bsearch",                 SVC_BSEARCH},/* invoke guest comparator via callback JIT */
    /* C++ one-time static-init guards; missing impl reruns engine init each frame.
 * __cxa_atexit/atexit only need successful registration. */
    {"__cxa_guard_acquire",     SVC_CXA_GUARD_ACQUIRE},
    {"__cxa_guard_release",     SVC_CXA_GUARD_RELEASE},
    {"__cxa_guard_abort",       SVC_CXA_GUARD_ABORT},
    {"__cxa_pure_virtual",      SVC_CXA_PURE_VIRTUAL},
    {"__aeabi_atexit",          SVC_AEABI_ATEXIT},
    
    {"btowc",                   SVC_BTOWC},
    {"wctob",                   SVC_WCTOB},
    {"towlower",                SVC_TOWLOWER},
    {"towupper",                SVC_TOWUPPER},
    {"iswctype",                SVC_ISWCTYPE},
    {"wctype",                  SVC_WCTYPE},
    {"mbrtowc",                 SVC_MBRTOWC},
    {"wcrtomb",                 SVC_WCRTOMB},
    {"wmemchr",                 SVC_WMEMCHR},
    {"strcoll",                 SVC_STRCOLL},
    {"strxfrm",                 SVC_STRXFRM},
    {"strcasestr",              SVC_STRCASESTR},
    {"strsep",                  SVC_STRSEP},
    /* fp classification, fenv, wide-char classification/conversion */
    {"isnan",                   SVC_ISNAN},
    {"__isnan",                 SVC_ISNAN},
    {"isnanf",                  SVC_ISNAN},  /* float variant: r0 only */
    {"isinf",                   SVC_ISINF},
    {"__isinf",                 SVC_ISINF},
    {"isinff",                  SVC_ISINF},
    {"isfinite",                SVC_ISFINITE},
    {"__isfinite",              SVC_ISFINITE},
    {"isfinitef",               SVC_ISFINITE},
    {"finite",                  SVC_ISFINITE},  /* deprecated BSD alias */
    {"finitef",                 SVC_ISFINITE},
    {"signbit",                 SVC_SIGNBIT},
    {"__signbit",               SVC_SIGNBIT},
    {"signbitf",                SVC_SIGNBIT},
    
    {"fegetround",              SVC_FEGETROUND},
    {"fesetround",              SVC_FESETROUND},
    {"feclearexcept",           SVC_FECLEAREXCEPT},
    {"feraiseexcept",           SVC_FERAISEEXCEPT},
    {"fetestexcept",            SVC_FETESTEXCEPT},
    {"fegetenv",                SVC_RET0},
    {"fesetenv",                SVC_RET0},
    {"feholdexcept",            SVC_RET0},
    {"feupdateenv",             SVC_RET0},
    
    {"iswspace",                SVC_ISWSPACE},
    {"iswdigit",                SVC_ISWDIGIT},
    {"iswalpha",                SVC_ISWALPHA},
    {"iswupper",                SVC_ISWUPPER},
    {"iswlower",                SVC_ISWLOWER},
    {"iswprint",                SVC_ISWPRINT},
    {"iswpunct",                SVC_ISWPUNCT},
    {"iswgraph",                SVC_ISWGRAPH},
    {"iswalnum",                SVC_ISWALNUM},
    {"iswblank",                SVC_ISWBLANK},
    {"iswcntrl",                SVC_ISWCNTRL},
    {"iswxdigit",               SVC_ISWDIGIT},  /* hex digit: delegate to iswdigit-like */
    {"wctrans",                 SVC_WCTRANS},
    {"towctrans",               SVC_TOWCTRANS},
    
    {"strtold",                 SVC_STRTOLD},
    {"wcstod",                  SVC_WCSTOD},
    {"wcstof",                  SVC_WCSTOD},    /* float variant: same path as double */
    {"wcstol",                  SVC_WCSTOL},
    {"wcstoul",                 SVC_WCSTOUL},
    {"wcstoll",                 SVC_WCSTOLL},
    {"wcstoull",                SVC_WCSTOULL},
    {"strtoimax",               SVC_STRTOIMAX},
    {"strtoumax",               SVC_STRTOUMAX},
    {"wcslen",                  SVC_WCSLEN},
    {"wcsnlen",                 SVC_WCSLEN},
    {"wcscmp",                  SVC_WCSCMP},
    {"wcsncmp",                 SVC_WCSNCMP},
    {"wcscpy",                  SVC_WCSCPY},
    {"wcsncpy",                 SVC_WCSCPY},
    {"wcscat",                  SVC_WCSCAT},
    {"wcsncat",                 SVC_WCSCAT},
    
    {"div",                     SVC_DIV},
    {"ldiv",                    SVC_LDIV},
    {"lldiv",                   SVC_RET0},  /* 64-bit variant: stub 0 due to complex ABI */
    /* Process/socket stubs return -1 instead of fake success (0) */
    {"fork",                    SVC_RETM1},
    {"execl",                   SVC_RETM1},
    {"execv",                   SVC_RETM1},
    {"execve",                  SVC_RETM1},
    {"execvp",                  SVC_RETM1},
    {"waitpid",                 SVC_RETM1},
    {"system",                  SVC_RETM1},
    {"popen",                   SVC_RETM1},
    {"pclose",                  SVC_RETM1},
    {"socket",                  SVC_RETM1},
    {"connect",                 SVC_RETM1},
    {"bind",                    SVC_RETM1},
    {"listen",                  SVC_RETM1},
    {"accept",                  SVC_RETM1},
    {"pipe",                    SVC_PIPE},
    {"pipe2",                   SVC_PIPE2},
    {"epoll_create",            SVC_RETM1},
    {"inotify_init",            SVC_RETM1},
    {"mkstemp",                 SVC_RETM1},
    {"mkstemps",                SVC_RETM1},
    {"mkdtemp",                 SVC_RETM1},
    {"getaddrinfo",             SVC_RETM1},
    {"gethostbyname",           SVC_RET0},
    
    
    {"data_start",              SVC_RET0},
    {"__sF",                    SVC_RET0},
    {"_ctype_",                 SVC_RET0},
    {"environ",                 SVC_RET0},
    {"__stack_chk_guard",       SVC_RET0},
    
    {"_Unwind_Complete",             SVC_RET0},
    {"_Unwind_DeleteException",      SVC_RET0},
    {"_Unwind_GetDataRelBase",       SVC_RET0},
    {"_Unwind_GetLanguageSpecificData", SVC_RET0},
    {"_Unwind_GetRegionStart",       SVC_RET0},
    {"_Unwind_GetTextRelBase",       SVC_RET0},
    {"_Unwind_RaiseException",       SVC_RET0},
    {"_Unwind_Resume",               SVC_RET0},
    {"_Unwind_Resume_or_Rethrow",    SVC_RET0},
    {"_Unwind_VRS_Get",              SVC_RET0},
    {"_Unwind_VRS_Set",              SVC_RET0},
    {"__gnu_unwind_frame",           SVC_RET0},
    {"__cxa_begin_cleanup",          SVC_RET0},
    {"__cxa_call_unexpected",        SVC_RET0},
    {"__cxa_type_match",             SVC_RET0},
    
    {"AInputEvent_getDeviceId",      SVC_RET0},
    {"AInputEvent_getSource",        SVC_RET0},
    {"AInputEvent_getType",          SVC_RET0},
    {"AInputQueue_attachLooper",     SVC_RET0},
    {"AInputQueue_finishEvent",      SVC_RET0},
    {"AInputQueue_getEvent",         SVC_RETM1},
    {"AInputQueue_preDispatchEvent", SVC_RET0},
    {"AKeyEvent_getAction",          SVC_RET0},
    {"AKeyEvent_getKeyCode",         SVC_RET0},
    {"AKeyEvent_getMetaState",       SVC_RET0},
    {"AMotionEvent_getAction",       SVC_RET0},
    {"AMotionEvent_getEventTime",    SVC_RET0},
    {"AMotionEvent_getHistoricalEventTime", SVC_RET0},
    {"AMotionEvent_getHistoricalX",  SVC_RET0},
    {"AMotionEvent_getHistoricalY",  SVC_RET0},
    {"AMotionEvent_getHistorySize",  SVC_RET0},
    {"AMotionEvent_getPointerCount", SVC_RET0},
    {"AMotionEvent_getPointerId",    SVC_RET0},
    {"AMotionEvent_getX",            SVC_RET0},
    {"AMotionEvent_getY",            SVC_RET0},
    {"_MotionEvent_getAxisValue",    SVC_RET0},
    {"_MotionEvent_getButtonState",  SVC_RET0},
    {"_MotionEvent_getToolType",     SVC_RET0},
    
    {"ASensor_getMinDelay",          SVC_RET0},
    {"ASensor_getName",              SVC_RET0},
    {"ASensor_getResolution",        SVC_RET0},
    {"ASensor_getType",              SVC_RET0},
    {"ASensor_getVendor",            SVC_RET0},
    {"ASensorEventQueue_disableSensor", SVC_RET0},
    {"ASensorEventQueue_enableSensor",  SVC_RET0},
    {"ASensorEventQueue_getEvents",     SVC_RETM1},
    {"ASensorEventQueue_hasEvents",     SVC_RETM1},
    {"ASensorEventQueue_setEventRate",  SVC_RET0},
    {"ASensorManager_createEventQueue", SVC_RET0},
    {"ASensorManager_destroyEventQueue",SVC_RET0},
    {"ASensorManager_getDefaultSensor", SVC_RET0},
    {"ASensorManager_getInstance",      SVC_RET0},
    {"ASensorManager_getSensorList",    SVC_RET0},
    
    {"_Z28ANativeActivity_feedWatchdogd",  SVC_RET0},
    {"_Z29ANativeActivity_getInputQueuev", SVC_RET0},
    {"_Z30ANativeActivity_enableWatchdogb",SVC_RET0},
    {"_Z42ANativeActivity_forwardMotionEventToDalvikP11AInputEventP7_JNIEnv", SVC_RET0},
    {"_Z42ANativeActivity_setFirstLevelHasBeenLoadedb", SVC_RET0},
    
    {"SHA1Init",   SVC_RET0},
    {"SHA1Update", SVC_RET0},
    {"SHA1Final",  SVC_RET0},
    
    {"deflate",       SVC_Z_DEFLATE},
    {"deflateEnd",    SVC_Z_DEFLATEEND},
    {"deflateInit2_", SVC_Z_DEFLATEINIT2},
    {"deflateReset",  SVC_Z_DEFLATERESET},
    {"inflateInit_",  SVC_Z_INFLATEINIT},
    
    {"recv",         SVC_RETM1},
    {"recvfrom",     SVC_RETM1},
    {"recvmsg",      SVC_RETM1},
    {"send",         SVC_RETM1},
    {"sendfile",     SVC_RETM1},
    {"sendmsg",      SVC_RETM1},
    {"sendto",       SVC_RETM1},
    {"shutdown",     SVC_RETM1},
    {"select",       SVC_RETM1},
    {"poll",         SVC_RETM1},
    {"getpeername",  SVC_RETM1},
    {"getsockname",  SVC_RETM1},
    {"getsockopt",   SVC_RETM1},
    {"setsockopt",   SVC_RETM1},
    {"epoll_ctl",    SVC_RETM1},
    {"epoll_wait",   SVC_RETM1},
    {"ptrace",       SVC_RETM1},
    {"tkill",        SVC_PTHREAD_KILL},
    {"freeaddrinfo", SVC_RET0},
    {"gai_strerror", SVC_RET0},
    {"gethostbyaddr",SVC_RET0},
    {"inet_aton",    SVC_RET0},
    {"inet_ntop",    SVC_RET0},
    {"inet_pton",    SVC_RET0},
    
    {"atexit",                   SVC_RET0},
    {"getgrgid",                 SVC_RET0},
    {"getgrnam",                 SVC_RET0},
    {"getpwnam",                 SVC_RET0},
    {"getpwuid",                 SVC_RET0},
    {"getresuid",                SVC_RET0},
    {"ioctl",                    SVC_RET0},
    {"setitimer",                SVC_RET0},
    {"setresuid",                SVC_RET0},
    {"sigsuspend",               SVC_SIGSUSPEND},
    {"unsetenv",                 SVC_RET0},
    {"utime",                    SVC_RET0},
    {"__div0",                   SVC_RET0},
    {"__gnu_uldivmod_helper",    SVC_RET0},
    {"pthread_attr_setschedparam",  SVC_PTHREAD_ATTR_NOOP},
    {"pthread_attr_setschedpolicy", SVC_PTHREAD_ATTR_NOOP},
    
    {"getwc",    SVC_RET0},
    {"putwc",    SVC_RET0},
    {"ungetwc",  SVC_RET0},
    {"wcscoll",  SVC_RET0},
    {"wcsxfrm",  SVC_RET0},
    {"wcsftime", SVC_RET0},
    
    {"fscanf",   SVC_RET0},
    {"fdopen",   SVC_FDOPEN},
    {"vsprintf", SVC_VSPRINTF2},
    
    {"frexp",          SVC_FREXP},
    {"rint",           SVC_RINT},
    {"lrand48",        SVC_LRAND48},
    {"srand48",        SVC_SRAND48},
    {"strpbrk",        SVC_STRPBRK},
    {"strtok",         SVC_STRTOK},
    {"dup2",           SVC_DUP2},
    {"clock_getres",   SVC_CLOCK_GETRES},
    {"gethostname",    SVC_GETHOSTNAME},
    {"getrusage",      SVC_GETRUSAGE},
    {"getc",           SVC_GETC},
    {"ungetc",         SVC_RET0},
    {"putc",           SVC_FPUTC},
    {"putchar",        SVC_PUTCHAR},
    {"__fpclassifyf",  SVC_FPCLASSIFYF},
    {"__get_h_errno",  SVC_RET0},
    {"inet_addr",      SVC_INET_ADDR},
    {"inet_ntoa",      SVC_RET0},
    {"fcntl",          SVC_FCNTL2},
    
    
    {"fma",            SVC_FMA},
    {"fmaf",           SVC_FMAF},
    {"scalbn",         SVC_SCALBN},
    {"scalbnf",        SVC_SCALBNF},
    {"ldexpl",         SVC_SCALBN},   /* ldexpl ≈ scalbn for bionic compatibility */
    {"ilogb",          SVC_ILOGB},
    {"ilogbf",         SVC_ILOGBF},
    /* libc fd / stdio */
    {"dup",            SVC_DUP},
    {"ferror",         SVC_FERROR},
    {"rewinddir",      SVC_REWINDDIR},
    /* multibyte / wchar */
    {"mbtowc",         SVC_MBTOWC},
    {"mbrlen",         SVC_MBRLEN},
    {"mbsrtowcs",      SVC_MBSRTOWCS},
    {"mbsnrtowcs",     SVC_MBSRTOWCS},  /* mbsnrtowcs: same path, n as limit */
    {"wcsnrtombs",     SVC_RET0},       /* wcsnrtombs: stub (zero-byte conversion) */
    {"strtold_l",      SVC_STRTOLD},    /* strtold_l(s, end, loc) ≈ strtold */
    
    {"getnameinfo",    SVC_RETM1},
    {"getprotobyname", SVC_RET0},
    {"if_nametoindex", SVC_RET0},
    
    {"setpgid",        SVC_RET0},
    {"truncate",       SVC_RETM1},
    {"symlink",        SVC_RETM1},
    {"utimes",         SVC_RET0},
    {"clock_nanosleep",SVC_NANOSLEEP},  /* clock_nanosleep(clkid, flags, req, rem) ≈ nanosleep */
    {"getpwuid_r",     SVC_RET0},       /* getpwuid_r: no user info */
    {"fnmatch",        SVC_RETM1},      /* fnmatch: fail on pattern mismatch */
    /* signal data symbols → SVC_RET0 (resolved as direct VA) */
    {"__libc_current_sigrtmax", SVC_RET0},
    {"__libc_current_sigrtmin", SVC_RET0},
    
    {"AImage_getHardwareBuffer",              SVC_RET0},
    {"AImage_delete",                         SVC_RET0},
    {"AImage_deleteAsync",                    SVC_RET0},
    {"AImage_getWidth",                       SVC_RET0},
    {"AImage_getTimestamp",                   SVC_RET0},
    {"AHardwareBuffer_acquire",               SVC_RET0},
    {"AHardwareBuffer_describe",              SVC_RET0},
    {"AHardwareBuffer_release",               SVC_RET0},
    {"AImageReader_setBufferRemovedListener", SVC_RET0},
    {"AImageReader_newWithUsage",             SVC_RET0},
    {"AImageReader_getWindow",                SVC_RET0},
    {"AImageReader_setImageListener",         SVC_RET0},
    {"AImageReader_delete",                   SVC_RET0},
    {"AImageReader_acquireLatestImage",       SVC_RETM1},
    {"AMediaExtractor_seekTo",                SVC_RET0},
    {"AMediaExtractor_new",                   SVC_RET0},
    {"AMediaExtractor_delete",                SVC_RET0},
    {"AMediaExtractor_setDataSourceFd",       SVC_RETM1},
    {"AMediaExtractor_setDataSource",         SVC_RETM1},
    {"AMediaExtractor_setDataSourceCustom",   SVC_RETM1},
    {"AMediaExtractor_getTrackCount",         SVC_RET0},
    {"AMediaExtractor_getTrackFormat",        SVC_RET0},
    {"AMediaExtractor_selectTrack",           SVC_RET0},
    {"AMediaExtractor_getSampleTrackIndex",   SVC_RETM1},
    {"AMediaExtractor_getSampleTime",         SVC_RET0},
    {"AMediaExtractor_readSampleData",        SVC_RETM1},
    {"AMediaExtractor_advance",               SVC_RET0},
    {"AMediaFormat_getInt32",                 SVC_RET0},
    {"AMediaFormat_getInt64",                 SVC_RET0},
    {"AMediaFormat_getFloat",                 SVC_RET0},
    {"AMediaFormat_getString",                SVC_RET0},
    {"AMediaFormat_setInt32",                 SVC_RET0},
    {"AMediaFormat_delete",                   SVC_RET0},
    {"AMediaCodec_createDecoderByType",       SVC_RET0},
    {"AMediaCodec_configure",                 SVC_RETM1},
    {"AMediaCodec_start",                     SVC_RETM1},
    {"AMediaCodec_stop",                      SVC_RET0},
    {"AMediaCodec_flush",                     SVC_RET0},
    {"AMediaCodec_delete",                    SVC_RET0},
    {"AMediaCodec_dequeueInputBuffer",        SVC_RETM1},
    {"AMediaCodec_dequeueOutputBuffer",       SVC_RETM1},
    {"AMediaCodec_getInputBuffer",            SVC_RET0},
    {"AMediaCodec_getOutputBuffer",           SVC_RET0},
    {"AMediaCodec_getOutputFormat",           SVC_RET0},
    {"AMediaCodec_queueInputBuffer",          SVC_RET0},
    {"AMediaCodec_releaseOutputBuffer",       SVC_RET0},
    {"AMediaCodec_setOutputSurface",          SVC_RET0},
    {"AMediaDataSource_new",                  SVC_RET0},
    {"AMediaDataSource_delete",               SVC_RET0},
    {"AMediaDataSource_setUserdata",          SVC_RET0},
    {"AMediaDataSource_setReadAt",            SVC_RET0},
    {"AMediaDataSource_setGetSize",           SVC_RET0},
    {"AMediaDataSource_setClose",             SVC_RET0},
    
    /* C++ runtime */
    {"__cxa_thread_atexit_impl",  SVC_RET0},
    /* __aeabi_memmove4/8: word-aligned memmove ABI variants */
    {"__aeabi_memmove4",          SVC_MEMMOVE},
    {"__aeabi_memmove8",          SVC_MEMMOVE},
    
    {"link",                      SVC_RETM1},   /* hard link: unsupported */
    {"pathconf",                  SVC_RETM1},   /* pathconf: unsupported */
    {"utimensat",                 SVC_RET0},
    {"fchmodat",                  SVC_RET0},
    {"futimens",                  SVC_RET0},
    /* stdio */
    {"fgetc",                     SVC_GETC},    /* fgetc ≡ getc */
    
    {"signal",                    SVC_RET0},    /* return SIG_DFL (0) */
    {"sigdelset",                 SVC_RET0},
    {"sigfillset",                SVC_RET0},
    
    {"pthread_atfork",            SVC_RET0},
    {"__register_atfork",         SVC_RET0},
    
    {"logb",                      SVC_LOGB},
    {"lrintf",                    SVC_LRINTF},
    {"expm1f",                    SVC_EXPM1F},
    {"nanf",                      SVC_NANF},
    
    {"wmemcmp",                   SVC_WMEMCMP},
    {"swprintf",                  SVC_SWPRINTF},
    
    {"localeconv",                SVC_LOCALECONV},
    
    {"socketpair",                SVC_SOCKETPAIR},
    
    {"AConfiguration_getSdkVersion",         SVC_ACFG_SDKVER},
    
    {"AChoreographer_getInstance",            SVC_ACHOREOGRAPHER_GET},
    {"AChoreographer_postFrameCallback",      SVC_ACHOREOGRAPHER_POST},
    {"AChoreographer_postFrameCallback64",    SVC_ACHOREOGRAPHER_POST64},
    {"AChoreographer_postFrameCallbackDelayed", SVC_ACHOREOGRAPHER_POSTDELAY},
    {"AChoreographer_registerRefreshRateCallback",   SVC_RET0},
    {"AChoreographer_unregisterRefreshRateCallback", SVC_RET0},
    
    {"AAssetManager_fromJava",    SVC_AASSETMGR_FROMJAVA},
    {"AAssetManager_open",        SVC_AASSETMGR_OPEN},
    {"AAsset_getBuffer",          SVC_AASSET_GETBUFFER},
    {"AAsset_getLength",          SVC_AASSET_GETLENGTH},
    {"AAsset_close",              SVC_RET0},
};

/* ---- Math passthrough tables (index = SVC offset from the block base) ---- */
typedef float  (*math_f1)(float);
typedef float  (*math_f2)(float, float);
typedef double (*math_d1)(double);
typedef double (*math_d2)(double, double);

static const std::pair<const char *, math_f1> kMathF1[SVC_MATH_F1_COUNT] = {
    {"sinf", sinf},   {"cosf", cosf},   {"tanf", tanf},
    {"asinf", asinf}, {"acosf", acosf}, {"atanf", atanf},
    {"sqrtf", sqrtf}, {"expf", expf},   {"exp2f", exp2f},
    {"logf", logf},   {"log2f", log2f}, {"log10f", log10f},
    {"floorf", floorf}, {"ceilf", ceilf}, {"roundf", roundf},
    {"truncf", truncf}, {"fabsf", fabsf}, {"cbrtf", cbrtf},
    {"sinhf", sinhf}, {"coshf", coshf},  {"tanhf", tanhf},
    {"asinhf", asinhf}, {"acoshf", acoshf}, {"atanhf", atanhf},
    {"nearbyintf", nearbyintf}, {"rintf", rintf},
};
static const std::pair<const char *, math_f2> kMathF2[SVC_MATH_F2_COUNT] = {
    {"atan2f", atan2f}, {"powf", powf},   {"fmodf", fmodf},
    {"fminf", fminf},   {"fmaxf", fmaxf}, {"copysignf", copysignf},
    {"hypotf", hypotf}, {"remainderf", remainderf},
};
static const std::pair<const char *, math_d1> kMathD1[SVC_MATH_D1_COUNT] = {
    {"sin", sin},   {"cos", cos},   {"tan", tan},
    {"asin", asin}, {"acos", acos}, {"atan", atan},
    {"sqrt", sqrt}, {"exp", exp},   {"exp2", exp2},
    {"log", log},   {"log2", log2}, {"log10", log10},
    {"floor", floor}, {"ceil", ceil}, {"round", round},
    {"trunc", trunc}, {"fabs", fabs}, {"cbrt", cbrt},
    {"sinh", sinh}, {"cosh", cosh},  {"tanh", tanh},
    {"log1p", log1p},
    /* inverse hyperbolic functions (were unresolved, returned 0) */
    {"acosh", acosh}, {"asinh", asinh}, {"atanh", atanh},
};
static const std::pair<const char *, math_d2> kMathD2[SVC_MATH_D2_COUNT] = {
    {"atan2", atan2}, {"pow", pow},   {"fmod", fmod},
    {"fmin", fmin},   {"fmax", fmax}, {"hypot", hypot},
};

/* Guest-side ARM stub for pthread_once: actually runs the init routine
 * (an SVC handler cannot re-enter the JIT, so this must be guest code). */
static constexpr uint32_t PTHREAD_ONCE_STUB = 0x41008000u;
static constexpr uint32_t MJIV_STUB         = 0x41009000u; /* mono_jit_init_version wrapper */
static constexpr uint32_t MONO_CFG_PARSE_STUB = 0x41009100u; /* mono_config_parse: noop (done in prepare) */
static constexpr uint32_t MONO_CFG_ASM_STUB   = 0x41009110u; /* mono_config_for_assembly: noop ret0 */
static constexpr uint32_t G_BUILD_FN_STUB     = 0x41009120u; /* g_build_filename: NULL-safe */
static uint32_t           g_mjiv_stub_va    = 0;
static uint32_t           g_mono_config_parse_tramp = 0;
static uint32_t           g_g_build_filename_tramp = 0;

/* Guest-side ARM stub that makes mono_jit_init_version() idempotent.
 * Unity's libunity calls mono_jit_init_version() from nativeRender every frame
 * (its engine-initialized flag only latches on a fully successful init, which
 * never completes because mini_init re-registers JIT icalls and asserts on the
 * 2nd+ call).  Each redundant call re-runs mini_init, leaking ~320 MB and
 * eventually exhausting the heap.  This stub caches the first non-NULL domain
 * and returns it on subsequent calls, so the real mini_init runs exactly once.
 * Built lazily when libunity's import is resolved (real address known by then).
 * Like pthread_once, it must be guest code (an SVC cannot re-enter the JIT). */

/* exception-logging tail-call stubs (LUNARIA_TRACE_EXC).  One page holds
 * up to a few stubs; each: `svc #N` (log args, regs preserved) then
 * `ldr pc,[pc,#-4]` to the real libmono function (preserving r0-r3/lr so the
 * real fn returns straight to the original caller).  Redirecting the JUMP_SLOT
 * to a stub catches every PLT/GOT call reliably. */
static constexpr uint32_t EXC_STUB_BASE = 0x4100a000u;
static uint32_t           g_exc_stub_from_name = 0;
static uint32_t           g_exc_stub_raise     = 0;

/* inline-detour state (defined early so dispatch_svc can read it). */
static constexpr uint32_t DETOUR_STUB_BASE = 0x4100c000u;
static uint32_t     g_detour_count = 0;
static const char  *g_detour_names[NUM_DETOURS]   = {};
static uint32_t     g_detour_targets[NUM_DETOURS] = {};

/* Fast in-guest pthread_mutex stubs (no SVC).
 * Unity hammers global mutexes millions of times per frame; SVC round-trips were
 * far too slow.  Under cooperative scheduling, preemption only happens at SVC
 * boundaries, so lock state ([mutex] = count<<8 | (tid+1)) can be updated in guest
 * code.  Low-frequency ops (cond_wait, init) still use SVC.  CUR_TID_VA mirrors tid+1. */
static constexpr uint32_t FAST_MUTEX_PAGE = 0x4100f000u;
static constexpr uint32_t CUR_TID_VA      = FAST_MUTEX_PAGE; /* word: tid+1 */
static uint32_t g_fastmutex_lock_va    = 0;
static uint32_t g_fastmutex_unlock_va  = 0;
static uint32_t g_fastmutex_trylock_va = 0;
struct ArmExecCtx;
static void set_cur_tid(ArmExecCtx &ctx, uint32_t tid);

/* Symbols resolved to a direct guest VA instead of an SVC trampoline */
static uint32_t lookup_symbol_direct_va(const char *name) {
    if (strcmp(name, "pthread_once") == 0) return PTHREAD_ONCE_STUB;
    if (g_fastmutex_lock_va) {
        if (strcmp(name, "pthread_mutex_lock") == 0)    return g_fastmutex_lock_va;
        if (strcmp(name, "pthread_mutex_unlock") == 0)  return g_fastmutex_unlock_va;
        if (strcmp(name, "pthread_mutex_trylock") == 0) return g_fastmutex_trylock_va;
    }
    /* bionic libc *data* exports (variables, not functions).  These must point
 * at real guest memory holding the value; if they fall through to an SVC
 * trampoline, a dereference reads the trampoline's instruction word. */
    if (strcmp(name, "__page_size")  == 0) return LIBC_PAGE_SIZE;
    if (strcmp(name, "__page_shift") == 0) return LIBC_PAGE_SHIFT;
    if (strcmp(name, "__page_mask")  == 0) return LIBC_PAGE_MASK;
    
    /* AMEDIAFORMAT_KEY_* are const char* data symbols; *ptr==0 yields empty string */
    if (strncmp(name, "AMEDIAFORMAT_KEY_", 17) == 0) return NOOP_RET0;
    /* __end__: linker section-end symbol for Boehm GC scan bound; stub suffices */
    if (strcmp(name, "__end__") == 0) return NOOP_RET0;
    /* sys_signame: BSD signal-name table; use a zeroed page */
    if (strcmp(name, "sys_signame") == 0) return NOOP_RET0;
    /* _ZTH15gDeferredAction: C++ TLS destructor hook; ret0 stub is harmless */
    if (strcmp(name, "_ZTH15gDeferredAction") == 0) return NOOP_RET0;
    /* __libc_current_sigrtmax/min: inline on bionic but sometimes GOT-referenced */
    if (strcmp(name, "__libc_current_sigrtmax") == 0) return NOOP_RET0;
    if (strcmp(name, "__libc_current_sigrtmin") == 0) return NOOP_RET0;
    return 0;
}

/* Unified symbol → SVC lookup (main map + math blocks) */
static uint32_t lookup_symbol_svc(const char *name) {
    for (auto &[n, s] : kSymbolSvcMap)
        if (strcmp(n, name) == 0) return s;
    for (uint32_t i = 0; i < SVC_MATH_F1_COUNT; ++i)
        if (strcmp(kMathF1[i].first, name) == 0) return SVC_MATH_F1_BASE + i;
    for (uint32_t i = 0; i < SVC_MATH_F2_COUNT; ++i)
        if (strcmp(kMathF2[i].first, name) == 0) return SVC_MATH_F2_BASE + i;
    for (uint32_t i = 0; i < SVC_MATH_D1_COUNT; ++i)
        if (strcmp(kMathD1[i].first, name) == 0) return SVC_MATH_D1_BASE + i;
    for (uint32_t i = 0; i < SVC_MATH_D2_COUNT; ++i)
        if (strcmp(kMathD2[i].first, name) == 0) return SVC_MATH_D2_BASE + i;
    return UINT32_MAX;
}

/* -------------------------------------------------------------------------
 * Memory model — flat 4 GB host reservation
 * The whole ARM 32-bit address space is backed by one anonymous MAP_NORESERVE
 * host mapping, so any guest VA maps to host_base + va in O(1).  Pages are
 * committed lazily by the kernel on first touch.  This also enables dynarmic
 * fastmem (guest loads/stores compile to direct host accesses).
 * ---------------------------------------------------------------------- */
class ArmMemory {
public:
    uint8_t *host = nullptr;

    void init() {
        if (host) return;
        /* 4 GB + 64 KB guard so 4-byte accesses at 0xFFFFFFFF stay in-bounds */
        void *p = ::mmap(nullptr, 0x100010000ull, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) { perror("arm_exec: mmap 4GB arena"); abort(); }
        host = static_cast<uint8_t *>(p);
    }
    void map(uint32_t base, uint32_t size, const void *src = nullptr) {
        init();
        if (src && size) std::memcpy(host + base, src, size);
        /* anonymous pages are already zero-filled */
        (void)base; (void)size;
    }
    uint8_t *ptr(uint32_t va) { return host + va; }
    void write32(uint32_t va, uint32_t v);
    uint32_t read32(uint32_t va) {
        uint32_t v; std::memcpy(&v, host + va, 4); return v; }
    const char *cstr(uint32_t va) {
        if (!va) return nullptr;
        return reinterpret_cast<const char *>(host + va);
    }
};

/* -------------------------------------------------------------------------
 * Execution context
 * ---------------------------------------------------------------------- */
struct RegisteredNative {
    std::string klass, name, sig;
    uint32_t    fn_va;
};

/* Forward-declare so ArmExecCtx can hold a persistent callbacks object */
class ArmCallbacks;

struct ArmExecCtx {
    ArmMemory                        mem;
    struct jvm                      *jvm = nullptr;
    std::vector<RegisteredNative>    natives;
    Dynarmic::ExclusiveMonitor       excl_mon{3};
    std::unique_ptr<Dynarmic::A32::Jit> jit;
    /* Persistent JIT + callbacks — avoids re-compilation on every call */
    std::unique_ptr<ArmCallbacks>    cb;
    /* Second JIT used to run guest threads while the main JIT is inside a
 * call (dynarmic cannot re-enter Run on the same instance). */
    std::unique_ptr<Dynarmic::A32::Jit> jit_aux;
    std::unique_ptr<ArmCallbacks>    cb_aux;
    /* Third JIT, reserved for short guest callbacks invoked from inside an SVC
 * handler (e.g. the comparator passed to bsearch) — the main and aux JITs
 * may both be mid-Run when such a callback fires, and dynarmic cannot
 * re-enter Run on a live instance. */
    std::unique_ptr<Dynarmic::A32::Jit> jit_cb;
    std::unique_ptr<ArmCallbacks>    cb_cb;
    /* bump allocator for ARM heap */
    uint32_t                         heap_ptr = HEAP_BASE;
    /* Callee-saved registers (R4-R11) persist between JNI calls to model JVM thread state */
    uint32_t                         saved_regs[8] = {};   /* R4..R11 */
    bool                             regs_valid = false;
};

/* -------------------------------------------------------------------------
 * ARM heap allocator (free-list based)
 * The original implementation was a pure bump pointer with free() as a no-op,
 * so every malloc permanently consumed heap. A long-running Unity title
 * allocates and frees continuously; with no reclamation the 512 MB heap was
 * exhausted after a few hundred frames, arm_malloc() returned 0, and the
 * guest's operator new threw an uncaught std::bad_alloc → abort loop.
 * Each allocation carries an 8-byte header immediately before its payload:
 * [payload_size : u32][magic : u32]
 * Freed blocks go on a best-fit free list (size → payload VA) and are reused;
 * oversized blocks are split. Payloads stay 8-byte aligned, matching the
 * previous contract. free() validates the magic so foreign / double frees are
 * ignored rather than corrupting the list.
 * ---------------------------------------------------------------------- */
static constexpr uint32_t HEAP_HDR    = 8u;
static constexpr uint32_t ALLOC_MAGIC = 0xA110C8EDu;
static constexpr uint32_t FREE_MAGIC  = 0xF2EEB10Cu;
/* payload_size → payload VA of free blocks (best-fit via lower_bound) */
static std::multimap<uint32_t, uint32_t> g_free_list;

static inline uint32_t arm_align8(uint32_t n) { return (n + 7u) & ~7u; }

static uint64_t g_malloc_calls = 0, g_malloc_reused = 0, g_free_calls = 0;

static uint32_t arm_malloc(ArmExecCtx &ctx, uint32_t size) {
    if (size == 0) size = 1;
    size = arm_align8(size);
    ++g_malloc_calls;

    /* best-fit reuse from the free list */
    auto it = g_free_list.lower_bound(size);
    if (it != g_free_list.end()) {
        uint32_t blk = it->first;   /* payload size of the free block */
        uint32_t va  = it->second;  /* payload VA */
        g_free_list.erase(it);
        ++g_malloc_reused;
        /* split if the leftover can hold a header + a minimal (8B) payload */
        if (blk >= size + HEAP_HDR + 8u) {
            uint32_t rem_hdr     = va + size;
            uint32_t rem_payload = rem_hdr + HEAP_HDR;
            uint32_t rem_size    = blk - size - HEAP_HDR;
            ctx.mem.write32(rem_hdr,     rem_size);
            ctx.mem.write32(rem_hdr + 4, FREE_MAGIC);
            g_free_list.insert({rem_size, rem_payload});
            ctx.mem.write32(va - HEAP_HDR, size); /* shrink this block */
        }
        ctx.mem.write32(va - 4, ALLOC_MAGIC);
        /* Zero reused block: g_malloc0 calls malloc expecting zeroed memory */
        memset(ctx.mem.ptr(va), 0, size);
        return va;
    }

    /* bump fresh memory: header + payload */
    if ((uint64_t)ctx.heap_ptr + HEAP_HDR + size > (uint64_t)HEAP_BASE + HEAP_SIZE) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[arm_exec] arm_malloc OOM: req=%u, heap full at %uMB "
                    "(calls=%llu reused=%llu free=%llu freelist=%zu)\n",
                    size, (uint32_t)((ctx.heap_ptr - HEAP_BASE) >> 20),
                    (unsigned long long)g_malloc_calls,
                    (unsigned long long)g_malloc_reused,
                    (unsigned long long)g_free_calls,
                    g_free_list.size());
        }
        return 0; /* OOM */
    }
    uint32_t hdr = ctx.heap_ptr;
    uint32_t va  = hdr + HEAP_HDR;
    ctx.mem.write32(hdr,     size);
    ctx.mem.write32(hdr + 4, ALLOC_MAGIC);
    ctx.heap_ptr = va + size;
    return va;
}

/* true if va looks like a live arm_malloc payload */
static bool arm_heap_owns(ArmExecCtx &ctx, uint32_t va) {
    return va >= HEAP_BASE + HEAP_HDR && va < ctx.heap_ptr &&
           ctx.mem.read32(va - 4) == ALLOC_MAGIC;
}

static void arm_free(ArmExecCtx &ctx, uint32_t va) {
    if (!va || !arm_heap_owns(ctx, va)) return; /* foreign / double free → ignore */
    ++g_free_calls;
    uint32_t size = ctx.mem.read32(va - HEAP_HDR);
    if (size == 0 || (uint64_t)va + size > ctx.heap_ptr) return; /* corrupt → ignore */
    ctx.mem.write32(va - 4, FREE_MAGIC);
    g_free_list.insert({size, va});
}

static uint32_t arm_realloc(ArmExecCtx &ctx, uint32_t va, uint32_t newsize) {
    if (!va) return arm_malloc(ctx, newsize);
    if (newsize == 0) { arm_free(ctx, va); return 0; }
    if (!arm_heap_owns(ctx, va)) {
        /* unknown origin: allocate fresh and copy the requested amount */
        uint32_t n = arm_malloc(ctx, newsize);
        if (n) std::memmove(ctx.mem.ptr(n), ctx.mem.ptr(va), newsize);
        return n;
    }
    uint32_t oldsize = ctx.mem.read32(va - HEAP_HDR);
    if (arm_align8(newsize) <= oldsize) return va; /* fits in place */
    uint32_t n = arm_malloc(ctx, newsize);
    if (n) { std::memmove(ctx.mem.ptr(n), ctx.mem.ptr(va), oldsize); arm_free(ctx, va); }
    return n;
}

/* memalign: bump an `align`-aligned payload preceded by a normal (freeable)
 * header.  For align ≤ 8 the plain allocator already satisfies the request. */
static uint32_t arm_memalign(ArmExecCtx &ctx, uint32_t align, uint32_t size) {
    if (align <= HEAP_HDR || (align & (align - 1))) return arm_malloc(ctx, size);
    if (size == 0) size = 1;
    size = arm_align8(size);
    uint32_t payload = (ctx.heap_ptr + HEAP_HDR + align - 1u) & ~(align - 1u);
    if ((uint64_t)payload + size > (uint64_t)HEAP_BASE + HEAP_SIZE) return 0;
    ctx.mem.write32(payload - HEAP_HDR, size);
    ctx.mem.write32(payload - 4,        ALLOC_MAGIC);
    ctx.heap_ptr = payload + size;
    return payload;
}

static ArmExecCtx *g_ctx = nullptr;

/* LUNARIA_WATCH: guest address range [lo,hi) whose writes we trap (wild-store debug).
 * Fastmem is disabled so every store routes through MemoryWrite*.  On the first
 * hit we arm dynarmic single-step mode; the next hit then reports a PRECISE pc
 * (single-instruction blocks update R[15] per instruction). */
static uint32_t g_watch_lo = 0;
static uint32_t g_watch_hi = 0;
/* LUNARIA_TRACE_XCL=lo:hi — log every exclusive (strex) write into [lo,hi) */
static uint32_t g_xcl_lo = 0;
static uint32_t g_xcl_hi = 0;
static bool     g_step_mode = false;     /* drive execution one instruction at a time */
static bool     g_step_capture_done = false;
static uint32_t g_watch_poll_addr = 0;   /* word polled between single-steps */

/* Host FILE* table for ARM fopen/fclose/fread/fwrite/fseek/ftell.
 * SVC fopen() does not hand the guest a raw table index; it returns a pointer
 * to a small guest "FILE shim" so that code reading the libc FILE layout works.
 * In particular mono's mono_file_map_size()/mono_file_map_fd() read the backing
 * fd straight out of FILE->[offset 14] (bionic's `_file` short) and then
 * fstat()/mmap() it.  The shim stores:
 * [0..3] table index into g_file_tab   (for fread/fwrite/fseek/ftell)
 * [14..15] real host fd (fileno)        (for mono_file_map_* + mmap)
 * resolve_file() accepts either a shim pointer or a legacy small index. */
static FILE*    g_file_tab[256] = {};

static FILE* resolve_file(ArmExecCtx &ctx, uint32_t h) {
    if (h == 0) return nullptr;
    uint32_t idx = (h < 256u) ? h : ctx.mem.read32(h); /* shim stores idx at [0] */
    return (idx > 0 && idx < 256u) ? g_file_tab[idx] : nullptr;
}

/* Register a host FILE* into g_file_tab and return a guest shim VA.
 * The shim is a 32-byte guest allocation with:
 * [0..3]   table index  (for fread/fwrite/fseek/ftell via resolve_file)
 * [14..15] host fd      (for mono_file_map_fd() which reads bionic FILE._file)
 * Falls back to bare index if arm_malloc fails.
 * Closes f and returns 0 if the table is full. */
static uint32_t register_file(ArmExecCtx &ctx, FILE *f) {
    if (!f) return 0u;
    uint32_t idx = 0;
    for (uint32_t k = 1; k < 256; ++k)
        if (!g_file_tab[k]) { idx = k; break; }
    if (!idx) {
        fprintf(stderr, "[register_file] ERROR: g_file_tab full (256 entries)\n");
        fclose(f);
        return 0u;
    }
    g_file_tab[idx] = f;
    uint32_t shim = arm_malloc(ctx, 32u);
    if (shim) {
        ctx.mem.write32(shim, idx);
        if (uint8_t *p = (uint8_t*)ctx.mem.ptr(shim)) {
            uint16_t fd16 = (uint16_t)fileno(f);
            memcpy(p + 14, &fd16, 2);
        }
        return shim;
    }
    return idx; /* arm_malloc failed: bare index fallback */
}

/* Guest library regions, recorded by load_elf, used to synthesize a guest-side
 * /proc/self/maps (see synthetic_proc_path).  Each entry is one PT_LOAD segment
 * with its actual ELF permissions (PF_R/W/X → r/w/x in maps notation). */
struct LoadedRegion { uint32_t lo, hi; uint32_t flags; std::string path; };
/* ELF p_flags: PF_X=1, PF_W=2, PF_R=4 */
static std::vector<LoadedRegion> g_loaded_regions;

/* Per-module .ARM.exidx tables (PT_ARM_EXIDX), recorded by load_elf.
 * dl_unwind_find_exidx must return the real table: libc++abi's unwinder
 * needs it to find C++ catch handlers — a fake CANTUNWIND entry turns every
 * thrown Il2CppExceptionWrapper into std::terminate. */
struct ModuleExidx { uint32_t lo, hi, exidx_va, exidx_count; };
static std::vector<ModuleExidx> g_module_exidx;

/* Build a synthetic /proc/self/maps describing the GUEST address space, written
 * to a temp file once and cached.  Why: Boehm GC calls
 * GC_register_dynamic_libraries(), which parses /proc/self/maps to find the
 * writable data segments of loaded libraries and register them as GC roots.
 * Handing it the *host* emulator's maps was doubly wrong — the addresses are
 * host pointers unrelated to the guest's, and the host map keeps growing as the
 * guest allocates, so the GC re-scanned an ever-larger map, push_back()ing
 * region records without bound until the 2.5 GB guest heap OOM'd.
 * A fixed snapshot of the guest library PT_LOAD segments with ACCURATE permissions
 * (code r-xp, data rw-p) prevents Boehm from scanning ~14 MB of code text as
 * roots — only the writable data segments need scanning, which cuts GC scan cost
 * by ~5–10×. */
static const char *synth_guest_maps() {
    static char file[64] = "";
    if (file[0]) return file;
    char buf[64]; strcpy(buf, "/tmp/lunaria-maps-XXXXXX");
    int fd = mkstemp(buf);
    if (fd < 0) return nullptr;
    std::string out;
    char line[512];
    for (const auto &r : g_loaded_regions) {
        /* Translate ELF p_flags to /proc/maps permission string */
        char perm[5];
        perm[0] = (r.flags & 4u) ? 'r' : '-';
        perm[1] = (r.flags & 2u) ? 'w' : '-';
        perm[2] = (r.flags & 1u) ? 'x' : '-';
        perm[3] = 'p'; perm[4] = '\0';
        snprintf(line, sizeof line, "%08x-%08x %s 00000000 00:00 0          %s\n",
                 r.lo, r.hi, perm, r.path.c_str());
        out += line;
    }
    /* Main thread stack, so the GC can find on-stack roots. */
    snprintf(line, sizeof line, "%08x-%08x rw-p 00000000 00:00 0          [stack]\n",
             STACK_BASE, STACK_BASE + STACK_SIZE);
    out += line;
    if (write(fd, out.data(), out.size()) < 0) { /* best effort */ }
    close(fd);
    strncpy(file, buf, sizeof file - 1);
    return file;
}

/* Synthetic /proc/self/stat.  Boehm GC's GC_linux_main_stack_base() opens
 * /proc/self/stat and reads field 28 (startstack) as the address of the top
 * of the main thread's stack, then sanity-checks it against the current SP
 * — aborting with "Absurd stack bottom value" if startstack isn't above SP.
 * Left unredirected, this passed through to the *host* emulator process's
 * own /proc/self/stat, whose startstack is a host address unrelated to the
 * guest's 32-bit stack region, so the check always failed.  Report the
 * guest's actual stack top (matching the "[stack]" entry synthesized for
 * /proc/self/maps above) so the sanity check passes. */
static const char *synth_guest_stat() {
    static char file[64] = "";
    if (file[0]) return file;
    char buf[64]; strcpy(buf, "/tmp/lunaria-stat-XXXXXX");
    int fd = mkstemp(buf);
    if (fd < 0) return nullptr;
    char line[512];
    snprintf(line, sizeof line,
             "1 (lunaria) R 0 1 1 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 0 0 0 0 0 0 "
             "%u 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
             (unsigned)(STACK_BASE + STACK_SIZE));
    if (write(fd, line, strlen(line)) < 0) { /* best effort */ }
    close(fd);
    strncpy(file, buf, sizeof file - 1);
    return file;
}

/* Synthetic /proc/meminfo.  The host may report 16 GB+ of RAM, but this is a
 * 32-bit guest: Unity's MemoryManager ("Dynamic Heap") and the Boehm GC size
 * their heaps from the reported total and, on a big host, expand by 1 MB blocks
 * far past the 4 GB arena until arm_malloc() fails and the guest throws an
 * uncaught std::bad_alloc.  Redirect reads of /proc/meminfo to a small file
 * advertising ~2 GB so both allocators stay within the guest address space.
 * Also redirect /proc/<pid>/maps to a synthetic guest map (see above), and
 * /proc/<pid>/stat to a synthetic stat with a guest-consistent startstack. */
static const char *synthetic_proc_path(const char *path) {
    if (path) {
        /* /proc/self/maps or /proc/<pid>/maps → synthetic guest map. */
        const char *slash = strrchr(path, '/');
        if (slash && !strcmp(slash, "/maps") && !strncmp(path, "/proc/", 6))
            return synth_guest_maps();
        if (slash && !strcmp(slash, "/stat") && !strncmp(path, "/proc/", 6))
            return synth_guest_stat();
    }
    if (!path || strcmp(path, "/proc/meminfo") != 0) return nullptr;
    static char file[64] = "";
    if (!file[0]) {
        char buf[64]; strcpy(buf, "/tmp/lunaria-meminfo-XXXXXX");
        int fd = mkstemp(buf);
        if (fd >= 0) {
            static const char content[] =
                "MemTotal:        2097152 kB\n"
                "MemFree:         1572864 kB\n"
                "MemAvailable:    1769472 kB\n"
                "Buffers:           32768 kB\n"
                "Cached:           262144 kB\n"
                "SwapTotal:             0 kB\n"
                "SwapFree:              0 kB\n";
            if (write(fd, content, sizeof content - 1) < 0) { /* best effort */ }
            close(fd);
            strncpy(file, buf, sizeof file - 1);
        }
    }
    return file[0] ? file : nullptr;
}

/* Cross-library exported symbol table: name → ARM VA (with base applied) */
static std::map<std::string, uint32_t> g_exported_syms;

/* High-water mark of all loaded library segments (page-rounded up).
 * arm_exec_load_library passes base=0 for auto-placement: we assign the next
 * non-overlapping base from this value.  Starts at 0 (libunity.so, loaded
 * first via arm_exec_jni_onload with its own explicit base=0, sets the mark). */
static uint32_t g_lib_load_end = 0u;
/* Absolute paths of every ELF loaded into guest memory (startup + dlopen) */
static std::set<std::string> g_loaded_lib_paths;
struct ArmExecCtx;
static bool load_elf(ArmExecCtx &ctx, const char *path,
                     uint32_t &jni_onload_va, uint32_t base_addr,
                     bool ctors_via_cb);

/* -------------------------------------------------------------------------
 * Cooperative guest threads
 * pthread_create captures a full register context with its own guest stack.
 * arm_exec_run_pending_threads() round-robins each runnable thread for a
 * tick-budgeted slice on the shared JIT, saving/restoring contexts.
 * ---------------------------------------------------------------------- */
struct ArmThread {
    uint32_t                     id = 0;       /* pthread_self value (2-based) */
    std::array<uint32_t, 16>     regs{};
    std::array<uint32_t, 64>     ext{};
    uint32_t                     cpsr  = 0x10;
    uint32_t                     fpscr = 0;
    uint32_t                     stack_base = 0; /* for pthread_attr_getstack */
    uint32_t                     entry_pc = 0;   /* pthread_create start PC */
    bool                         finished = false;
    uint32_t                     defer_count = 0; /* empty-callback deferrals */
    uint32_t                     stuck_pc = 0;   /* PC of repeated zero-instruction fault */
    uint32_t                     stuck_count = 0;/* how many times same PC faulted */
    uint64_t                     total_ticks = 0;/* LUNARIA_SCHED_DUMP: cumulative ticks */
    /* sem_wait blocking: skip slices until g_sems[waiting_sem] > 0 (SVC retries) */
    uint32_t                     waiting_sem = 0;
    uint32_t                     sem_skip_passes = 0; /* for timedwait timeout */
    bool                         sem_timed = false;
    /* futex_wait parking: a worker that FUTEX_WAITs on `waiting_futex` while the
     * guest word still equals `futex_val` is de-scheduled until a matching
     * FUTEX_WAKE token arrives (or the word changes).  Without this the worker
     * kept getting re-run and its Baselib semaphore-acquire loop would read a
     * phantom "available" state and dispatch a not-yet-posted (NULL) job,
     * crashing the Unity JobSystem worker pool at startup. */
    uint32_t                     waiting_futex = 0;
    uint32_t                     futex_val = 0;
};
static std::vector<ArmThread> g_threads;
static uint32_t g_current_tid = 0;      /* 0 = loader/main context */

/* LUNARIA_WATCH_RANGE=lo:hi: log all writes in [lo,hi) (guest stores + host SVC mem ops) */
static uint32_t g_wrange_lo = 0;
static uint32_t g_wrange_hi = 0;
static void wrange_log(uint32_t va, uint32_t val, uint32_t bytes,
                       uint32_t pc, uint32_t lr, const char *src) {
    if (g_wrange_hi == 0 || va >= g_wrange_hi || va + bytes <= g_wrange_lo) return;
    static int n = 0;
    if (n++ < 60)
        fprintf(stderr, "[wrange] %s va=0x%08x val=0x%08x bytes=%u pc~0x%08x "
                "lr=0x%08x tid=%u\n", src, va, val, bytes, pc, lr, g_current_tid);
}
void ArmMemory::write32(uint32_t va, uint32_t v) {
    wrange_log(va, v, 4, 0, 0, "host-write32");
    std::memcpy(host + va, &v, 4);
}

static uint32_t g_thread_stack_next = THREAD_STACK_BASE;

/* pthread TLS: per-(thread,key) value store */
static std::map<std::pair<uint32_t, uint32_t>, uint32_t> g_tls;
static uint32_t g_tls_next_key = 1;

/* LUNARIA_WATCH_FLAG: poll one guest word at SVC granularity and log changes. */
static uint32_t g_watch_flag_va = 0;
static uint32_t g_watch_flag_last = 0;

/* semaphore counters keyed by guest sem_t VA */
static std::map<uint32_t, int32_t> g_sems;

/* Signal handler table: VA registered via sigaction().
 * Used by pthread_kill emulation to simulate Mono GC stop-the-world. */
static std::map<int, uint32_t> g_sighandlers;

/* pthread_mutex guest VA → lock depth; ARM32 mutex is 4 bytes; blocking via SVC_WAIT */
/* Mutex state stored in the mutex word ([va] = count<<8 | (tid+1)) */

/* pthread_cond guest VA → waiter count (signal tracking only) */
static std::map<uint32_t, uint32_t> g_conds; /* VA → signal_pending */

/* futex wake tokens: futex_wake records pending wakes so next futex_wait on same addr returns 0 */
static std::map<uint32_t, uint32_t> g_futex_wake_tokens; /* uaddr → pending wake count */
/* futex waiters: tracks which addrs non-main threads are currently blocking on */
static std::map<uint32_t, uint32_t> g_futex_wait_addrs; /* uaddr → count of waiting threads */

/* pthread_rwlock: no real blocking under cooperative scheduling; track state for EBUSY */
static std::map<uint32_t, int32_t>  g_rwlocks;
static std::map<uint32_t, uint32_t> g_rwlock_writer; /* VA → writer tid */

/* pthread_detach: set of detached thread IDs */
static std::unordered_set<uint32_t> g_detached_threads;

/* recent SVC ring buffer (dumped at run end when LUNARIA_DUMP_LAST_SVC=1) */
struct SvcTraceEnt { uint32_t svc, lr, r0, tid; };
static SvcTraceEnt g_svc_ring[1024];
static uint32_t    g_svc_ring_pos = 0;
/* Dedicated main-thread (tid=0) ring so worker SVC calls don't overwrite it */
static SvcTraceEnt g_svc_ring_main[256];
static uint32_t    g_svc_ring_main_pos = 0;
static void svc_ring_record(uint32_t svc, uint32_t lr, uint32_t r0) {
    g_svc_ring[g_svc_ring_pos++ & 1023] = {svc, lr, r0, g_current_tid};
    if (g_current_tid == 0)
        g_svc_ring_main[g_svc_ring_main_pos++ & 255] = {svc, lr, r0, 0};
}
static void svc_ring_dump(void) {
    fprintf(stderr, "[arm_exec] last SVCs (oldest first):\n");
    for (uint32_t i = 0; i < 1024; ++i) {
        const SvcTraceEnt &e = g_svc_ring[(g_svc_ring_pos + i) & 1023];
        if (e.svc || e.lr)
            fprintf(stderr, "  svc=%u lr=0x%08x r0=0x%08x tid=%u\n",
                    e.svc, e.lr, e.r0, e.tid);
    }
    fprintf(stderr, "[arm_exec] last main-thread SVCs (oldest first):\n");
    for (uint32_t i = 0; i < 256; ++i) {
        const SvcTraceEnt &e = g_svc_ring_main[(g_svc_ring_main_pos + i) & 255];
        if (e.svc || e.lr)
            fprintf(stderr, "  main svc=%u lr=0x%08x r0=0x%08x\n",
                    e.svc, e.lr, e.r0);
    }
}
/* Per-thread errno: shared errno breaks sem_wait EINTR retry across threads */
static std::map<uint32_t, uint32_t> g_errno_vas;
/* return guest errno address for current thread (lazy alloc) */
static uint32_t errno_va(ArmExecCtx &ctx, uint32_t tid) {
    auto it = g_errno_vas.find(tid);
    if (it != g_errno_vas.end()) return it->second;
    uint32_t va = arm_malloc(ctx, 4);
    if (va) ctx.mem.write32(va, 0);
    g_errno_vas[tid] = va;
    return va;
}
/* true when guest thread is blocked — CallSVC yields the slice */
static bool g_yield_requested = false;
/* ALooper_wake() sets this; ALooper_pollOnce spin loop checks it */
static std::atomic<bool> g_alooper_wake_pending{false};
static bool g_in_cb = false;  /* true while call_guest_cb is executing */

/* __errno location in guest memory (allocated lazily from the heap) */
static uint32_t g_errno_va = 0; /* legacy shared errno (kept for compatibility) */

/* host zlib streams keyed by guest z_stream VA */
static std::map<uint32_t, z_stream*> *g_zstreams = nullptr;  /* inflate */
static std::map<uint32_t, z_stream*> *g_dstreams = nullptr;  /* deflate */

/* -------------------------------------------------------------------------
 * EGL / GLFW state for ARM guest rendering
 * ---------------------------------------------------------------------- */
static GLFWwindow *g_glfw     = nullptr;
static EGLDisplay  g_egl_dpy  = EGL_NO_DISPLAY;
static EGLConfig   g_egl_cfg  = nullptr;
static EGLSurface  g_egl_surf = EGL_NO_SURFACE;
static EGLContext  g_egl_ctx  = EGL_NO_CONTEXT;

/* Opaque ARM handles (non-zero = valid, maps to the globals above) */
static constexpr uint32_t ARM_EGL_DISPLAY = 1u;
static constexpr uint32_t ARM_EGL_CONFIG  = 2u;
static constexpr uint32_t ARM_EGL_SURFACE = 3u;
static constexpr uint32_t ARM_EGL_CONTEXT = 4u;
/* Maps fake EGL config handles to host EGLConfig (Android-style chooser) */
static constexpr uint32_t ARM_EGL_CFGTAB_BASE = 0x10u;
static std::vector<EGLConfig> g_egl_cfg_tab;
static EGLConfig resolve_egl_config(uint32_t handle) {
    if (handle >= ARM_EGL_CFGTAB_BASE &&
        handle - ARM_EGL_CFGTAB_BASE < g_egl_cfg_tab.size())
        return g_egl_cfg_tab[handle - ARM_EGL_CFGTAB_BASE];
    return g_egl_cfg;
}
static uint32_t g_fake_anw_va = 0; /* guest ANativeWindow { ops*, ... } */
static uint32_t g_mono_base   = 0; /* libmono.so load base (set in load_elf) */
static std::string g_unity_machine_config; /* embedded in libunity.so rodata */

static const char *memmem_buf(const void *hay, size_t hlen, const char *needle) {
    if (!hay || !needle) return nullptr;
    size_t nlen = strlen(needle);
    if (!nlen || hlen < nlen) return nullptr;
    const char *h = static_cast<const char *>(hay);
    for (size_t i = 0; i + nlen <= hlen; ++i)
        if (memcmp(h + i, needle, nlen) == 0) return h + i;
    return nullptr;
}

struct ArmExecCtx;

static int run_arm(ArmExecCtx &ctx, uint32_t entry_va,
                   uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3,
                   uint64_t tick_budget);

static uint32_t mono_sym_va(const char *sym) {
    if (!sym || !*sym) return 0;
    auto it = g_exported_syms.find(sym);
    return it != g_exported_syms.end() ? (it->second & ~1u) : 0u;
}

/* libmono export by name; fall back to legacy file offset for internal symbols
 * that are not exported in dynsym (build-specific). */
static uint32_t mono_fn_va(const char *sym, uint32_t legacy_off = 0) {
    if (uint32_t va = mono_sym_va(sym)) return va;
    return (legacy_off && g_mono_base) ? g_mono_base + legacy_off : 0u;
}

static void ensure_mono_domain_slot(ArmExecCtx &ctx) {
    if (!g_mono_base) return;
    const uint32_t slot = g_mono_base + 0x3bd918u;
    if (ctx.mem.read32(slot)) return;
    uint32_t domain = (uint32_t)run_arm(ctx, mono_fn_va("mono_get_root_domain", 0x175508u),
                                        0, 0, 0, 0, 10'000'000ULL);
    if (domain) ctx.mem.write32(slot, domain);
}

static constexpr uint32_t ARM_ALOOPER     = 6u; /* non-NULL sentinel for ALooper handle */
static constexpr uint32_t ARM_LIBANDROID  = 0xAB10D000u; /* fake dlopen handle for libandroid.so */
static constexpr uint32_t ARM_LIBAAUDIO   = 0xAB10D100u; /* fake dlopen handle for libaaudio.so */
static constexpr uint32_t ARM_LIBC        = 0xAB10D200u; /* fake dlopen handle for libc.so */
static constexpr uint32_t ARM_AAUDIO_BUILDER = 0xAB10D110u; /* opaque AAudioStreamBuilder* */
static constexpr uint32_t ARM_AAUDIO_STREAM  = 0xAB10D120u; /* opaque AAudioStream* (16B stride) */
static constexpr uint32_t ARM_ACHOREOGRAPHER = 0xC0FEBEEFu;

/* Guest-visible audio buffer handed to AAudio data callbacks (above CB stack) */
static constexpr uint32_t AAUDIO_BUF_BASE = 0x47800000u;
static constexpr uint32_t AAUDIO_BUF_SIZE = 0x00010000u;  /* 64KB */
static constexpr uint32_t AAUDIO_BURST_FRAMES = 480u;     /* 10ms @ 48kHz */

/* FMOD's mixer — and its async command processing, which resolves e.g. the
 * nonblocking-open completion flags Unity's main thread waits on — is driven
 * from the AAudio data callback on a real device.  Track enough builder /
 * stream state to periodically invoke the guest callback ourselves. */
struct AAudioStreamState {
    bool     started   = false;
    bool     input     = false;    /* AAUDIO_DIRECTION_INPUT */
    uint32_t data_cb   = 0;
    uint32_t user_data = 0;
    uint32_t channels  = 2;
    uint32_t format    = 2;        /* AAUDIO_FORMAT_PCM_FLOAT */
    uint32_t sample_rate = 48000;
};
static AAudioStreamState g_aaudio_builder;
static std::map<uint32_t, AAudioStreamState> g_aaudio_streams;
static uint32_t g_aaudio_stream_count = 0;
static bool g_aaudio_buf_mapped = false;

static int32_t call_guest_cb(ArmExecCtx &ctx, uint32_t fn_va, uint32_t a, uint32_t b,
                             uint32_t c = 0, uint32_t d = 0,
                             const uint32_t *stk = nullptr, int nstk = 0);
static void drive_aaudio_callbacks(ArmExecCtx &ctx);
static void drive_java_choreographer(ArmExecCtx &ctx);
/* Java Choreographer.postFrameCallback registrations awaiting a doFrame.
 * The registration arrives while the callback JIT is busy running
 * handleMessage, so delivery is deferred to the drive loops. */
static uint32_t g_java_choreo_pending = 0;

/* Swappy / Unity frame pacing: invoke NDK AChoreographer callbacks from SVC. */
static uint64_t g_choreo_frame_ns = 16'666'666ull;

static void fire_choreographer_callback(ArmExecCtx &ctx, uint32_t fn_va, uint32_t data_va) {
    if (!fn_va) return;
    g_choreo_frame_ns += 16'666'666ull;
    call_guest_cb(ctx, fn_va, (uint32_t)g_choreo_frame_ns, data_va);
}

/* glMapBufferRange shadows: host pointers can't be exposed to the guest, so
 * each mapping gets a guest-side buffer copied back on flush/unmap. */
struct GlMappedBuf { void *host; uint32_t gva, len, access; };
static std::map<uint32_t, GlMappedBuf> g_gl_mapped;   /* keyed by GL target */
/* GLsync handles: guest sees index+1 into this table (host GLsync is 64-bit) */
static std::vector<void*> g_gl_syncs;

/* Set to true when ARM guest calls eglSwapBuffers via SVC.
 * arm_exec_egl_swap() checks this to avoid double-swap. */
static bool g_arm_did_swap = false;
static uint64_t g_guest_egl_swap_count = 0;
static uint64_t g_guest_gl_clear_count = 0;
static uint64_t g_gl_draw_count = 0;          /* draw calls since process start */
static int g_gl_last_viewport[4] = {0,0,0,0}; /* last glViewport arguments */
static uint64_t g_host_egl_swap_count = 0;
/* GL buffer-binding state.  With a VBO/IBO bound, the `pointer` argument of
 * glVertexAttribPointer / glDrawElements is a byte OFFSET into the buffer, not
 * a client memory address — it must be passed through untranslated.  Only when
 * no buffer is bound is it a guest VA needing host translation. */
static uint32_t g_gl_bound_array_buf = 0;     /* GL_ARRAY_BUFFER (0x8892) */
static uint32_t g_gl_bound_elem_buf  = 0;     /* GL_ELEMENT_ARRAY_BUFFER (0x8893) */

/* Unity 4: nativeRender queues GfxDevice work as Java Runnables on a
 * ConcurrentLinkedQueue; Activity.executeGLThreadJobs() drains it on the GL
 * thread.  In single-threaded emulation we enqueue on Queue.add() and drain
 * when executeGLThreadJobs is invoked (JNI CallVoidMethod or Activity stub). */
static std::deque<uint32_t> g_gl_thread_jobs;

/* ReflectionHelper.newProxyInstance(player, long nativePtr, Class[]) creates a
 * java.lang.reflect.Proxy whose InvocationHandler forwards every call to
 * libunity's nativeProxyInvoke(nativePtr, methodName, args).  Unity posts such
 * proxies as Runnables to the UI thread (e.g. the GFX-init round trip that
 * sets a done-flag and signals a cond).  Map: proxy handle -> nativePtr. */
static std::map<uint32_t, uint64_t> g_jproxy_ptrs;

/* bitter.jnibridge.JNIBridge proxies: newInterfaceProxy(jlong ptr, Class[])
 * creates a Java Proxy whose InvocationHandler forwards every call to the
 * native JNIBridge.invoke(ptr, class, method, args) (RegisterNatives'd).
 * UnityChoreographer uses one as its Handler$Callback; when the guest posts
 * Message.sendToTarget() we re-enter invoke() directly since no Java Looper
 * thread exists to dispatch it. */
static std::map<uint32_t, uint64_t> g_jnibridge_ptrs;   /* proxy handle → ptr */
static uint64_t g_jnibridge_last_ptr = 0;
static std::vector<uint64_t> g_jnibridge_order;         /* creation order */
struct JniBridgeIface { uint32_t cls = 0; std::string name; };
/* ptr → all interfaces the proxy was created with (a UnityChoreographer
 * proxy implements both Handler$Callback and Choreographer$FrameCallback) */
static std::map<uint64_t, std::vector<JniBridgeIface>> g_jnibridge_iface;
static std::map<uint32_t, std::string> g_method_stub_names; /* Method handle → name */
static std::map<uint32_t, int32_t> g_message_what; /* Message handle → msg.what */

extern "C" int arm_exec_call6(uint32_t fn_va, uint32_t r0, uint32_t r1,
                              uint32_t r2, uint32_t r3,
                              uint32_t stk0, uint32_t stk1);

static void arm_exec_run_runnable(uint32_t runnable_h)
{
    if (!g_ctx || !g_ctx->jvm || !runnable_h || runnable_h > 65536u) return;
    struct jvm *jvm = g_ctx->jvm;
    JNIEnv *env = &jvm->env;
    auto pit = g_jproxy_ptrs.find(runnable_h);
    if (pit != g_jproxy_ptrs.end()) {
        uint32_t fn = 0;
        for (auto &rn : g_ctx->natives)
            if (rn.name == "nativeProxyInvoke") { fn = rn.fn_va; break; }
        if (!fn) {
            fprintf(stderr, "[jproxy] run h=0x%x: nativeProxyInvoke not registered, skip\n",
                    runnable_h);
            return;
        }
        static uint32_t rh_cls = 0, run_str = 0;
        if (!rh_cls) rh_cls = (uint32_t)(uintptr_t)jvm->native.FindClass(
                                  env, "com/unity3d/player/ReflectionHelper");
        if (!run_str) run_str = (uint32_t)(uintptr_t)jvm->native.NewStringUTF(env, "run");
        fprintf(stderr, "[jproxy] run h=0x%x -> nativeProxyInvoke(ptr=0x%llx, \"run\") @0x%08x\n",
                runnable_h, (unsigned long long)pit->second, fn);
        arm_exec_call6(fn, ENV_SLOT_BASE, rh_cls,
                       (uint32_t)pit->second, (uint32_t)(pit->second >> 32),
                       run_str, 0);
        return;
    }
    jclass rcls = jvm->native.GetObjectClass(env, (jobject)(uintptr_t)runnable_h);
    if ((uintptr_t)rcls > 65536u) return;
    const char *cname = jvm_get_class_name(jvm, rcls);
    uint32_t run_va = 0;
    for (auto &rn : g_ctx->natives) {
        if (rn.name == "run" && rn.sig == "()V") {
            if (cname && rn.klass == cname) { run_va = rn.fn_va; break; }
            if (!run_va) run_va = rn.fn_va;
        }
    }
    if (getenv("LUNARIA_TRACE_JNI"))
        fprintf(stderr, "[runnable] h=0x%x class=%s run_va=0x%x\n",
                runnable_h, cname ? cname : "(null)", run_va);
    if (run_va) {
        arm_exec_call(run_va, ENV_SLOT_BASE, runnable_h, 0, 0);
        return;
    }
    bool java_side = !cname || !strcmp(cname, "java.lang.Object");
    if (!java_side) {
        fprintf(stderr, "[arm_jni] Runnable.run(): no run()V native for class=%s\n", cname);
    } else {
        static int warned = 0;
        if (warned++ < 3 || getenv("LUNARIA_TRACE_JNI"))
            fprintf(stderr, "[arm_jni] Runnable.run(): skipping Java-side Runnable "
                    "(class=java.lang.Object, run() is dex bytecode) — benign\n");
    }
}

extern "C" void arm_exec_gl_job_enqueue(uint32_t runnable_h)
{
    if (runnable_h > 0 && runnable_h <= 65536u)
        g_gl_thread_jobs.push_back(runnable_h);
}

extern "C" void arm_exec_run_pending_threads(void);

extern "C" void arm_exec_drain_gl_thread_jobs(void)
{
    if (!g_ctx) return;
    int drained = 0;
    while (!g_gl_thread_jobs.empty()) {
        uint32_t h = g_gl_thread_jobs.front();
        g_gl_thread_jobs.pop_front();
        arm_exec_run_runnable(h);
        ++drained;
    }
    if (drained && getenv("LUNARIA_TRACE_GL"))
        fprintf(stderr, "[gl] executeGLThreadJobs: ran %d queued Runnable(s)\n", drained);
    arm_exec_run_pending_threads();
}

/* -------------------------------------------------------------------------
 * Host OpenGL ES 2.0 function pointers (loaded from libGLESv2 via dlsym)
 * ---------------------------------------------------------------------- */
static void *g_libgles2 = nullptr;
/* Directory of the main library (e.g. APK lib/armeabi-v7a) — used to satisfy
 * ClassLoader.findLibrary(name) so Unity can locate libil2cpp.so etc. */
static std::string g_main_lib_dir;

/* Helper: reinterpret uint32 register as float */
static inline float rf(uint32_t r) { float f; std::memcpy(&f, &r, 4); return f; }

/* Macro to declare + load a GL function pointer */
#define GL_DECL(ret, name, ...) \
    static ret (*pfn_##name)(__VA_ARGS__) = nullptr;
#define GL_LOAD(name) do { \
    pfn_##name = (decltype(pfn_##name))eglGetProcAddress(#name); \
    if (!pfn_##name && g_libgles2) pfn_##name = (decltype(pfn_##name))dlsym(g_libgles2, #name); \
} while(0)

GL_DECL(void,    glViewport,              GLint,GLint,GLsizei,GLsizei)
GL_DECL(void,    glClear,                 GLbitfield)
GL_DECL(void,    glClearColor,            GLclampf,GLclampf,GLclampf,GLclampf)
GL_DECL(void,    glClearDepthf,           GLclampf)
GL_DECL(void,    glClearStencil,          GLint)
GL_DECL(void,    glEnable,               GLenum)
GL_DECL(void,    glDisable,              GLenum)
GL_DECL(void,    glDepthFunc,            GLenum)
GL_DECL(void,    glDepthMask,            GLboolean)
GL_DECL(void,    glColorMask,            GLboolean,GLboolean,GLboolean,GLboolean)
GL_DECL(void,    glScissor,              GLint,GLint,GLsizei,GLsizei)
GL_DECL(void,    glFrontFace,            GLenum)
GL_DECL(void,    glCullFace,             GLenum)
GL_DECL(void,    glBlendFuncSeparate,    GLenum,GLenum,GLenum,GLenum)
GL_DECL(void,    glBlendEquationSeparate,GLenum,GLenum)
GL_DECL(GLenum,  glGetError,             void)
GL_DECL(const GLubyte*, glGetString,     GLenum)
GL_DECL(void,    glGetIntegerv,          GLenum,GLint*)
GL_DECL(void,    glPixelStorei,          GLenum,GLint)
GL_DECL(void,    glReadPixels,           GLint,GLint,GLsizei,GLsizei,GLenum,GLenum,void*)
GL_DECL(void,    glFlush,               void)
GL_DECL(void,    glFinish,              void)
GL_DECL(void,    glGenBuffers,           GLsizei,GLuint*)
GL_DECL(void,    glBindBuffer,           GLenum,GLuint)
GL_DECL(void,    glBufferData,           GLenum,GLsizeiptr,const void*,GLenum)
GL_DECL(void,    glBufferSubData,        GLenum,GLintptr,GLsizeiptr,const void*)
GL_DECL(void,    glDeleteBuffers,        GLsizei,const GLuint*)
GL_DECL(void,    glGenTextures,          GLsizei,GLuint*)
GL_DECL(void,    glBindTexture,          GLenum,GLuint)
GL_DECL(void,    glActiveTexture,        GLenum)
GL_DECL(void,    glDeleteTextures,       GLsizei,const GLuint*)
GL_DECL(void,    glTexParameteri,        GLenum,GLenum,GLint)
GL_DECL(void,    glTexImage2D,           GLenum,GLint,GLint,GLsizei,GLsizei,GLint,GLenum,GLenum,const void*)
GL_DECL(void,    glTexSubImage2D,        GLenum,GLint,GLint,GLint,GLsizei,GLsizei,GLenum,GLenum,const void*)
GL_DECL(void,    glCopyTexSubImage2D,    GLenum,GLint,GLint,GLint,GLint,GLint,GLsizei,GLsizei)
GL_DECL(void,    glCompressedTexImage2D, GLenum,GLint,GLenum,GLsizei,GLsizei,GLint,GLsizei,const void*)
GL_DECL(void,    glCompressedTexSubImage2D, GLenum,GLint,GLint,GLint,GLsizei,GLsizei,GLenum,GLsizei,const void*)
GL_DECL(void,    glGenerateMipmap,       GLenum)
GL_DECL(void,    glGenFramebuffers,      GLsizei,GLuint*)
GL_DECL(void,    glBindFramebuffer,      GLenum,GLuint)
GL_DECL(void,    glDeleteFramebuffers,   GLsizei,const GLuint*)
GL_DECL(GLenum,  glCheckFramebufferStatus, GLenum)
GL_DECL(void,    glFramebufferTexture2D, GLenum,GLenum,GLenum,GLuint,GLint)
GL_DECL(void,    glFramebufferRenderbuffer, GLenum,GLenum,GLenum,GLuint)
GL_DECL(void,    glGetFramebufferAttachmentParameteriv, GLenum,GLenum,GLenum,GLint*)
GL_DECL(void,    glGenRenderbuffers,     GLsizei,GLuint*)
GL_DECL(void,    glBindRenderbuffer,     GLenum,GLuint)
GL_DECL(void,    glDeleteRenderbuffers,  GLsizei,const GLuint*)
GL_DECL(void,    glRenderbufferStorage,  GLenum,GLenum,GLsizei,GLsizei)
GL_DECL(GLuint,  glCreateShader,         GLenum)
GL_DECL(void,    glShaderSource,         GLuint,GLsizei,const GLchar**,const GLint*)
GL_DECL(void,    glCompileShader,        GLuint)
GL_DECL(void,    glDeleteShader,         GLuint)
GL_DECL(void,    glGetShaderiv,          GLuint,GLenum,GLint*)
GL_DECL(void,    glGetShaderInfoLog,     GLuint,GLsizei,GLsizei*,GLchar*)
GL_DECL(void,    glGetShaderSource,      GLuint,GLsizei,GLsizei*,GLchar*)
GL_DECL(GLuint,  glCreateProgram,        void)
GL_DECL(void,    glAttachShader,         GLuint,GLuint)
GL_DECL(void,    glLinkProgram,          GLuint)
GL_DECL(void,    glUseProgram,           GLuint)
GL_DECL(void,    glDeleteProgram,        GLuint)
GL_DECL(void,    glGetProgramiv,         GLuint,GLenum,GLint*)
GL_DECL(void,    glGetProgramInfoLog,    GLuint,GLsizei,GLsizei*,GLchar*)
GL_DECL(GLint,   glGetAttribLocation,    GLuint,const GLchar*)
GL_DECL(GLint,   glGetUniformLocation,   GLuint,const GLchar*)
GL_DECL(void,    glGetActiveAttrib,      GLuint,GLuint,GLsizei,GLsizei*,GLint*,GLenum*,GLchar*)
GL_DECL(void,    glGetActiveUniform,     GLuint,GLuint,GLsizei,GLsizei*,GLint*,GLenum*,GLchar*)
GL_DECL(void,    glBindAttribLocation,   GLuint,GLuint,const GLchar*)
GL_DECL(void,    glUniform1i,            GLint,GLint)
GL_DECL(void,    glUniform1iv,           GLint,GLsizei,const GLint*)
GL_DECL(void,    glUniform2iv,           GLint,GLsizei,const GLint*)
GL_DECL(void,    glUniform3iv,           GLint,GLsizei,const GLint*)
GL_DECL(void,    glUniform4iv,           GLint,GLsizei,const GLint*)
GL_DECL(void,    glUniform1fv,           GLint,GLsizei,const GLfloat*)
GL_DECL(void,    glUniform2fv,           GLint,GLsizei,const GLfloat*)
GL_DECL(void,    glUniform3fv,           GLint,GLsizei,const GLfloat*)
GL_DECL(void,    glUniform4fv,           GLint,GLsizei,const GLfloat*)
GL_DECL(void,    glUniformMatrix3fv,     GLint,GLsizei,GLboolean,const GLfloat*)
GL_DECL(void,    glUniformMatrix4fv,     GLint,GLsizei,GLboolean,const GLfloat*)
GL_DECL(void,    glEnableVertexAttribArray, GLuint)
GL_DECL(void,    glDisableVertexAttribArray, GLuint)
GL_DECL(void,    glVertexAttribPointer,  GLuint,GLint,GLenum,GLboolean,GLsizei,const void*)
GL_DECL(void,    glGetVertexAttribiv,    GLuint,GLenum,GLint*)
GL_DECL(void,    glGetVertexAttribPointerv, GLuint,GLenum,void**)
GL_DECL(void,    glDrawArrays,           GLenum,GLint,GLsizei)
GL_DECL(void,    glDrawElements,         GLenum,GLsizei,GLenum,const void*)
GL_DECL(void,    glStencilFunc,          GLenum,GLint,GLuint)
GL_DECL(void,    glStencilFuncSeparate,  GLenum,GLenum,GLint,GLuint)
GL_DECL(void,    glStencilMask,          GLuint)
GL_DECL(void,    glStencilOp,            GLenum,GLenum,GLenum)
GL_DECL(void,    glStencilOpSeparate,    GLenum,GLenum,GLenum,GLenum)
GL_DECL(void,    glBlendFunc,            GLenum,GLenum)
GL_DECL(void,    glTexParameterf,        GLenum,GLenum,GLfloat)
GL_DECL(void,    glDepthRangef,          GLclampf,GLclampf)
GL_DECL(void,    glPolygonOffset,        GLfloat,GLfloat)
GL_DECL(void,    glLineWidth,            GLfloat)
GL_DECL(void,    glSampleCoverage,       GLclampf,GLboolean)
/* Scalar uniforms */
GL_DECL(void,    glUniform1f,            GLint,GLfloat)
GL_DECL(void,    glUniform2f,            GLint,GLfloat,GLfloat)
GL_DECL(void,    glUniform3f,            GLint,GLfloat,GLfloat,GLfloat)
GL_DECL(void,    glUniform4f,            GLint,GLfloat,GLfloat,GLfloat,GLfloat)

GL_DECL(void,    glVertexAttrib1f,       GLuint,GLfloat)
GL_DECL(void,    glVertexAttrib2f,       GLuint,GLfloat,GLfloat)
GL_DECL(void,    glVertexAttrib3f,       GLuint,GLfloat,GLfloat,GLfloat)
GL_DECL(void,    glVertexAttrib4f,       GLuint,GLfloat,GLfloat,GLfloat,GLfloat)
GL_DECL(void,    glVertexAttrib4fv,      GLuint,const GLfloat*)
GL_DECL(void,    glVertexAttrib1fv,      GLuint,const GLfloat*)
GL_DECL(void,    glVertexAttrib2fv,      GLuint,const GLfloat*)
GL_DECL(void,    glVertexAttrib3fv,      GLuint,const GLfloat*)

GL_DECL(void,    glGetFloatv,            GLenum,GLfloat*)
GL_DECL(void,    glGetBooleanv,          GLenum,GLboolean*)
GL_DECL(GLboolean, glIsEnabled,          GLenum)
GL_DECL(GLboolean, glIsProgram,          GLuint)
GL_DECL(GLboolean, glIsShader,           GLuint)
GL_DECL(GLboolean, glIsTexture,          GLuint)
GL_DECL(GLboolean, glIsBuffer,           GLuint)
GL_DECL(GLboolean, glIsFramebuffer,      GLuint)
GL_DECL(GLboolean, glIsRenderbuffer,     GLuint)

GL_DECL(void,    glBlendEquation,        GLenum)
GL_DECL(void,    glBlendColor,           GLclampf,GLclampf,GLclampf,GLclampf)
GL_DECL(void,    glReleaseShaderCompiler, void)
GL_DECL(void,    glGetShaderPrecisionFormat, GLenum,GLenum,GLint*,GLint*)
GL_DECL(void,    glUniformMatrix2fv,     GLint,GLsizei,GLboolean,const GLfloat*)

/* GLES 3.x (GLsync/GLuint64 aren't in GLES2/gl2.h; use ABI equivalents) */
GL_DECL(const GLubyte*, glGetStringi,    GLenum,GLuint)
GL_DECL(void,    glGetIntegeri_v,        GLenum,GLuint,GLint*)
GL_DECL(void,    glGetInternalformativ,  GLenum,GLenum,GLenum,GLsizei,GLint*)
GL_DECL(void,    glGetProgramInterfaceiv, GLuint,GLenum,GLenum,GLint*)
GL_DECL(void,    glGetProgramResourceiv, GLuint,GLenum,GLuint,GLsizei,const GLenum*,GLsizei,GLsizei*,GLint*)
GL_DECL(void,    glGetProgramResourceName, GLuint,GLenum,GLuint,GLsizei,GLsizei*,GLchar*)
GL_DECL(void,    glGenVertexArrays,      GLsizei,GLuint*)
GL_DECL(void,    glBindVertexArray,      GLuint)
GL_DECL(void,    glDeleteVertexArrays,   GLsizei,const GLuint*)
GL_DECL(GLboolean, glIsVertexArray,      GLuint)
GL_DECL(void,    glBindSampler,          GLuint,GLuint)
GL_DECL(void,    glBindBufferBase,       GLenum,GLuint,GLuint)
GL_DECL(void,    glBindBufferRange,      GLenum,GLuint,GLuint,intptr_t,intptr_t)
GL_DECL(void*,   glMapBufferRange,       GLenum,intptr_t,intptr_t,GLbitfield)
GL_DECL(GLboolean, glUnmapBuffer,        GLenum)
GL_DECL(void,    glFlushMappedBufferRange, GLenum,intptr_t,intptr_t)
GL_DECL(void,    glTexStorage2D,         GLenum,GLsizei,GLenum,GLsizei,GLsizei)
GL_DECL(void,    glTexStorage3D,         GLenum,GLsizei,GLenum,GLsizei,GLsizei,GLsizei)
GL_DECL(void,    glTexSubImage3D,        GLenum,GLint,GLint,GLint,GLint,GLsizei,GLsizei,GLsizei,GLenum,GLenum,const void*)
GL_DECL(void,    glProgramParameteri,    GLuint,GLenum,GLint)
GL_DECL(void,    glGetProgramBinary,     GLuint,GLsizei,GLsizei*,GLenum*,void*)
GL_DECL(void,    glProgramBinary,        GLuint,GLenum,const void*,GLsizei)
GL_DECL(void*,   glFenceSync,            GLenum,GLbitfield)
GL_DECL(GLenum,  glClientWaitSync,       void*,GLbitfield,uint64_t)
GL_DECL(void,    glDeleteSync,           void*)
GL_DECL(void,    glInvalidateFramebuffer, GLenum,GLsizei,const GLenum*)
GL_DECL(void,    glDetachShader,         GLuint,GLuint)
GL_DECL(void,    glDrawBuffers,          GLsizei,const GLenum*)

static void load_gl_procs() {
    g_libgles2 = dlopen("libGLESv2.so.2", RTLD_LAZY | RTLD_GLOBAL);
    if (!g_libgles2) g_libgles2 = dlopen("libGLESv2.so", RTLD_LAZY | RTLD_GLOBAL);
    GL_LOAD(glViewport); GL_LOAD(glClear); GL_LOAD(glClearColor);
    GL_LOAD(glClearDepthf); GL_LOAD(glClearStencil);
    GL_LOAD(glEnable); GL_LOAD(glDisable);
    GL_LOAD(glDepthFunc); GL_LOAD(glDepthMask); GL_LOAD(glColorMask);
    GL_LOAD(glScissor); GL_LOAD(glFrontFace); GL_LOAD(glCullFace);
    GL_LOAD(glBlendFuncSeparate); GL_LOAD(glBlendEquationSeparate);
    GL_LOAD(glGetError); GL_LOAD(glGetString); GL_LOAD(glGetIntegerv);
    GL_LOAD(glPixelStorei); GL_LOAD(glReadPixels);
    GL_LOAD(glFlush); GL_LOAD(glFinish);
    GL_LOAD(glGenBuffers); GL_LOAD(glBindBuffer); GL_LOAD(glBufferData);
    GL_LOAD(glBufferSubData); GL_LOAD(glDeleteBuffers);
    GL_LOAD(glGenTextures); GL_LOAD(glBindTexture); GL_LOAD(glActiveTexture);
    GL_LOAD(glDeleteTextures); GL_LOAD(glTexParameteri);
    GL_LOAD(glTexImage2D); GL_LOAD(glTexSubImage2D); GL_LOAD(glCopyTexSubImage2D);
    GL_LOAD(glCompressedTexImage2D); GL_LOAD(glCompressedTexSubImage2D);
    GL_LOAD(glGenerateMipmap);
    GL_LOAD(glGenFramebuffers); GL_LOAD(glBindFramebuffer);
    GL_LOAD(glDeleteFramebuffers); GL_LOAD(glCheckFramebufferStatus);
    GL_LOAD(glFramebufferTexture2D); GL_LOAD(glFramebufferRenderbuffer);
    GL_LOAD(glGetFramebufferAttachmentParameteriv);
    GL_LOAD(glGenRenderbuffers); GL_LOAD(glBindRenderbuffer);
    GL_LOAD(glDeleteRenderbuffers); GL_LOAD(glRenderbufferStorage);
    GL_LOAD(glCreateShader); GL_LOAD(glShaderSource); GL_LOAD(glCompileShader);
    GL_LOAD(glDeleteShader); GL_LOAD(glGetShaderiv); GL_LOAD(glGetShaderInfoLog);
    GL_LOAD(glGetShaderSource);
    GL_LOAD(glCreateProgram); GL_LOAD(glAttachShader); GL_LOAD(glLinkProgram);
    GL_LOAD(glUseProgram); GL_LOAD(glDeleteProgram);
    GL_LOAD(glGetProgramiv); GL_LOAD(glGetProgramInfoLog);
    GL_LOAD(glGetAttribLocation); GL_LOAD(glGetUniformLocation);
    GL_LOAD(glGetActiveAttrib); GL_LOAD(glGetActiveUniform);
    GL_LOAD(glBindAttribLocation);
    GL_LOAD(glUniform1i); GL_LOAD(glUniform1iv); GL_LOAD(glUniform2iv);
    GL_LOAD(glUniform3iv); GL_LOAD(glUniform4iv);
    GL_LOAD(glUniform1fv); GL_LOAD(glUniform2fv); GL_LOAD(glUniform3fv);
    GL_LOAD(glUniform4fv); GL_LOAD(glUniformMatrix3fv); GL_LOAD(glUniformMatrix4fv);
    GL_LOAD(glEnableVertexAttribArray); GL_LOAD(glDisableVertexAttribArray);
    GL_LOAD(glVertexAttribPointer); GL_LOAD(glGetVertexAttribiv);
    GL_LOAD(glGetVertexAttribPointerv);
    GL_LOAD(glDrawArrays); GL_LOAD(glDrawElements);
    GL_LOAD(glStencilFunc); GL_LOAD(glStencilFuncSeparate);
    GL_LOAD(glStencilMask); GL_LOAD(glStencilOp); GL_LOAD(glStencilOpSeparate);
    GL_LOAD(glBlendFunc); GL_LOAD(glTexParameterf);
    GL_LOAD(glDepthRangef); GL_LOAD(glPolygonOffset);
    GL_LOAD(glLineWidth); GL_LOAD(glSampleCoverage);
    GL_LOAD(glUniform1f); GL_LOAD(glUniform2f); GL_LOAD(glUniform3f); GL_LOAD(glUniform4f);
    GL_LOAD(glVertexAttrib1f); GL_LOAD(glVertexAttrib2f); GL_LOAD(glVertexAttrib3f);
    GL_LOAD(glVertexAttrib4f); GL_LOAD(glVertexAttrib4fv);
    GL_LOAD(glVertexAttrib1fv); GL_LOAD(glVertexAttrib2fv); GL_LOAD(glVertexAttrib3fv);
    GL_LOAD(glGetFloatv); GL_LOAD(glGetBooleanv);
    GL_LOAD(glIsEnabled); GL_LOAD(glIsProgram); GL_LOAD(glIsShader); GL_LOAD(glIsTexture);
    GL_LOAD(glIsBuffer); GL_LOAD(glIsFramebuffer); GL_LOAD(glIsRenderbuffer);
    GL_LOAD(glBlendEquation); GL_LOAD(glBlendColor);
    GL_LOAD(glReleaseShaderCompiler); GL_LOAD(glGetShaderPrecisionFormat);
    GL_LOAD(glUniformMatrix2fv);
    GL_LOAD(glGetStringi); GL_LOAD(glGetIntegeri_v); GL_LOAD(glGetInternalformativ);
    GL_LOAD(glGetProgramInterfaceiv); GL_LOAD(glGetProgramResourceiv);
    GL_LOAD(glGetProgramResourceName);
    GL_LOAD(glGenVertexArrays); GL_LOAD(glBindVertexArray);
    GL_LOAD(glDeleteVertexArrays); GL_LOAD(glIsVertexArray);
    GL_LOAD(glBindSampler); GL_LOAD(glBindBufferBase); GL_LOAD(glBindBufferRange);
    GL_LOAD(glMapBufferRange); GL_LOAD(glUnmapBuffer); GL_LOAD(glFlushMappedBufferRange);
    GL_LOAD(glTexStorage2D); GL_LOAD(glTexStorage3D); GL_LOAD(glTexSubImage3D);
    GL_LOAD(glProgramParameteri); GL_LOAD(glGetProgramBinary); GL_LOAD(glProgramBinary);
    GL_LOAD(glFenceSync); GL_LOAD(glClientWaitSync); GL_LOAD(glDeleteSync);
    GL_LOAD(glInvalidateFramebuffer); GL_LOAD(glDetachShader); GL_LOAD(glDrawBuffers);
    fprintf(stderr, "[arm_exec] GL procs loaded (%s, GLES3 %s)\n",
            pfn_glCreateProgram ? "ok" : "FAILED",
            pfn_glGetProgramInterfaceiv ? "ok" : "missing");
}

/* ---- Touch input: GLFW mouse → Android MotionEvent bridge -----------------
 * The loader polls arm_exec_touch_next() each frame and calls Unity's
 * nativeInjectEvent with a MotionEvent stub object; the MotionEvent JNI
 * getters in libjvm-android.c read the "current" event via the accessors
 * below.  Android MotionEvent actions: 0=DOWN 1=UP 2=MOVE. */
struct TouchEvent { int action; float x, y; };
static std::deque<TouchEvent> g_touch_queue;
static TouchEvent g_touch_cur = {1, 0.f, 0.f};   /* current (last-popped) */
static bool g_touch_down = false;

static void glfw_mouse_button_cb(GLFWwindow *w, int button, int action, int) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    double cx = 0, cy = 0;
    glfwGetCursorPos(w, &cx, &cy);
    g_touch_down = (action == GLFW_PRESS);
    if (g_touch_queue.size() < 64)
        g_touch_queue.push_back({g_touch_down ? 0 : 1, (float)cx, (float)cy});
}

static void glfw_cursor_pos_cb(GLFWwindow *, double cx, double cy) {
    if (!g_touch_down) return;   /* touchscreen semantics: no hover events */
    /* Coalesce: replace a pending MOVE instead of queueing thousands */
    if (!g_touch_queue.empty() && g_touch_queue.back().action == 2)
        g_touch_queue.back() = {2, (float)cx, (float)cy};
    else if (g_touch_queue.size() < 64)
        g_touch_queue.push_back({2, (float)cx, (float)cy});
}

extern "C" int arm_exec_touch_next(void) {
    if (g_touch_queue.empty()) return 0;
    g_touch_cur = g_touch_queue.front();
    g_touch_queue.pop_front();
    return 1;
}

/* Programmatic event injection (LUNARIA_TOUCH_TEST) — lets the loader tap without
 * a real mouse, so tests don't depend on X11 focus. */
extern "C" void arm_exec_touch_push(int action, float x, float y) {
    if (g_touch_queue.size() < 64)
        g_touch_queue.push_back({action, x, y});
}
extern "C" int   arm_exec_touch_action(void) { return g_touch_cur.action; }
extern "C" float arm_exec_touch_x(void)      { return g_touch_cur.x; }
extern "C" float arm_exec_touch_y(void)      { return g_touch_cur.y; }
extern "C" long long arm_exec_touch_time(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000ll + ts.tv_nsec / 1000000ll;
}

/* Ensure GLFW window exists (called lazily) */
static bool ensure_glfw_window() {
    if (g_glfw) return true;
    fprintf(stderr, "[arm_exec] creating GLFW window (1280×720)...\n");
    if (!glfwInit()) {
        fprintf(stderr, "[arm_exec] glfwInit failed\n");
        return false;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    g_glfw = glfwCreateWindow(1280, 720, "Lunaria Unity", nullptr, nullptr);
    if (!g_glfw) fprintf(stderr, "[arm_exec] glfwCreateWindow failed\n");
    else {
        fprintf(stderr, "[arm_exec] GLFW window created\n");
        glfwSetMouseButtonCallback(g_glfw, glfw_mouse_button_cb);
        glfwSetCursorPosCallback(g_glfw, glfw_cursor_pos_cb);
    }
    return g_glfw != nullptr;
}

/* Guest ARM stubs for ANativeWindow_Ops (incRef..perform).  Unity's surf_chk
 * calls ops+0xc (dequeueBuffer) and ops+0x18 (query) via function pointers. */
static constexpr uint32_t ANW_OPS_STUB = 0x4100b800u;

static void init_anw_op_stubs(ArmExecCtx &ctx,
                              uint32_t &ret0_va, uint32_t &query_va, uint32_t &dequeue_va) {
    if (ret0_va) return;
    ctx.mem.map(ANW_OPS_STUB, 128);
    uint32_t p = ANW_OPS_STUB;
    ret0_va = p;
    ctx.mem.write32(p,     0xE3A00000u); /* mov r0, #0 */
    ctx.mem.write32(p + 4, 0xE12FFF1Eu); /* bx lr */
    p += 8;
    /* query(window, what, value*): WIDTH=4 HEIGHT=5 MIN_UNDEQUEUED=3 */
    query_va = p;
    const uint32_t query[] = {
        0xE3520000u, /* cmp r2, #0 */
        0x0A000006u, /* beq fail */
        0xE5903008u, /* ldr r3, [r0, #8]  default width */
        0xE3510005u, /* cmp r1, #5 */
        0x0590300Cu, /* ldreq r3, [r0, #12] height */
        0xE3510003u, /* cmp r1, #3 */
        0x13A03001u, /* moveq r3, #1 */
        0xE5823000u, /* str r3, [r2] */
        0xE3A00000u, /* mov r0, #0 */
        0xE12FFF1Eu, /* bx lr */
        0xE3E00015u, /* fail: mvn r0, #21  (-22 BAD_VALUE) */
        0xE12FFF1Eu,
    };
    for (uint32_t w : query) { ctx.mem.write32(p, w); p += 4; }
    /* dequeueBuffer(window, buffer*, fence*): fill ANativeWindow_Buffer */
    dequeue_va = p;
    const uint32_t dequeue[] = {
        0xE3510000u, /* cmp r1, #0 */
        0x0A000009u, /* beq fail */
        0xE5902008u, /* ldr r2, [r0, #8]  width */
        0xE5812000u, /* str r2, [r1] */
        0xE5902012u, /* ldr r2, [r0, #12] height */
        0xE5812014u, /* str r2, [r1, #4] */
        0xE5812008u, /* str r2, [r1, #8] stride = width */
        0xE3A02001u, /* mov r2, #1  RGBA_8888 */
        0xE5812010u, /* str r2, [r1, #0xc] format */
        0xE3A00000u, /* mov r0, #0 */
        0xE12FFF1Eu,
        0xE3E00015u, /* fail */
        0xE12FFF1Eu,
    };
    for (uint32_t w : dequeue) { ctx.mem.write32(p, w); p += 4; }
}

/* Unity may call ANativeWindow_* via PLT or dereference window->ops inline. */
static uint32_t ensure_fake_anative_window(ArmExecCtx &ctx) {
    if (g_fake_anw_va) return g_fake_anw_va;
    const uint32_t win_va = arm_malloc(ctx, 64);
    const uint32_t ops_va = arm_malloc(ctx, 64);
    if (!win_va || !ops_va) return 0;
    uint32_t ret0 = 0, query = 0, dequeue = 0;
    init_anw_op_stubs(ctx, ret0, query, dequeue);
    ctx.mem.write32(win_va, ops_va);
    ctx.mem.write32(ops_va + 0u,  ret0);
    ctx.mem.write32(ops_va + 4u,  ret0);    /* decRef */
    ctx.mem.write32(ops_va + 8u,  ret0);    /* setSwapInterval */
    ctx.mem.write32(ops_va + 12u, dequeue); /* dequeueBuffer */
    ctx.mem.write32(ops_va + 16u, ret0);    /* cancelBuffer */
    ctx.mem.write32(ops_va + 20u, ret0);    /* queueBuffer */
    ctx.mem.write32(ops_va + 24u, query);   /* query */
    ctx.mem.write32(ops_va + 28u, ret0);    /* perform */
    ctx.mem.write32(win_va + 8u, 1280u);
    ctx.mem.write32(win_va + 12u, 720u);
    g_fake_anw_va = win_va;
    fprintf(stderr, "[arm_exec] fake ANativeWindow @%#x ops@%#x (query=%#x dequeue=%#x)\n",
            win_va, ops_va, query, dequeue);
    return win_va;
}

/* Initialize host-side EGL + GLES2 context and make it current.
 * Called once before Unity's nativeRecreateGfxState so that all subsequent
 * GL calls from Unity's ARM code use a real GLES2 context.
 * Falls back to a pbuffer surface when no X11 display is available (headless). */
static bool init_host_egl() {
    if (g_egl_ctx != EGL_NO_CONTEXT) {
        /* Already created — just make current */
        return eglMakeCurrent(g_egl_dpy, g_egl_surf, g_egl_surf, g_egl_ctx) == EGL_TRUE;
    }

    /* Try GLFW window first (requires DISPLAY); fall back to headless EGL */
    bool has_window = ensure_glfw_window();

    Display *x11dpy = has_window ? glfwGetX11Display() : nullptr;
    EGLint major = 0, minor = 0;

    /* Headless first: the surfaceless Mesa platform gives a working GLES
 * context (HW radeonsi or llvmpipe SW) without needing an X11/GLX server.
 * This is the reliable path under Xvfb where GLFW/GLX init fails. */
    if (!has_window) {
        PFNEGLGETPLATFORMDISPLAYEXTPROC getPlatformDisplay =
            (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
        if (getPlatformDisplay) {
            g_egl_dpy = getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,
                                           EGL_DEFAULT_DISPLAY, nullptr);
            if (g_egl_dpy != EGL_NO_DISPLAY && eglInitialize(g_egl_dpy, &major, &minor)) {
                fprintf(stderr, "[arm_exec] init_host_egl: surfaceless EGL %d.%d "
                        "(%s)\n", major, minor, eglQueryString(g_egl_dpy, EGL_VENDOR));
            } else {
                g_egl_dpy = EGL_NO_DISPLAY; /* fall through to default display */
            }
        }
    }

    if (g_egl_dpy == EGL_NO_DISPLAY) {
        g_egl_dpy = eglGetDisplay(x11dpy ? (EGLNativeDisplayType)x11dpy : EGL_DEFAULT_DISPLAY);
        if (g_egl_dpy == EGL_NO_DISPLAY) {
            fprintf(stderr, "[arm_exec] init_host_egl: eglGetDisplay failed\n");
            return false;
        }
        if (!eglInitialize(g_egl_dpy, &major, &minor)) {
            fprintf(stderr, "[arm_exec] init_host_egl: eglInitialize failed\n");
            return false;
        }
    }

    /* Bind GLES API explicitly: eglCreateContext otherwise targets the
 * currently-bound API (desktop GL by default on some drivers). */
    eglBindAPI(EGL_OPENGL_ES_API);

    /* Surfaceless displays expose only PBUFFER configs (no WINDOW_BIT), so
 * requesting WINDOW_BIT there makes eglChooseConfig return zero configs.
 * Only ask for WINDOW_BIT when we actually have an X11 window to bind. */
    const EGLint cfg_attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_SURFACE_TYPE,    (has_window && g_glfw) ? (EGL_WINDOW_BIT | EGL_PBUFFER_BIT)
                                                    : EGL_PBUFFER_BIT,
        EGL_RED_SIZE,   8, EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE,  8, EGL_DEPTH_SIZE, 16,
        EGL_NONE
    };
    EGLint num = 0;
    if (!eglChooseConfig(g_egl_dpy, cfg_attribs, &g_egl_cfg, 1, &num) || num == 0) {
        fprintf(stderr, "[arm_exec] init_host_egl: eglChooseConfig failed\n");
        return false;
    }

    /* Window surface path (X11 display available) */
    if (has_window && g_glfw) {
        Window x11win = glfwGetX11Window(g_glfw);
        g_egl_surf = eglCreateWindowSurface(g_egl_dpy, g_egl_cfg,
                                             (EGLNativeWindowType)x11win, nullptr);
        if (g_egl_surf == EGL_NO_SURFACE)
            fprintf(stderr, "[arm_exec] init_host_egl: eglCreateWindowSurface failed (err=0x%x)"
                    " — falling back to pbuffer\n", eglGetError());
    }

    /* Pbuffer fallback: headless mode or window surface creation failed */
    if (g_egl_surf == EGL_NO_SURFACE) {
        static const EGLint pb_attribs[] = {
            EGL_WIDTH, 1280, EGL_HEIGHT, 720, EGL_NONE
        };
        g_egl_surf = eglCreatePbufferSurface(g_egl_dpy, g_egl_cfg, pb_attribs);
        if (g_egl_surf == EGL_NO_SURFACE) {
            fprintf(stderr, "[arm_exec] init_host_egl: pbuffer also failed (err=0x%x)\n",
                    eglGetError());
            return false;
        }
        fprintf(stderr, "[arm_exec] init_host_egl: using 1280x720 pbuffer (headless mode)\n");
    }

    static const EGLint ctx_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    g_egl_ctx = eglCreateContext(g_egl_dpy, g_egl_cfg, EGL_NO_CONTEXT, ctx_attribs);
    if (g_egl_ctx == EGL_NO_CONTEXT) {
        fprintf(stderr, "[arm_exec] init_host_egl: eglCreateContext failed (err=0x%x)\n",
                eglGetError());
        return false;
    }
    EGLBoolean ok = eglMakeCurrent(g_egl_dpy, g_egl_surf, g_egl_surf, g_egl_ctx);
    fprintf(stderr, "[arm_exec] init_host_egl: EGL %d.%d ready, makeCurrent=%d\n",
            major, minor, ok);
    if (ok) load_gl_procs();
    return ok == EGL_TRUE;
}

/* Read EGL attrib list from ARM memory into a host vector */
static std::vector<EGLint> read_egl_attribs(ArmExecCtx &ctx, uint32_t va) {
    std::vector<EGLint> out;
    if (!va) { out.push_back(EGL_NONE); return out; }
    for (int i = 0; i < 64; ++i) {
        auto *p = ctx.mem.ptr(va + (uint32_t)(i * 4));
        if (!p) break;
        EGLint v; memcpy(&v, p, 4);
        out.push_back(v);
        if (v == EGL_NONE) break;
    }
    if (out.empty() || out.back() != EGL_NONE) out.push_back(EGL_NONE);
    return out;
}

/* Run runnable guest threads for a tick-budgeted slice on the auxiliary JIT.
 * Safe to call from inside an SVC handler (main JIT busy).  Defined after
 * ArmCallbacks below. */
static void schedule_threads(uint64_t slice);
static uint64_t env_ticks(const char *name, uint64_t def);
static bool g_scheduling = false;
/* Run a short guest function fn_va(a,b) on the dedicated callback JIT and
 * return its r0.  Defined after ArmCallbacks/schedule_threads; forward-declared
 * here so dispatch_svc's bsearch handler can drive the guest comparator. */
static int32_t call_guest_cb(ArmExecCtx &ctx, uint32_t fn_va, uint32_t a, uint32_t b,
                             uint32_t c, uint32_t d,
                             const uint32_t *stk, int nstk);

/* Called from wait-flavoured SVCs (cond_wait/sem_wait/usleep/yield): give
 * worker threads CPU time so the predicate the caller spins on can change. */
static void maybe_schedule_on_wait() {
    if (g_threads.empty()) return;
    /* Called from nanosleep/usleep/cond_wait; give worker threads enough ticks */
    schedule_threads(20'000'000ULL);
}

/* -------------------------------------------------------------------------
 * JNI AssetManager bridge
 * Unity reads every player asset (boot.config, globalgamemanagers, level*,
 * .assets.split*, …) through the Java side: getAssets().open(path) returns a
 * java.io.InputStream that native code drains with InputStream.read(byte[]).
 * boot.config is read via a Scanner.  None of that reaches host code unless we
 * (a) marshal the call arguments out of the ARM registers/stack (the generic
 * JNI dispatch drops them) and (b) actually serve the bytes from the APK.
 * This bridge reads entries straight out of the APK zip (STORE + DEFLATE) and
 * keeps per-InputStream cursors host-side, keyed by the jobject handle.
 * ---------------------------------------------------------------------- */
namespace asset_bridge {

struct ZipEntry { uint32_t lho, comp, uncomp; uint16_t method; };
static std::map<std::string, ZipEntry> g_dir;
static FILE *g_fp = nullptr;
static bool  g_tried = false;
/* lunaria-apk.sh extracts the APK and points ANDROID_PACKAGE_CODE_PATH at the
 * extraction *directory* rather than the .apk file.  In that case there is no
 * zip to index — assets are plain files under <dir>/ — so the bridge serves
 * reads straight from disk.  g_apk_dir is set (non-empty) only in this mode. */
static std::string g_apk_dir;

static bool apk_open() {
    if (g_tried) return g_fp != nullptr || !g_apk_dir.empty();
    g_tried = true;
    const char *path = getenv("ANDROID_PACKAGE_CODE_PATH");
    if (!path) { fprintf(stderr, "[asset] ANDROID_PACKAGE_CODE_PATH unset\n"); return false; }
    /* Directory mode: the APK was already unzipped to <path>/. */
    struct stat sb;
    if (stat(path, &sb) == 0 && S_ISDIR(sb.st_mode)) {
        g_apk_dir = path;
        fprintf(stderr, "[asset] APK code path is a directory: %s (disk mode)\n", path);
        return true;
    }
    if (!(g_fp = fopen(path, "rb"))) { fprintf(stderr, "[asset] cannot open APK %s\n", path); return false; }
    fseek(g_fp, 0, SEEK_END);
    long fsz = ftell(g_fp);
    long tail = fsz < 65557 ? fsz : 65557;
    std::vector<uint8_t> b(tail);
    fseek(g_fp, fsz - tail, SEEK_SET);
    if ((long)fread(b.data(), 1, tail, g_fp) != tail) return false;
    long e = -1;
    for (long i = tail - 22; i >= 0; --i)
        if (b[i]==0x50 && b[i+1]==0x4b && b[i+2]==0x05 && b[i+3]==0x06) { e = i; break; }
    if (e < 0) { fprintf(stderr, "[asset] zip EOCD not found\n"); return false; }
    auto rd16 = [&](long o){ return (uint32_t)(b[e+o] | (b[e+o+1]<<8)); };
    auto rd32 = [&](long o){ return (uint32_t)(b[e+o] | (b[e+o+1]<<8) | (b[e+o+2]<<16) | ((uint32_t)b[e+o+3]<<24)); };
    uint32_t count = rd16(10);
    uint32_t cdoff = rd32(16);
    fseek(g_fp, cdoff, SEEK_SET);
    for (uint32_t i = 0; i < count; ++i) {
        uint8_t h[46];
        if (fread(h, 1, 46, g_fp) != 46) break;
        if (!(h[0]==0x50 && h[1]==0x4b && h[2]==0x01 && h[3]==0x02)) break;
        auto h16 = [&](int o){ return (uint16_t)(h[o] | (h[o+1]<<8)); };
        auto h32 = [&](int o){ return (uint32_t)(h[o] | (h[o+1]<<8) | (h[o+2]<<16) | ((uint32_t)h[o+3]<<24)); };
        ZipEntry ze; ze.method = h16(10); ze.comp = h32(20); ze.uncomp = h32(24); ze.lho = h32(42);
        uint16_t nlen = h16(28), elen = h16(30), clen = h16(32);
        std::string name(nlen, '\0');
        if (nlen && fread(&name[0], 1, nlen, g_fp) != nlen) break;
        if (elen + clen) fseek(g_fp, elen + clen, SEEK_CUR);
        g_dir.emplace(std::move(name), ze);
    }
    fprintf(stderr, "[asset] APK indexed: %zu entries\n", g_dir.size());
    return true;
}

static bool apk_read(const std::string &name, std::vector<uint8_t> &out) {
    if (!apk_open()) return false;
    if (!g_apk_dir.empty()) {
        std::string fp = g_apk_dir + "/" + name;
        /* Reject directories: fopen() of a directory succeeds on Linux but
 * yields 0 bytes, which would otherwise masquerade as an empty file.
 * That made apk_extract_to_cache() materialise e.g. bin/Data/Managed as
 * an empty regular file instead of letting the stat() handler report it
 * as a directory — breaking Unity's IL2CPP resource extraction. */
        struct stat sb;
        if (stat(fp.c_str(), &sb) != 0 || !S_ISREG(sb.st_mode)) return false;
        FILE *f = fopen(fp.c_str(), "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz < 0) { fclose(f); return false; }
        out.resize((size_t)sz);
        bool ok = (sz == 0) || (fread(out.data(), 1, (size_t)sz, f) == (size_t)sz);
        fclose(f);
        return ok;
    }
    auto it = g_dir.find(name);
    if (it == g_dir.end()) return false;
    ZipEntry &e = it->second;
    uint8_t lh[30];
    fseek(g_fp, e.lho, SEEK_SET);
    if (fread(lh, 1, 30, g_fp) != 30) return false;
    if (!(lh[0]==0x50 && lh[1]==0x4b && lh[2]==0x03 && lh[3]==0x04)) return false;
    uint16_t nlen = lh[26] | (lh[27]<<8), elen = lh[28] | (lh[29]<<8);
    fseek(g_fp, (long)e.lho + 30 + nlen + elen, SEEK_SET);
    std::vector<uint8_t> comp(e.comp);
    if (e.comp && fread(comp.data(), 1, e.comp, g_fp) != e.comp) return false;
    if (e.method == 0) { out = std::move(comp); return true; }
    if (e.method == 8) {
        out.resize(e.uncomp);
        z_stream zs{};
        if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) return false;
        zs.next_in = comp.data(); zs.avail_in = e.comp;
        zs.next_out = out.data(); zs.avail_out = e.uncomp;
        int r = inflate(&zs, Z_FINISH);
        inflateEnd(&zs);
        if (r != Z_STREAM_END) { fprintf(stderr, "[asset] inflate failed for %s (r=%d)\n", name.c_str(), r); return false; }
        return true;
    }
    fprintf(stderr, "[asset] unsupported compression %u for %s\n", e.method, name.c_str());
    return false;
}

/* True if `name` is a directory prefix of some zip entry (e.g. the APK has
 * "base/assets/bin/Data/Managed/Metadata/global-metadata.dat" so "base/assets/
 * bin/Data/Managed" is a directory). */
static bool apk_is_dir(const std::string &name) {
    if (!apk_open()) return false;
    if (name.empty()) return true;
    if (!g_apk_dir.empty()) {
        struct stat sb;
        std::string fp = g_apk_dir + "/" + name;
        return stat(fp.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
    }
    std::string pfx = name;
    if (pfx.back() != '/') pfx.push_back('/');
    auto it = g_dir.lower_bound(pfx);
    return it != g_dir.end() && it->first.compare(0, pfx.size(), pfx) == 0;
}

/* List the immediate children of an APK directory prefix.  Each entry is
 * (name, is_dir).  Used to back opendir()/readdir() over the APK zip so
 * Unity's IL2CPP installer can enumerate assets/bin/Data/Managed/. */
static void apk_list_dir(const std::string &name,
                         std::vector<std::pair<std::string,bool>> &out) {
    if (!apk_open()) return;
    if (!g_apk_dir.empty()) {
        std::string dp = g_apk_dir + "/" + name;
        DIR *d = opendir(dp.c_str());
        if (!d) return;
        struct dirent *de;
        while ((de = readdir(d))) {
            if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
            struct stat sb;
            std::string fp = dp + "/" + de->d_name;
            bool is_dir = (stat(fp.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode));
            out.emplace_back(de->d_name, is_dir);
        }
        closedir(d);
        return;
    }
    std::string pfx = name;
    if (!pfx.empty() && pfx.back() != '/') pfx.push_back('/');
    std::map<std::string,bool> seen;
    for (auto it = g_dir.lower_bound(pfx); it != g_dir.end(); ++it) {
        const std::string &e = it->first;
        if (e.compare(0, pfx.size(), pfx) != 0) break;
        std::string rest = e.substr(pfx.size());
        if (rest.empty()) continue;
        size_t slash = rest.find('/');
        bool is_dir = (slash != std::string::npos);
        std::string child = is_dir ? rest.substr(0, slash) : rest;
        if (child.empty()) continue;
        auto r = seen.emplace(child, is_dir);
        if (r.second) out.emplace_back(child, is_dir);
    }
}

struct Stream { std::vector<uint8_t> data; size_t pos = 0; };
static std::map<uint32_t, Stream>   g_streams;        /* InputStream handle -> data */
static std::map<uint32_t, uint32_t> g_scanner2stream; /* Scanner handle -> InputStream handle */
static uint32_t g_last_stream = 0;                    /* most recently opened stream */

/* Count JNI signature arguments, e.g. "([BII)I" -> 3, "()Z" -> 0. */
static int sig_argc(const char *sig) {
    if (!sig) return -1;
    const char *p = strchr(sig, '(');
    if (!p) return 0;
    ++p;
    int n = 0;
    while (*p && *p != ')') {
        if (*p == '[') { ++p; continue; }
        if (*p == 'L') { while (*p && *p != ';') ++p; if (*p) ++p; ++n; continue; }
        ++p; ++n;
    }
    return n;
}

} // namespace asset_bridge

/* NDK AAsset table (outside asset_bridge namespace) */
struct GuestAsset { std::vector<uint8_t> data; uint32_t buf_va; };
static std::map<uint32_t, GuestAsset> g_assets;
static uint32_t g_next_asset_handle = 0xA55E7000u;
static uint32_t g_lconv_va          = 0u;

/* Materialize guest "<apk>/assets/..." paths from the APK zip into host cache */
static bool apk_extract_to_cache(const char *path, char *out, size_t outsz) {
    const char *apk = getenv("ANDROID_PACKAGE_CODE_PATH");
    if (!path) return false;
    const char *inner;
    size_t alen = apk ? strlen(apk) : 0;
    const char *apk_suffix = strstr(path, ".apk/");
    if (apk && strncmp(path, apk, alen) == 0 && path[alen] == '/') {
        inner = path + alen + 1;                  /* absolute path: <code_path>/assets/... */
    } else if (apk_suffix) {
        /* Android semantics: resolve ".apk/assets/..." as zip entries inside the APK */
        inner = apk_suffix + 5;
    } else if (strncmp(path, "assets/", 7) == 0) {
        /* Unity also opens asset files by a path relative to the (Android) data
 * root, e.g. open("assets/bin/Data/unity_app_guid").  The CWD here is
 * the Lunaria dir, so that misses on disk; resolve it against the APK too —
 * otherwise unity_app_guid reads empty and Unity loops "Will re-extract
 * il2cpp resources on next run." every frame. */
        inner = path;
    } else if (path[0] == '/' && strncmp(path + 1, "assets/", 7) == 0) {
        /* Unity 2022.3 Mono constructs absolute "/assets/..." paths when the
 * code_path doesn't have the expected APK-file structure.  Strip the
 * leading '/' so it resolves the same way as the relative form. */
        inner = path + 1;
    } else {
        return false;
    }
    /* Standard APK path first, then App Bundle split prefixes */
    static const char *cache_prefixes[] = {
        "", "base/", "UnityDataAssetPack/", nullptr
    };
    /* Directory mode: the APK was pre-extracted.  Return the source file path
 * directly instead of copying to a cache — avoids a redundant 4+ MB copy
 * for large files like mscorlib.dll and fixes a race where a freshly-opened
 * fd ends up at a stale position after the cache write. */
    if (!asset_bridge::g_apk_dir.empty()) {
        for (int pi = 0; cache_prefixes[pi]; pi++) {
            std::string fp = asset_bridge::g_apk_dir + "/" +
                             std::string(cache_prefixes[pi]) + inner;
            struct stat sb;
            if (stat(fp.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) {
                if ((size_t)snprintf(out, outsz, "%s", fp.c_str()) < outsz)
                    return true;
            }
        }
        return false;
    }
    std::vector<uint8_t> data;
    std::string inner_str;
    bool found = false;
    for (int pi = 0; cache_prefixes[pi]; pi++) {
        inner_str = std::string(cache_prefixes[pi]) + inner;
        if (asset_bridge::apk_read(inner_str, data)) { found = true; break; }
    }
    if (!found) return false;
    inner = inner_str.c_str();
    static const char cache_root[] = "/tmp/lunaria-apk-cache";
    if ((size_t)snprintf(out, outsz, "%s/%s", cache_root, inner) >= outsz)
        return false;
    /* reuse cache if already populated */
    struct stat st;
    if (stat(out, &st) == 0 && (uint64_t)st.st_size == data.size()) return true;
    /* mkdir -p */
    mkdir(cache_root, 0755);
    for (char *p = out + sizeof(cache_root); *p; ++p)
        if (*p == '/') { *p = '\0'; mkdir(out, 0755); *p = '/'; }
    FILE *f = fopen(out, "wb");
    if (!f) return false;
    size_t n = fwrite(data.data(), 1, data.size(), f);
    fclose(f);
    if (n != data.size()) { unlink(out); return false; }
    fprintf(stderr, "[asset] extracted %s (%zu bytes) -> %s\n", inner, data.size(), out);
    return true;
}

/* True if `path` points inside the APK at a directory (a prefix of zip
 * entries).  Handles App Bundle prefixes the same way as extraction so that
 * stat() on e.g. ".../base.apk/assets/bin/Data/Managed" reports a directory. */
static bool apk_path_is_dir(const char *path) {
    if (!path) return false;
    const char *apk = getenv("ANDROID_PACKAGE_CODE_PATH");
    const char *inner = nullptr;
    if (apk) {
        size_t alen = strlen(apk);
        if (strncmp(path, apk, alen) == 0 && path[alen] == '/')
            inner = path + alen + 1;
    }
    if (!inner) {
        /* "<pkg>.apk/assets/..." — in-APK path (like apk_extract_to_cache) */
        if (const char *sfx = strstr(path, ".apk/")) inner = sfx + 5;
    }
    if (!inner && path[0] == '/' && strncmp(path + 1, "assets/", 7) == 0)
        inner = path + 1;
    if (!inner && strncmp(path, "assets/", 7) == 0)
        inner = path;
    if (!inner) return false;
    static const char *prefixes[] = { "", "base/", "UnityDataAssetPack/", nullptr };
    for (int pi = 0; prefixes[pi]; pi++)
        if (asset_bridge::apk_is_dir(std::string(prefixes[pi]) + inner)) return true;
    return false;
}

/* List immediate children of an APK directory referenced by host path.
 * Returns true and fills `out` when `path` resolves to a zip directory. */
static bool apk_path_list_dir(const char *path,
                              std::vector<std::pair<std::string,bool>> &out) {
    const char *apk = getenv("ANDROID_PACKAGE_CODE_PATH");
    if (!apk || !path) return false;
    size_t alen = strlen(apk);
    const char *inner = nullptr;
    if (strncmp(path, apk, alen) == 0 && path[alen] == '/')
        inner = path + alen + 1;
    else if (const char *sfx = strstr(path, ".apk/"))
        inner = sfx + 5;   /* in-APK path ("<pkg>.apk/assets/...") */
    if (!inner) return false;
    static const char *prefixes[] = { "", "base/", "UnityDataAssetPack/", nullptr };
    for (int pi = 0; prefixes[pi]; pi++) {
        std::string full = std::string(prefixes[pi]) + inner;
        if (asset_bridge::apk_is_dir(full)) {
            asset_bridge::apk_list_dir(full, out);
            return true;
        }
    }
    return false;
}

/* Host open() for a guest path string (file:// URI, APK asset fallback). */
static int guest_open_path(const char *path, int flags, mode_t mode) {
    if (!path) return -1;
    if (strncmp(path, "file://", 7) == 0) {
        path += 7;
        if (*path != '/') {
            const char *sl = strchr(path, '/');
            if (sl) path = sl;
        }
    }
    int fd = open(path, flags, mode);
    if (fd < 0) {
        char cache[PATH_MAX];
        if (apk_extract_to_cache(path, cache, sizeof cache))
            fd = open(cache, flags, mode);
    }
    /* Mono reads <assembly>.dll.config beside managed DLLs; missing file is OK
 * but an empty readable file avoids fopen failure noise and edge cases in
 * mono_config_parse. */
    if (fd < 0 && strstr(path, ".config")) {
#if defined(__linux__)
        fd = memfd_create("lunaria-mono-config", MFD_CLOEXEC);
#else
        char tmpl[] = "/tmp/lunaria-mono-config-XXXXXX";
        fd = mkstemp(tmpl);
        if (fd >= 0) unlink(tmpl);
#endif
    }
    if (path && strstr(path, "Managed") && (strstr(path, ".dll") || strstr(path, ".config")))
        fprintf(stderr, "[guest_open] %s flags=%#x -> fd=%d\n", path, flags, fd);
    else if (getenv("LUNARIA_TRACE_OPEN"))
        fprintf(stderr, "[guest_open] %s flags=%#x -> fd=%d\n", path, flags, fd);
    return fd;
}

/* Fetch the i-th 32-bit word argument of a Call*Method/NewObject SVC.
 * variant 0 = varargs (r3 then caller stack), 1 = ...V (va_list at r3),
 * 2 = ...A (jvalue[] at r3, 8 bytes each).
 * ARM32 ABI: va_list is { void *__ap; };
 * JNI CallObjectMethodV passes va_list* as args;
 * r3 points at va_list; va_list.__ap is the argument array. */
static uint32_t jni_arg_word(ArmExecCtx &ctx, std::array<uint32_t,16> &regs, int variant, int i) {
    if (variant == 0)
        return (i == 0) ? regs[3] : ctx.mem.read32(regs[13] + (uint32_t)((i - 1) * 4));
    if (variant == 2)
        return ctx.mem.read32(regs[3] + (uint32_t)(i * 8));
    /* variant == 1 (MethodV): r3 = va_list.__ap (argument array pointer directly).
 * Android ARM32 passes va_list by value in r3; r3 itself IS the ap pointer. */
    return ctx.mem.read32(regs[3] + (uint32_t)(i * 4));
}

/* Try to service an asset-related JNI call host-side.  Returns true (and sets
 * regs[0]/regs[1]) when handled; false to fall through to generic dispatch.
 * Covers NewObject (28-30), CallObjectMethod (34-36), CallBooleanMethod
 * (37-39), CallIntMethod (49-51), CallLongMethod (52-54), CallVoidMethod
 * (61-63). */
static bool try_asset_jni(ArmExecCtx &ctx, uint32_t svc, std::array<uint32_t,16> &regs) {
    using namespace asset_bridge;
    struct jvm *jvm = ctx.jvm;
    JNIEnv *env = &jvm->env;
    auto &N = jvm->native;

    int base =
        (svc >= 28 && svc <= 30) ? 28 : (svc >= 34 && svc <= 36) ? 34 :
        (svc >= 37 && svc <= 39) ? 37 : (svc >= 49 && svc <= 51) ? 49 :
        (svc >= 52 && svc <= 54) ? 52 : (svc >= 61 && svc <= 63) ? 61 : -1;
    if (base < 0) return false;
    int variant = (int)svc - base;

    auto safe_class = [&](uint32_t h) -> const char* {
        if (h == 0 || h > 65536) return nullptr;
        auto &o = jvm->objects[h - 1];
        if (o.type == jvm_object::JVM_OBJECT_NONE) return nullptr;
        uintptr_t ki = (uintptr_t)o.this_klass;
        if (ki == 0 || ki > 65536) return nullptr;
        auto &ko = jvm->objects[ki - 1];
        if (ko.type != jvm_object::JVM_OBJECT_CLASS) return nullptr;
        return ko.klass.name.data;
    };

    /* NewObject: link a Scanner to the InputStream it wraps.  regs[1] is the
 * class being instantiated, so read its name directly (not via this_klass). */
    if (base == 28) {
        uint32_t ch = regs[1];
        const char *cn = (ch > 0 && ch <= 65536 && jvm->objects[ch - 1].type == jvm_object::JVM_OBJECT_CLASS)
                         ? jvm->objects[ch - 1].klass.name.data : nullptr;
        if (!cn || !strstr(cn, "Scanner")) return false;
        /* The InputStream is the constructor's first arg; tolerate the varargs,
 * va_list and jvalue[] layouts by picking whichever candidate is a
 * stream we created, else fall back to the last opened one. */
        uint32_t ish = jni_arg_word(ctx, regs, variant, 0);
        if (!g_streams.count(ish)) {
            for (uint32_t c : { regs[3], ctx.mem.read32(regs[3]) })
                if (g_streams.count(c)) { ish = c; break; }
        }
        if (!g_streams.count(ish)) ish = g_last_stream;
        jobject sc = N.AllocObject(env, (jclass)(uintptr_t)ch);
        uint32_t sh = (uint32_t)(uintptr_t)sc;
        if (g_streams.count(ish)) {
            g_scanner2stream[sh] = ish;
            fprintf(stderr, "[asset] Scanner 0x%x -> stream 0x%x\n", sh, ish);
        }
        regs[0] = sh;
        return true;
    }

    uint32_t self = regs[1], mid = regs[2];
    const char *name = nullptr, *msig = nullptr;
    if (mid > 0 && mid <= 65536) {
        auto &mo = jvm->objects[mid - 1];
        if (mo.type == jvm_object::JVM_OBJECT_METHOD) {
            name = mo.method.name.data;
            msig = mo.method.signature.data;
        }
    }
    if (!name) return false;

    if (!strcmp(name, "getAssets")) {
        
        static jobject s_asset_manager = nullptr;
        if (!s_asset_manager) {
            s_asset_manager = N.NewGlobalRef(env,
                N.AllocObject(env, N.FindClass(env, "android/content/res/AssetManager")));
        }
        regs[0] = (uint32_t)(uintptr_t)s_asset_manager;
        return true;
    }

    if (!strcmp(name, "open")) {
        const char *cn = safe_class(self);
        if (!cn || !strstr(cn, "AssetManager")) return false;
        uint32_t hpath = jni_arg_word(ctx, regs, variant, 0);
        const char *rel = N.GetStringUTFChars(env, (jstring)(uintptr_t)hpath, nullptr);
        if (!rel) { regs[0] = 0; return true; }
        /* Try standard APK path, then App Bundle split paths */
        static const char *prefixes[] = {
            "assets/", "base/assets/", "UnityDataAssetPack/assets/", nullptr
        };
        std::string full;
        std::vector<uint8_t> data;
        bool found = false;
        for (int pi = 0; prefixes[pi]; pi++) {
            full = std::string(prefixes[pi]) + rel;
            if (apk_read(full, data)) { found = true; break; }
        }
        if (!found) {
            fprintf(stderr, "[asset] open MISS %s\n", rel);
            regs[0] = 0;
            return true;
        }
        jobject is = N.AllocObject(env, N.FindClass(env, "java/io/InputStream"));
        uint32_t h = (uint32_t)(uintptr_t)is;
        size_t sz = data.size();
        g_streams[h] = Stream{ std::move(data), 0 };
        g_last_stream = h;
        fprintf(stderr, "[asset] open %s -> stream 0x%x (%zu bytes)\n", full.c_str(), h, sz);
        regs[0] = h;
        return true;
    }

    if (!strcmp(name, "read") && g_streams.count(self)) {
        Stream &st = g_streams[self];
        size_t remaining = st.data.size() - st.pos;
        int argc = sig_argc(msig);
        if (argc <= 0) { /* read() -> next byte or -1 */
            regs[0] = remaining ? st.data[st.pos++] : (uint32_t)-1;
            return true;
        }
        uint32_t buf = jni_arg_word(ctx, regs, variant, 0);
        jbyteArray ba = (jbyteArray)(uintptr_t)buf;
        jsize alen = N.GetArrayLength(env, ba);
        uint32_t off = 0, len = (uint32_t)alen;
        if (argc >= 3) { off = jni_arg_word(ctx, regs, variant, 1); len = jni_arg_word(ctx, regs, variant, 2); }
        if (off > (uint32_t)alen) off = (uint32_t)alen;
        if (len > (uint32_t)alen - off) len = (uint32_t)alen - off;
        if (len == 0) { regs[0] = 0; return true; }
        if (remaining == 0) { regs[0] = (uint32_t)-1; return true; }
        uint32_t n = len < remaining ? len : (uint32_t)remaining;
        jbyte *dst = N.GetByteArrayElements(env, ba, nullptr);
        if (dst) memcpy(dst + off, st.data.data() + st.pos, n);
        st.pos += n;
        regs[0] = n;
        return true;
    }

    if (!strcmp(name, "available") && g_streams.count(self)) {
        Stream &st = g_streams[self];
        regs[0] = (uint32_t)(st.data.size() - st.pos);
        return true;
    }

    if (!strcmp(name, "skip") && g_streams.count(self)) {
        Stream &st = g_streams[self];
        uint32_t want = jni_arg_word(ctx, regs, variant, 0); /* low word of long */
        size_t remaining = st.data.size() - st.pos;
        uint32_t n = want < remaining ? want : (uint32_t)remaining;
        st.pos += n;
        regs[0] = n; regs[1] = 0;
        return true;
    }

    if (!strcmp(name, "close") && g_streams.count(self)) {
        g_streams.erase(self);
        return true;
    }

    /* Scanner over an asset stream: useDelimiter("\\z") then next() yields the
 * whole text (this is how Unity reads boot.config).  The constructor link
 * may be missed depending on how the guest builds the Scanner, so fall back
 * to binding a Scanner-class receiver to the most recently opened stream. */
    bool scanner_call = !strcmp(name, "useDelimiter") || !strcmp(name, "next") || !strcmp(name, "hasNext");
    if (scanner_call && !g_scanner2stream.count(self)) {
        const char *cn = safe_class(self);
        if (cn && strstr(cn, "Scanner") && g_streams.count(g_last_stream))
            g_scanner2stream[self] = g_last_stream;
    }
    if (!strcmp(name, "useDelimiter") && g_scanner2stream.count(self)) {
        regs[0] = self;
        return true;
    }
    if (!strcmp(name, "next") && g_scanner2stream.count(self)) {
        Stream &st = g_streams[g_scanner2stream[self]];
        std::string s((const char*)st.data.data() + st.pos, st.data.size() - st.pos);
        st.pos = st.data.size();
        jstring js = N.NewStringUTF(env, s.c_str());
        regs[0] = (uint32_t)(uintptr_t)js;
        return true;
    }
    if (!strcmp(name, "hasNext") && g_scanner2stream.count(self)) {
        Stream &st = g_streams[g_scanner2stream[self]];
        regs[0] = (st.pos < st.data.size()) ? 1u : 0u;
        return true;
    }

    return false;
}

/* -------------------------------------------------------------------------
 * SVC dispatch
 * ---------------------------------------------------------------------- */
static void do_register_natives(ArmExecCtx &ctx,
                                uint32_t klass_jobj,
                                uint32_t methods_va, int32_t cnt) {
    struct jvm *jvm = ctx.jvm;
    const char *klass_name = jvm_get_class_name(jvm, (jobject)(uintptr_t)klass_jobj);
    for (int i = 0; i < cnt; ++i) {
        uint32_t name_va = ctx.mem.read32(methods_va + i*12 + 0);
        uint32_t sig_va  = ctx.mem.read32(methods_va + i*12 + 4);
        uint32_t fn_va   = ctx.mem.read32(methods_va + i*12 + 8);
        const char *name = ctx.mem.cstr(name_va);
        const char *sig  = ctx.mem.cstr(sig_va);
        if (!name || !sig) continue;
        RegisteredNative rn;
        rn.klass  = klass_name ? klass_name : "";
        rn.name   = name; rn.sig = sig; rn.fn_va = fn_va;
        ctx.natives.push_back(std::move(rn));
        fprintf(stderr, "[arm_exec] RegisterNatives: %s.%s%s @ 0x%08x\n",
                klass_name ? klass_name : "?", name, sig, fn_va);
    }
}

/* -------------------------------------------------------------------------
 * ---------------------------------------------------------------------- */
struct ArmVarArgs {
    ArmExecCtx &ctx;
    const std::array<uint32_t, 16> &regs;
    uint32_t reg_idx;   
    uint32_t mem_ptr;   
    bool     valist;    

    uint32_t next32() {
        if (!valist && reg_idx < 4) return regs[reg_idx++];
        uint32_t v = ctx.mem.read32(mem_ptr); mem_ptr += 4; return v;
    }
    uint64_t next64() {
        if (!valist && reg_idx < 4) {
            reg_idx = (reg_idx + 1u) & ~1u; 
            if (reg_idx < 4) {
                uint64_t lo = regs[reg_idx], hi = regs[reg_idx + 1];
                reg_idx += 2;
                return lo | (hi << 32);
            }
        }
        mem_ptr = (mem_ptr + 7u) & ~7u;
        uint64_t lo = ctx.mem.read32(mem_ptr), hi = ctx.mem.read32(mem_ptr + 4);
        mem_ptr += 8;
        return lo | (hi << 32);
    }
};

static std::string arm_vformat(ArmExecCtx &ctx, const char *fmt, ArmVarArgs &ap) {
    std::string out;
    if (!fmt) return out;
    char buf[512];
    for (const char *p = fmt; *p; ++p) {
        if (*p != '%') { out += *p; continue; }
        ++p;
        if (*p == '%') { out += '%'; continue; }
        if (!*p) break;
        std::string spec = "%";
        while (*p && strchr("-+ #0", *p)) spec += *p++;
        if (*p == '*') { spec += std::to_string((int)ap.next32()); ++p; }
        else while (*p >= '0' && *p <= '9') spec += *p++;
        if (*p == '.') {
            spec += *p++;
            if (*p == '*') { spec += std::to_string((int)ap.next32()); ++p; }
            else while (*p >= '0' && *p <= '9') spec += *p++;
        }
        int longs = 0;
        while (*p == 'l' || *p == 'h' || *p == 'z' || *p == 'j' ||
               *p == 't' || *p == 'L') {
            if (*p == 'l') ++longs;
            ++p;
        }
        char conv = *p;
        if (!conv) break;
        buf[0] = '\0';
        switch (conv) {
        case 'd': case 'i':
            if (longs >= 2) { spec += "lld"; snprintf(buf, sizeof buf, spec.c_str(), (long long)(int64_t)ap.next64()); }
            else            { spec += conv;  snprintf(buf, sizeof buf, spec.c_str(), (int)ap.next32()); }
            out += buf; break;
        case 'u': case 'x': case 'X': case 'o':
            if (longs >= 2) { spec += "ll"; spec += conv; snprintf(buf, sizeof buf, spec.c_str(), (unsigned long long)ap.next64()); }
            else            { spec += conv; snprintf(buf, sizeof buf, spec.c_str(), (unsigned)ap.next32()); }
            out += buf; break;
        case 'p':
            snprintf(buf, sizeof buf, "0x%x", ap.next32());
            out += buf; break;
        case 'c':
            spec += 'c'; snprintf(buf, sizeof buf, spec.c_str(), (int)ap.next32());
            out += buf; break;
        case 's': case 'S': {
            uint32_t va = ap.next32();
            if (longs >= 1 || conv == 'S') {
                
                std::string ws;
                if (va) for (uint32_t i = 0;; i += 4) {
                    uint32_t c = ctx.mem.read32(va + i);
                    if (!c) break;
                    ws += (c < 0x80) ? (char)c : '?';
                    if (ws.size() > 4096) break;
                }
                spec += 's'; snprintf(buf, sizeof buf, spec.c_str(),
                                      va ? ws.c_str() : "(null)");
            } else {
                const char *s = va ? ctx.mem.cstr(va) : "(null)";
                spec += 's'; snprintf(buf, sizeof buf, spec.c_str(), s ? s : "(null)");
            }
            out += buf; break;
        }
        case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': case 'a': {
            uint64_t bits = ap.next64();
            double d; std::memcpy(&d, &bits, 8);
            spec += conv; snprintf(buf, sizeof buf, spec.c_str(), d);
            out += buf; break;
        }
        case 'n': {
            uint32_t va = ap.next32();
            if (va) ctx.mem.write32(va, (uint32_t)out.size());
            break;
        }
        default: 
            out += spec; out += conv; break;
        }
    }
    return out;
}


static uint32_t arm_format_to(ArmExecCtx &ctx, uint32_t buf_va, uint32_t size,
                              const std::string &s) {
    if (buf_va && size) {
        uint32_t n = (uint32_t)s.size() < size - 1 ? (uint32_t)s.size() : size - 1;
        memcpy(ctx.mem.ptr(buf_va), s.data(), n);
        *ctx.mem.ptr(buf_va + n) = 0;
    }
    return (uint32_t)s.size();
}


struct ArmDir {
    DIR *d;                                          /* host dir, or null for APK */
    uint32_t dirent_va;
    std::vector<std::pair<std::string,bool>> apk;    /* synthetic APK entries */
    size_t apk_idx = 0;
};
static std::map<uint32_t, ArmDir> g_dir_tab;
static uint32_t g_dir_next = 1u;


static uint32_t arm_vsscanf(ArmExecCtx &ctx, const char *in, const char *fmt,
                            ArmVarArgs &ap) {
    if (!in || !fmt) return 0;
    const char *ip = in;
    uint32_t count = 0;
    for (const char *p = fmt; *p; ++p) {
        if (isspace((unsigned char)*p)) {
            while (isspace((unsigned char)*ip)) ++ip;
            continue;
        }
        if (*p != '%') {
            if (*ip != *p) return count;
            ++ip; continue;
        }
        ++p;
        bool suppress = false;
        if (*p == '*') { suppress = true; ++p; }
        std::string width;
        while (*p >= '0' && *p <= '9') width += *p++;
        int longs = 0; bool half = false;
        while (*p=='l'||*p=='h'||*p=='z'||*p=='j'||*p=='L') {
            if (*p=='l'||*p=='L') ++longs;
            if (*p=='h') half = true;
            ++p;
        }
        char conv = *p;
        if (!conv) return count;
        char hfmt[48]; int consumed = -1;
        switch (conv) {
        case 'd': case 'i': {
            long long v = 0;
            snprintf(hfmt, sizeof hfmt, "%%%sll%c%%n", width.c_str(), conv);
            if (sscanf(ip, hfmt, &v, &consumed) < 1 || consumed < 0) return count;
            if (!suppress) {
                uint32_t dst = ap.next32();
                if (longs >= 2) { ctx.mem.write32(dst, (uint32_t)v);
                                  ctx.mem.write32(dst+4, (uint32_t)((uint64_t)v >> 32)); }
                else if (half)  { uint16_t hv = (uint16_t)v; memcpy(ctx.mem.ptr(dst), &hv, 2); }
                else            ctx.mem.write32(dst, (uint32_t)v);
                ++count;
            }
            ip += consumed; break;
        }
        case 'u': case 'x': case 'X': case 'o': {
            unsigned long long v = 0;
            snprintf(hfmt, sizeof hfmt, "%%%sll%c%%n", width.c_str(), conv);
            if (sscanf(ip, hfmt, &v, &consumed) < 1 || consumed < 0) return count;
            if (!suppress) {
                uint32_t dst = ap.next32();
                if (longs >= 2) { ctx.mem.write32(dst, (uint32_t)v);
                                  ctx.mem.write32(dst+4, (uint32_t)(v >> 32)); }
                else if (half)  { uint16_t hv = (uint16_t)v; memcpy(ctx.mem.ptr(dst), &hv, 2); }
                else            ctx.mem.write32(dst, (uint32_t)v);
                ++count;
            }
            ip += consumed; break;
        }
        case 'f': case 'g': case 'e': case 'E': case 'G': {
            double d = 0;
            snprintf(hfmt, sizeof hfmt, "%%%slf%%n", width.c_str());
            if (sscanf(ip, hfmt, &d, &consumed) < 1 || consumed < 0) return count;
            if (!suppress) {
                uint32_t dst = ap.next32();
                if (longs >= 1) { uint64_t b; memcpy(&b, &d, 8);
                                  ctx.mem.write32(dst, (uint32_t)b);
                                  ctx.mem.write32(dst+4, (uint32_t)(b >> 32)); }
                else { float f = (float)d; uint32_t b; memcpy(&b, &f, 4);
                       ctx.mem.write32(dst, b); }
                ++count;
            }
            ip += consumed; break;
        }
        case 's': {
            char tmp[512];
            int w = width.empty() ? 511 : atoi(width.c_str());
            if (w > 511 || w <= 0) w = 511;
            snprintf(hfmt, sizeof hfmt, "%%%ds%%n", w);
            if (sscanf(ip, hfmt, tmp, &consumed) < 1 || consumed < 0) return count;
            if (!suppress) {
                uint32_t dst = ap.next32();
                strcpy((char*)ctx.mem.ptr(dst), tmp);
                ++count;
            }
            ip += consumed; break;
        }
        case 'c': {
            if (!*ip) return count;
            if (!suppress) { uint32_t dst = ap.next32(); *ctx.mem.ptr(dst) = (uint8_t)*ip; ++count; }
            ++ip; break;
        }
        case '%':
            if (*ip != '%') return count;
            ++ip; break;
        default:
            return count;
        }
    }
    return count;
}

/* guest struct stat (bionic NDK arm LP32, not packed):
 * st_dev@0(8) __pad0@8(4) __st_ino@12 st_mode@16 st_nlink@20 st_uid@24
 * st_gid@28 st_rdev@32(8) __pad3@40(4) [pad@44] st_size@48(8) st_blksize@56
 * [pad@60] st_blocks@64(8) st_atim@72 st_mtim@80 st_ctim@88 st_ino@96(8) */
static void write_guest_stat(ArmExecCtx &ctx, uint32_t va, const struct stat &st) {
    memset(ctx.mem.ptr(va), 0, 104);
    ctx.mem.write32(va + 0,  (uint32_t)st.st_dev);
    ctx.mem.write32(va + 4,  (uint32_t)((uint64_t)st.st_dev >> 32));
    ctx.mem.write32(va + 12, (uint32_t)st.st_ino);
    ctx.mem.write32(va + 16, (uint32_t)st.st_mode);
    ctx.mem.write32(va + 20, (uint32_t)st.st_nlink);
    ctx.mem.write32(va + 24, (uint32_t)st.st_uid);
    ctx.mem.write32(va + 28, (uint32_t)st.st_gid);
    ctx.mem.write32(va + 32, (uint32_t)st.st_rdev);
    ctx.mem.write32(va + 36, (uint32_t)((uint64_t)st.st_rdev >> 32));
    ctx.mem.write32(va + 48, (uint32_t)st.st_size);
    ctx.mem.write32(va + 52, (uint32_t)((uint64_t)st.st_size >> 32));
    ctx.mem.write32(va + 56, (uint32_t)st.st_blksize);
    ctx.mem.write32(va + 64, (uint32_t)st.st_blocks);
    ctx.mem.write32(va + 72, (uint32_t)st.st_atime);
    ctx.mem.write32(va + 80, (uint32_t)st.st_mtime);
    ctx.mem.write32(va + 88, (uint32_t)st.st_ctime);
    ctx.mem.write32(va + 96, (uint32_t)st.st_ino);
    ctx.mem.write32(va + 100, (uint32_t)((uint64_t)st.st_ino >> 32));
}

/* guest struct tm (bionic LP32): int×9 @0..32, long tm_gmtoff@36, char* tm_zone@40 */
static void write_guest_tm(ArmExecCtx &ctx, uint32_t va, const struct tm &t) {
    ctx.mem.write32(va +  0, (uint32_t)t.tm_sec);
    ctx.mem.write32(va +  4, (uint32_t)t.tm_min);
    ctx.mem.write32(va +  8, (uint32_t)t.tm_hour);
    ctx.mem.write32(va + 12, (uint32_t)t.tm_mday);
    ctx.mem.write32(va + 16, (uint32_t)t.tm_mon);
    ctx.mem.write32(va + 20, (uint32_t)t.tm_year);
    ctx.mem.write32(va + 24, (uint32_t)t.tm_wday);
    ctx.mem.write32(va + 28, (uint32_t)t.tm_yday);
    ctx.mem.write32(va + 32, (uint32_t)t.tm_isdst);
    ctx.mem.write32(va + 36, (uint32_t)t.tm_gmtoff);
    ctx.mem.write32(va + 40, 0);
}
static void read_guest_tm(ArmExecCtx &ctx, uint32_t va, struct tm &t) {
    memset(&t, 0, sizeof t);
    t.tm_sec   = (int)ctx.mem.read32(va +  0);
    t.tm_min   = (int)ctx.mem.read32(va +  4);
    t.tm_hour  = (int)ctx.mem.read32(va +  8);
    t.tm_mday  = (int)ctx.mem.read32(va + 12);
    t.tm_mon   = (int)ctx.mem.read32(va + 16);
    t.tm_year  = (int)ctx.mem.read32(va + 20);
    t.tm_wday  = (int)ctx.mem.read32(va + 24);
    t.tm_yday  = (int)ctx.mem.read32(va + 28);
    t.tm_isdst = (int)ctx.mem.read32(va + 32);
}

/* Session 10 (LUNARIA_TRACE_WAPI): when mono logs a WAPI handle-store warning
 * ("Attempting to ref/unref unused handle", "error looking up thread handle"),
 * dump lr + the libmono-range return addresses on the guest stack so the
 * internal (non-exported) WAPI functions — _wapi_handle_ref / _wapi_handle_new /
 * thread_attach — can be located by guest VA for follow-up disassembly/detour. */
static void wapi_trace_caller(ArmExecCtx &ctx, std::array<uint32_t,16> &regs,
                              const char *msg) {
    static const bool on = getenv("LUNARIA_TRACE_WAPI") != nullptr;
    if (!on || !msg) return;
    if (!strstr(msg, "unused handle") && !strstr(msg, "looking up thread handle") &&
        !strstr(msg, "thread handle"))
        return;
    /* libmono is preloaded at 0x20000000; .text spans ~0x20015b08..0x202e40d0. */
    auto in_libmono = [](uint32_t v){ uint32_t a=v&~1u; return a>=0x20010000u && a<0x20400000u; };
    fprintf(stderr, "[wapi] >>> '%s'  lr=0x%08x sp=0x%08x\n",
            msg, regs[14], regs[13]);
    uint32_t sp = regs[13];
    int shown = 0;
    for (uint32_t off = 0; off < 0x400 && shown < 16; off += 4) {
        uint32_t w = ctx.mem.read32(sp + off);
        if (in_libmono(w)) {
            fprintf(stderr, "[wapi]     stack+0x%03x -> 0x%08x\n", off, w);
            ++shown;
        }
    }
    /* Decode the io-layer handle-store slot directly from guest memory.
 * From the disassembly (libmono @0x20000000):
 * _wapi_private_handles array base = *(GOT[0x7c8])  (GOT @ .got 0x3b8604)
 * segment ptr = array_base[handle>>8]   (each segment = 256 slots)
 * slot        = segment + (handle&0xff)*60   (sizeof _WapiHandleUnshared=60)
 * slot[0]=type  slot[4]=ref
 * If the slot type reads 0 while _wapi_handle_new just returned this handle,
 * the mark store in _wapi_handle_init (0x279ab0) did not persist where the
 * reader looks. */
    const char *hp = strrchr(msg, 'x');
    uint32_t handle = hp ? (uint32_t)strtoul(hp - 1, nullptr, 0) : 0;
    if (!handle) return;
    const uint32_t GOT_WPH = 0x20000000u + 0x3b8604u + 0x7c8u; /* &_wapi_private_handles */
    uint32_t arr  = ctx.mem.read32(GOT_WPH);
    uint32_t seg  = arr ? ctx.mem.read32(arr + ((handle >> 8) & 0xffffff) * 4u) : 0;
    uint32_t slot = seg ? seg + (handle & 0xff) * 60u : 0;
    uint32_t type = slot ? ctx.mem.read32(slot + 0) : 0;
    uint32_t ref  = slot ? ctx.mem.read32(slot + 4) : 0;
    fprintf(stderr, "[wapi]   slot handle=0x%x: &arr=0x%08x arr=0x%08x seg=0x%08x "
            "slot=0x%08x type=%u ref=%u\n",
            handle, GOT_WPH, arr, seg, slot, type, ref);
    if (arr) {
        fprintf(stderr, "[wapi]   arr[0..3]=%08x %08x %08x %08x\n",
                ctx.mem.read32(arr), ctx.mem.read32(arr+4),
                ctx.mem.read32(arr+8), ctx.mem.read32(arr+12));
    }
    /* handle_count @0x203c016c, segment-count @0x203c0160 (from disassembly of
 * the grow path in _wapi_handle_new worker 0x27a018). */
    fprintf(stderr, "[wapi]   handle_count=%u seg_count=%u\n",
            ctx.mem.read32(0x203c016cu), ctx.mem.read32(0x203c0160u));
    /* Where did the marks actually land?  Scan slots 0..7 of arr[0..2]. */
    for (uint32_t s = 0; s < 3; ++s) {
        uint32_t sp2 = arr ? ctx.mem.read32(arr + s*4u) : 0;
        if (!sp2) { fprintf(stderr, "[wapi]   arr[%u]=NULL\n", s); continue; }
        fprintf(stderr, "[wapi]   arr[%u]=0x%08x slot types[0..7]:", s, sp2);
        for (uint32_t i = 0; i < 8; ++i)
            fprintf(stderr, " %u", ctx.mem.read32(sp2 + i*60u));
        fprintf(stderr, "\n");
    }
}

static int run_arm(ArmExecCtx &ctx, uint32_t entry_va,
                   uint32_t r0_arg, uint32_t r1_arg,
                   uint32_t r2_arg, uint32_t r3_arg,
                   uint64_t max_ticks);
extern "C" void arm_exec_prepare_mono_config(void);

static void dispatch_svc(ArmExecCtx &ctx, uint32_t svc_no,
                         std::array<uint32_t, 16> &regs) {
    struct jvm *jvm = ctx.jvm;
    JNIEnv     *env = &jvm->env;
    uint32_t r0=regs[0], r1=regs[1], r2=regs[2], r3=regs[3];

    auto ret32 = [&](uint32_t v)  { regs[0] = v; };
    auto ret64 = [&](uint64_t v)  { regs[0]=(uint32_t)v; regs[1]=(uint32_t)(v>>32); };

    /* inline-detour logging (LUNARIA_TRACE_EXC).  Fires as the first
 * instruction of a patched libmono function — regs are exactly the callee's
 * entry args, lr is the caller.  MUST NOT touch any register (the detour
 * stub then runs the relocated prologue and resumes the real function). */
    if (svc_no >= SVC_DETOUR_BASE && svc_no < SVC_DETOUR_BASE + NUM_DETOURS) {
        uint32_t n = svc_no - SVC_DETOUR_BASE;
        const char *nm = (n < NUM_DETOURS && g_detour_names[n]) ? g_detour_names[n] : "?";
        static uint64_t hit[NUM_DETOURS] = {};
        uint64_t h = hit[n]++;   /* n < NUM_DETOURS, checked above (no power-of-2 mask) */
        bool verbose = h < 30;  /* rate-limit: only the first 30 hits per detour */
        if (verbose)
        fprintf(stderr, "[detour] >>> %s #%llu r0=%08x r1=%08x r2=%08x r3=%08x lr=%08x tid=%u\n",
                nm, (unsigned long long)h, r0, r1, r2, r3, regs[14], g_current_tid);
        if (verbose && nm && strstr(nm, "mono_image_open") && r0) {
            const char *p = ctx.mem.cstr(r0);
            if (p) fprintf(stderr, "[detour]   path=\"%s\" status=%#x\n", p, r1);
        }
        /* typelookup (0x210b4): r0=image/scope, r1=key.  It looks up r1 in the
 * g_hash_table at r0+0x70; if missing it recurses to the parent scope.
 * The table's hash function is table[0]; if that resolves to a trampoline
 * (0x41000xxx) the symbol was unresolved, every lookup mis-hashes, and the
 * parent-fallback recursion runs ~46600 deep.  Flag that even past the limit. */
        if (strstr(nm, "typelookup")) {
            uint32_t table  = (r0 >= 0x10000u && !(r0 & 3u)) ? (r0 + 0x70u) : 0;
            /* eglib GHashTable: [0]=hash_func [4]=key_equal_func [16]=in_use. */
            uint32_t hashfn = table ? ctx.mem.read32(table)        : 0;
            uint32_t eqfn   = table ? ctx.mem.read32(table + 4u)   : 0;
            uint32_t in_use = table ? ctx.mem.read32(table + 16u)  : 0;
            auto isT = [](uint32_t v){ return v >= 0x41000000u && v < 0x41010000u; };
            if (verbose || isT(hashfn) || isT(eqfn))
                fprintf(stderr, "[detour]   typelookup table@0x%08x hashfn=0x%08x%s eqfn=0x%08x%s in_use=%u\n",
                        table, hashfn, isT(hashfn) ? "(TRAMP!)" : "",
                        eqfn, isT(eqfn) ? "(TRAMP!)" : "", in_use);
        }
        /* ghashtable: NULL table pointer triggers ghashtable.c:184 assertion. */
        if (strstr(nm, "g_hash_insert_int") && !r0)
            fprintf(stderr, "[detour]   *** %s NULL table! lr=%#x tid=%u\n",
                    nm, regs[14], g_current_tid);
        if (strstr(nm, "g_assert") || strstr(nm, "assert_fail")) {
            uint32_t sp = regs[13];
            uint32_t a0 = sp ? ctx.mem.read32(sp) : 0;
            uint32_t a1 = sp ? ctx.mem.read32(sp + 4u) : 0;
            const char *cond = r2 ? ctx.mem.cstr(r2) : nullptr;
            const char *file = a1 ? ctx.mem.cstr(a1) : nullptr;
            fprintf(stderr, "[detour]   %s r0=%u r1=%u cond=%s file=%s line=%u lr=%#x tid=%u\n",
                    nm, r0, r1, cond ? cond : "(null)", file ? file : "(null)", a0, regs[14], g_current_tid);
        }
        if (strstr(nm, "g_hash")) {
            if (verbose || !r0)
                fprintf(stderr, "[detour]   %s r0=%#x r1=%#x r2=%#x lr=%#x tid=%u\n",
                        nm, r0, r1, r2, regs[14], g_current_tid);
            if (r0 >= 0x10000u && !(r0 & 3u) && strstr(nm, "new")) {
                
            }
            if (!r0 && (strstr(nm, "insert") || strstr(nm, "lookup") || strstr(nm, "foreach")))
                fprintf(stderr, "[detour]   *** %s called with NULL hash table! lr=%#x\n",
                        nm, regs[14]);
            if (r0 >= 0x10000u && !(r0 & 3u) &&
                (strstr(nm, "lookup") || strstr(nm, "insert"))) {
                uint32_t hashfn = ctx.mem.read32(r0);
                uint32_t eqfn   = ctx.mem.read32(r0 + 4u);
                auto isT = [](uint32_t v){ return v >= 0x41000000u && v < 0x41010000u; };
                if (isT(hashfn) || isT(eqfn))
                    fprintf(stderr, "[detour]   %s table=%#x hashfn=%#x%s eqfn=%#x%s\n",
                            nm, r0, hashfn, isT(hashfn)?"(TRAMP!)":"",
                            eqfn, isT(eqfn)?"(TRAMP!)":"");
            }
        }
        /* mono_object_new_specific(r0=vtable).  The slow remoting/
 * Activator path (which recurses into mono_type_get_object and drives the
 * runaway recursion) is taken iff:
 * vtable->remote        (byte vtable+0x1b, bit0)  OR
 * vtable->klass->is_com_object (byte klass+0x17, bit6 = 0x40).
 * For System.MonoType/RuntimeType BOTH should be 0 on real hardware, so
 * dump them to see which flag Lunaria has wrong. */
        if (strstr(nm, "mono_object_new_specific")) {
            uint32_t vt    = r0;
            uint32_t klass = (vt >= 0x10000u && !(vt & 3u)) ? ctx.mem.read32(vt) : 0;
            uint8_t  remote = ctx.mem.ptr(vt + 0x1bu)[0];
            uint8_t  kflags = klass ? ctx.mem.ptr(klass + 0x17u)[0] : 0;
            bool slow = (remote & 1u) || (kflags & 0x40u);
            /* MonoClass name/name_space (mono 2.x: name at +0x2c, name_space +0x30). */
            const char *kname = "?", *kns = "?";
            if (klass) {
                uint32_t np = ctx.mem.read32(klass + 0x2cu);
                uint32_t sp = ctx.mem.read32(klass + 0x30u);
                if (np && np < 0xf0000000u) kname = ctx.mem.cstr(np);
                if (sp && sp < 0xf0000000u) kns   = ctx.mem.cstr(sp);
            }
            if (verbose || slow)
                fprintf(stderr, "[detour]   new_specific vt=0x%08x klass=0x%08x remote(+0x1b)=0x%02x "
                        "is_com(+0x17&0x40)=%u -> %s  class=%s.%s\n",
                        vt, klass, remote, (kflags & 0x40u) ? 1u : 0u,
                        slow ? "SLOW(remoting/Activator->recurse)" : "fast",
                        kns ? kns : "?", kname ? kname : "?");
        }
        
        if (strstr(nm, "mempoolgrow")) {
            uint32_t next  = ctx.mem.read32(r0);
            uint32_t hsize = next ? ctx.mem.read32(next + 0x10u) : ctx.mem.read32(r0 + 0x10u);
            static int zn = 0;
            if (verbose || ((int32_t)hsize <= 0 && zn++ < 50))
                fprintf(stderr, "[detour]   mempoolgrow pool=0x%08x next=0x%08x "
                        "growbase=%d pool_size=%d needed=%u lr=0x%08x tid=%u%s\n",
                        r0, next, (int32_t)hsize, (int32_t)ctx.mem.read32(r0 + 0x10u),
                        r1, regs[14], g_current_tid,
                        (int32_t)hsize <= 0 ? "  *** INFINITE LOOP ***" : "");
        }
        
        if (strstr(nm, "mempoolalloc")) {
            uint32_t psize = ctx.mem.read32(r0 + 0x10u);
            static int bn = 0;
            if ((int32_t)psize < 0x200 && bn++ < 50) {
                fprintf(stderr, "[detour]   mempoolalloc BAD pool=0x%08x pool_size=%d "
                        "needed=%u lr=0x%08x tid=%u  pool[0..7]:", r0, (int32_t)psize,
                        r1, regs[14], g_current_tid);
                for (int k = 0; k < 8; ++k)
                    fprintf(stderr, " %08x", ctx.mem.read32(r0 + (uint32_t)k*4u));
                fprintf(stderr, "\n");
            }
        }
        
        if (strstr(nm, "badvtable")) {
            static int vn = 0;
            if (r0 >= 0x10000000u && r0 < 0x20000000u && vn++ < 50)
                fprintf(stderr, "[detour]   badvtable domain=0x%08x class=0x%08x "
                        "lr=0x%08x tid=%u\n", r0, r1, regs[14], g_current_tid);
        }
        
        if (strstr(nm, "domainalloc")) {
            uint32_t mp = ctx.mem.read32(r0 + 8u);
            uint32_t msz = mp ? ctx.mem.read32(mp + 0x10u) : 0;
            static int dn = 0;
            if (((int32_t)msz < 0x200 || !mp) && dn++ < 50)
                fprintf(stderr, "[detour]   domainalloc BAD domain=0x%08x mp=0x%08x "
                        "mp_size=%d size=%u lr=0x%08x tid=%u\n",
                        r0, mp, (int32_t)msz, r1, regs[14], g_current_tid);
        }
        
        if (strstr(nm, "imgblit") && r0 >= 0x10000u && r1 >= 0x10000u) {
            auto dumpdesc = [&](const char *tag, uint32_t d) {
                fprintf(stderr, "[detour]   %s w=%u h=%u rowbytes=%u fmtbits=0x%08x "
                        "fmt=%u data=0x%08x\n", tag,
                        ctx.mem.read32(d), ctx.mem.read32(d + 4u),
                        ctx.mem.read32(d + 8u), ctx.mem.read32(d + 0xcu),
                        ctx.mem.read32(d + 0x10u), ctx.mem.read32(d + 0x28u));
            };
            dumpdesc("dst", r0);
            dumpdesc("src", r1);
        }
        
        if (strstr(nm, "create_21054") || strstr(nm, "gate_182040")) {
            uint32_t key = strstr(nm, "create_21054") ? r1 : r0;
            if (key >= 0x10000u && !(key & 3u)) {
                uint8_t  b14 = ctx.mem.ptr(key + 0x14u)[0];
                uint8_t  b15 = ctx.mem.ptr(key + 0x15u)[0];
                uint8_t  ty  = ctx.mem.ptr(key + 0x06u)[0];
                uint32_t kls = ctx.mem.read32(key + 0x08u);
                uint32_t gcont = (kls >= 0x10000u && !(kls & 3u)) ? ctx.mem.read32(kls + 0x88u) : 0;
                const char *kn = "?", *kns = "?";
                if (kls >= 0x10000u && !(kls & 3u)) {
                    uint32_t np = ctx.mem.read32(kls + 0x2cu), sp = ctx.mem.read32(kls + 0x30u);
                    if (np && np < 0xf0000000u) kn  = ctx.mem.cstr(np);
                    if (sp && sp < 0xf0000000u) kns = ctx.mem.cstr(sp);
                }
                bool open15 = (b15 & 8u);
                bool gen    = !(b14 & 0x7cu) && gcont;
                if (verbose) {
                    fprintf(stderr, "[detour]   key=0x%08x type=0x%02x +0x14=0x%02x +0x15=0x%02x "
                            "klass=0x%08x gcont(+0x88)=0x%08x helper=%d (open15=%d gen=%d) class=%s.%s\n",
                            key, ty, b14, b15, kls, gcont, (open15 || gen) ? 1 : 0,
                            open15 ? 1 : 0, gen ? 1 : 0, kns ? kns : "?", kn ? kn : "?");
                    /* raw dump + class name (image@+0x24, name@+0x28,
 * name_space@+0x2c in this libmono's MonoClass layout). */
                    if (h < 3 && kls >= 0x10000u && !(kls & 3u)) {
                        uint32_t np = ctx.mem.read32(kls + 0x28u);   /* name */
                        uint32_t sp = ctx.mem.read32(kls + 0x2cu);   /* name_space */
                        const char *nn = (np && np < 0xf0000000u) ? ctx.mem.cstr(np) : "?";
                        const char *ss = (sp && sp < 0xf0000000u) ? ctx.mem.cstr(sp) : "?";
                        fprintf(stderr, "[detour]   FAILING TYPE = %s.%s\n", ss ? ss : "?", nn ? nn : "?");
                        fprintf(stderr, "[detour]   raw klass +0x20:");
                        for (uint32_t o = 0x20u; o < 0x48u; o += 4) fprintf(stderr, " %08x", ctx.mem.read32(kls + o));
                        fprintf(stderr, "\n");
                    }
                }
            }
        }
        /* the IL pointer mono_image_rva_map returned for the method body. */
        if (strstr(nm, "rvamap_ret") && verbose) {
            uint32_t p = r0;
            fprintf(stderr, "[detour]   rva_map ilptr=0x%08x bytes=%08x %08x %08x\n", p,
                    (p >= 0x10000u) ? ctx.mem.read32(p)      : 0,
                    (p >= 0x10000u) ? ctx.mem.read32(p + 4u) : 0,
                    (p >= 0x10000u) ? ctx.mem.read32(p + 8u) : 0);
        }
        
        if (strstr(nm, "method_get_header") && h == 0) {
            fprintf(stderr, "[detour]   verifier globals *0x203c0100=0x%08x *0x203c00fc=0x%08x\n",
                    ctx.mem.read32(0x203c0100u), ctx.mem.read32(0x203c00fcu));
        }
        
        if (strstr(nm, "method_get_header") && verbose) {
            uint32_t mth = r0, klass = 0, np = 0;
            const char *mn = "?", *kn = "?", *ks = "?";
            if (mth >= 0x10000u && !(mth & 3u)) {
                np    = ctx.mem.read32(mth + 0x10u);
                klass = ctx.mem.read32(mth + 0x08u);
                if (np && np < 0xf0000000u) mn = ctx.mem.cstr(np);
                if (klass >= 0x10000u && !(klass & 3u)) {
                    uint32_t kp = ctx.mem.read32(klass + 0x28u), sp = ctx.mem.read32(klass + 0x2cu);
                    if (kp && kp < 0xf0000000u) kn = ctx.mem.cstr(kp);
                    if (sp && sp < 0xf0000000u) ks = ctx.mem.cstr(sp);
                }
            }
            uint32_t w0 = (mth >= 0x10000u && !(mth & 3u)) ? ctx.mem.read32(mth) : 0;
            uint16_t flags = (uint16_t)w0, iflags = (uint16_t)(w0 >> 16);
            uint8_t  b15 = (mth >= 0x10000u && !(mth & 3u)) ? ctx.mem.ptr(mth + 0x15u)[0] : 0;
            
            uint32_t tok = (mth >= 0x10000u && !(mth & 3u)) ? ctx.mem.read32(mth + 4u) : 0;
            fprintf(stderr, "[detour]   get_header method=%s.%s::%s flags=0x%04x iflags=0x%04x "
                    "b15=0x%02x token=0x%08x earlyNULL=%d\n",
                    ks ? ks : "?", kn ? kn : "?", mn ? mn : "?", flags, iflags, b15, tok,
                    ((flags & 0x400u) || (iflags & 3u) || (iflags & 0x1000u) || (flags & 0x2000u)) ? 1 : 0);
        }

        
        if (strstr(nm, "builder_throwcheck")) {
            uint32_t kls    = r0;   /* class returned by loader 0x1ea80 (detour at 0x21724) */
            uint32_t etype  = (kls >= 0x10000u && !(kls & 3u)) ? ctx.mem.read32(kls + 0x1a8u) : 0xffffffffu;
            const char *nn = "?", *ss = "?";
            if (kls >= 0x10000u && !(kls & 3u)) {
                uint32_t np = ctx.mem.read32(kls + 0x28u), sp = ctx.mem.read32(kls + 0x2cu);
                if (np && np < 0xf0000000u) nn = ctx.mem.cstr(np);
                if (sp && sp < 0xf0000000u) ss = ctx.mem.cstr(sp);
            }
            uint32_t tok = (kls >= 0x10000u && !(kls & 3u)) ? ctx.mem.read32(kls + 0x30u) : 0;
            static int btc = 0;
            if (btc++ < 20)
                fprintf(stderr, "[detour]   builder_throwcheck: klass=0x%08x %s.%s "
                        "token=0x%08x exception_type(+0x1a8)=%u%s\n",
                        kls, ss, nn, tok, etype & 0xffu,
                        (etype & 0xffu) == 7u ? "  ← TYPE_LOAD → throws TypeLoadException" : "");
        }
        
        if (strstr(nm, "loader_1ea80")) {
            uint32_t method = r0;
            uint32_t mklass = (method >= 0x10000u && !(method & 3u)) ? ctx.mem.read32(method + 0x8u) : 0;
            uint32_t mtok   = (method >= 0x10000u && !(method & 3u)) ? ctx.mem.read32(method + 0x4u) : 0;
            uint32_t mnp    = (method >= 0x10000u && !(method & 3u)) ? ctx.mem.read32(method + 0x10u) : 0;
            const char *mn = (mnp && mnp < 0xf0000000u) ? ctx.mem.cstr(mnp) : "?";
            const char *kn = "?", *ks = "?";
            uint32_t ktok = 0, gcont = 0;
            if (mklass >= 0x10000u && !(mklass & 3u)) {
                uint32_t np = ctx.mem.read32(mklass + 0x28u), sp = ctx.mem.read32(mklass + 0x2cu);
                if (np && np < 0xf0000000u) kn = ctx.mem.cstr(np);
                if (sp && sp < 0xf0000000u) ks = ctx.mem.cstr(sp);
                ktok  = ctx.mem.read32(mklass + 0x30u);
                gcont = ctx.mem.read32(mklass + 0x88u);
            }
            static int l8 = 0;
            if (l8++ < 12)
                fprintf(stderr, "[detour]   loader_1ea80: method=0x%08x %s.%s::%s mtok=0x%08x "
                        "klass=0x%08x ktok=0x%08x generic_container(+0x88)=0x%08x r2(scope)=0x%08x r3=0x%08x\n",
                        method, ks, kn, mn, mtok, mklass, ktok, gcont, r2, r3);
        }
        
        if (strstr(nm, "getmethod_in")) {
            uint32_t method = r0, token = r1;
            uint32_t mnp = (method >= 0x10000u && !(method & 3u)) ? ctx.mem.read32(method + 0x10u) : 0;
            uint32_t mk  = (method >= 0x10000u && !(method & 3u)) ? ctx.mem.read32(method + 0x8u)  : 0;
            const char *mn = (mnp && mnp < 0xf0000000u) ? ctx.mem.cstr(mnp) : "?";
            const char *kn = "?", *ks = "?";
            if (mk >= 0x10000u && !(mk & 3u)) {
                uint32_t np = ctx.mem.read32(mk + 0x28u), sp = ctx.mem.read32(mk + 0x2cu);
                if (np && np < 0xf0000000u) kn = ctx.mem.cstr(np);
                if (sp && sp < 0xf0000000u) ks = ctx.mem.cstr(sp);
            }
            
            uint32_t image = (mk >= 0x10000u && !(mk & 3u)) ? ctx.mem.read32(mk + 0x24u) : 0;
            uint32_t mdrows = (image >= 0x10000u && !(image & 3u)) ? (ctx.mem.read32(image + 0xb0u) & 0xffffffu) : 0;
            uint32_t rid = token & 0xffffffu;
            uint32_t tbl = token >> 24;
            
            uint32_t td_base = 0, td_rows = 0, td_rs = 0, md_base = 0, md_rs = 0;
            if (image >= 0x10000u && !(image & 3u)) {
                td_base = ctx.mem.read32(image + 0x7cu);
                td_rows = ctx.mem.read32(image + 0x80u) & 0xffffffu;
                td_rs   = (ctx.mem.read32(image + 0x80u) >> 24) & 0xffu;
                md_base = ctx.mem.read32(image + 0xacu);
                md_rs   = (ctx.mem.read32(image + 0xb0u) >> 24) & 0xffu;
            }
            if (verbose)
                fprintf(stderr, "[detour]   getmethod_in: caller=%s.%s::%s token=0x%08x "
                        "(tbl=0x%02x rid=%u) image=0x%08x MethodDef[base=0x%08x rows=%u rs=%u] "
                        "TypeDef[base=0x%08x rows=%u rs=%u] %s\n",
                        ks, kn, mn, token, tbl, rid, image, md_base, mdrows, md_rs,
                        td_base, td_rows, td_rs,
                        (tbl == 6 && rid > mdrows) ? "← RID > rows (BOUNDS FAIL)" : "");
        }
        
        if (strstr(nm, "getmethod_ret")) {
            uint32_t result = r0;
            uint32_t fp = regs[11];
            uint32_t token = (fp >= 0x10000u) ? ctx.mem.read32(fp - 0x14u) : 0;
            const char *rn = "?", *rk = "?", *rs = "?";
            if (result >= 0x10000u && !(result & 3u)) {
                uint32_t rnp = ctx.mem.read32(result + 0x10u);
                uint32_t rmk = ctx.mem.read32(result + 0x8u);
                if (rnp && rnp < 0xf0000000u) rn = ctx.mem.cstr(rnp);
                if (rmk >= 0x10000u && !(rmk & 3u)) {
                    uint32_t np = ctx.mem.read32(rmk + 0x28u), sp = ctx.mem.read32(rmk + 0x2cu);
                    if (np && np < 0xf0000000u) rk = ctx.mem.cstr(np);
                    if (sp && sp < 0xf0000000u) rs = ctx.mem.cstr(sp);
                }
            }
            static int gr = 0;
            if (result == 0 || gr++ < 20)
                fprintf(stderr, "[detour]   getmethod_ret: token=0x%08x -> %s%s.%s::%s\n",
                        token, result == 0 ? "NULL (RESOLUTION FAILED) " : "", rs, rk, rn);
        }
        
        if (strstr(nm, "typedef_ret")) {
            static int t = 0;
            if (r0 == 0 || t++ < 12)
                fprintf(stderr, "[detour]   typedef_ret: mono_metadata_typedef_from_method -> 0x%08x%s\n",
                        r0, r0 == 0 ? " (FAILED: no owning TypeDef found)" : "");
        }
        if (strstr(nm, "mclass_ret")) {
            uint32_t cls = r0;
            const char *kn = "?", *ks = "?";
            if (cls >= 0x10000u && !(cls & 3u)) {
                uint32_t np = ctx.mem.read32(cls + 0x28u), sp = ctx.mem.read32(cls + 0x2cu);
                if (np && np < 0xf0000000u) kn = ctx.mem.cstr(np);
                if (sp && sp < 0xf0000000u) ks = ctx.mem.cstr(sp);
            }
            static int m = 0;
            if (cls == 0 || m++ < 12)
                fprintf(stderr, "[detour]   mclass_ret: mono_class_get -> 0x%08x %s%s.%s\n",
                        cls, cls == 0 ? "(FAILED) " : "", ks, kn);
        }
        
        if (strstr(nm, "jit_cfg_watch")) {
            static bool s_jit_armed = false;
            if (!s_jit_armed && getenv("LUNARIA_WATCH_EXC_TYPE")) {
                uint32_t cfg = ctx.mem.read32(regs[11] - 0xa8u);
                if (cfg >= 0x10000u && !(cfg & 3u)) {
                    g_watch_lo = cfg + 0x1a8u;
                    g_watch_hi = cfg + 0x1acu;
                    g_step_capture_done = false;
                    s_jit_armed = true;
                    fprintf(stderr, "[watch] jit_cfg_watch: armed on cfg=0x%08x "
                            "exception_type watch=[0x%08x,0x%08x)\n",
                            cfg, g_watch_lo, g_watch_hi);
                }
            }
        }
        
        if (strstr(nm, "jit_getheader_ret")) {
            static int jh = 0;
            if (jh++ < 12)
                fprintf(stderr, "[detour]   jit_getheader_ret: method_get_header → 0x%08x %s\n",
                        r0, r0 == 0 ? "(NULL → cfg->exception_type = loader_error.kind)" : "(ok → deeper compile)");
        }
        
        if (strstr(nm, "fn22c3c_entry")) {
            uint32_t kls = r0;
            const char *nn = "?", *ss = "?";
            uint8_t etype = 0;
            if (kls >= 0x10000u && !(kls & 3u)) {
                etype = ctx.mem.read32(kls + 0x1a8u) & 0xff;
                uint32_t np = ctx.mem.read32(kls + 0x28u);
                uint32_t sp = ctx.mem.read32(kls + 0x2cu);
                if (np && np < 0xf0000000u) nn = ctx.mem.cstr(np);
                if (sp && sp < 0xf0000000u) ss = ctx.mem.cstr(sp);
            }
            fprintf(stderr, "[trace]   fn22c3c_entry: r0=0x%08x etype=%u name=%s ns=%s\n",
                    kls, (unsigned)etype, nn, ss);
        }
        if (strstr(nm, "fn22c3c_lookup1")) {
            
            uint32_t S = r3;
            fprintf(stderr, "[trace]   fn22c3c_lookup1: result=0x%08x", S);
            if (S >= 0x10000u && !(S & 3u)) {
                uint32_t f0 = ctx.mem.read32(S + 0x00u);
                uint32_t f4 = ctx.mem.read32(S + 0x04u);
                uint32_t f8 = ctx.mem.read32(S + 0x08u);
                fprintf(stderr, " [+0]=0x%08x [+4]=0x%08x [+8]=0x%08x", f0, f4, f8);
                if (f8 >= 0x10000u && !(f8 & 3u)) {
                    uint8_t e8 = ctx.mem.read32(f8 + 0x1a8u) & 0xff;
                    uint32_t np8 = ctx.mem.read32(f8 + 0x28u);
                    const char *n8 = (np8 && np8 < 0xf0000000u) ? ctx.mem.cstr(np8) : "?";
                    fprintf(stderr, " → [+8].etype=%u name=%s", (unsigned)e8, n8);
                }
            }
            fprintf(stderr, "\n");
        }
        

        
        if (strstr(nm, "class_init_14f9c8") && verbose) {
            uint32_t kls = r0;
            uint32_t etype = (kls >= 0x10000u && !(kls & 3u)) ? ctx.mem.read32(kls + 0x1a8u) : 0xffu;
            const char *nn2 = "?", *ss2 = "?";
            if (kls >= 0x10000u && !(kls & 3u)) {
                uint32_t np = ctx.mem.read32(kls + 0x28u), sp = ctx.mem.read32(kls + 0x2cu);
                if (np && np < 0xf0000000u) nn2 = ctx.mem.cstr(np);
                if (sp && sp < 0xf0000000u) ss2 = ctx.mem.cstr(sp);
            }
            /* exception_type is a byte at +0x1a8; read as 32-bit (str word at 0x19080) */
            uint8_t etype8 = etype & 0xffu;
            fprintf(stderr, "[detour]   class_init: klass=0x%08x %s.%s exception_type=%u%s\n",
                    kls, ss2 ? ss2 : "?", nn2 ? nn2 : "?", etype8,
                    etype8 == 6u ? " ← TYPE_LOAD(6) already set on entry!" : "");
            /* Arm watch as early as possible: when class_init fires for OOM with etype=0. */
            static bool s_ci_watch_armed = false;
            if (!s_ci_watch_armed && etype8 == 0 && kls > 0x10000u &&
                    nn2 && strstr(nn2, "OutOfMemoryException") &&
                    (getenv("LUNARIA_WATCH") || getenv("LUNARIA_WATCH_EXC_TYPE"))) {
                g_watch_lo = kls + 0x1a8u;
                g_watch_hi = kls + 0x1acu;
                g_step_capture_done = false;
                s_ci_watch_armed = true;
                fprintf(stderr, "[watch] session22: armed at class_init for OOM: "
                        "klass=0x%08x watch=[0x%08x,0x%08x)\n",
                        kls, g_watch_lo, g_watch_hi);
            }
        }
        
        if (strstr(nm, "builder_21274")) {
            /* builder arg r0 wraps the class at r0->[0x8] (the mono_class_init arg). */
            uint32_t kls = (r0 >= 0x10000u && !(r0 & 3u)) ? ctx.mem.read32(r0 + 0x8u) : 0;
            uint32_t etype = (kls >= 0x10000u && !(kls & 3u)) ? ctx.mem.read32(kls + 0x1a8u) : 0xffffffff;
            const char *nn = "?", *ss = "?";
            if (kls >= 0x10000u && !(kls & 3u)) {
                uint32_t np = ctx.mem.read32(kls + 0x28u), sp = ctx.mem.read32(kls + 0x2cu);
                if (np && np < 0xf0000000u) nn = ctx.mem.cstr(np);
                if (sp && sp < 0xf0000000u) ss = ctx.mem.cstr(sp);
            }
            uint8_t etype8 = etype & 0xffu;
            if (verbose)
            fprintf(stderr, "[detour]   builder class=%s.%s exception_type(+0x1a8)=%u%s\n",
                    ss ? ss : "?", nn ? nn : "?", etype8,
                    etype8 == 6u ? " (TYPE_LOAD=6 already set on entry!)" : "");
            
            static bool s_exc_watch_armed = false;
            if (!s_exc_watch_armed && etype8 == 0 && kls > 0x10000u &&
                    nn && strstr(nn, "OutOfMemoryException") &&
                    (getenv("LUNARIA_WATCH") || getenv("LUNARIA_WATCH_EXC_TYPE"))) {
                g_watch_lo = kls + 0x1a8u;
                g_watch_hi = kls + 0x1acu;
                g_step_capture_done = false;
                s_exc_watch_armed = true;
                fprintf(stderr, "[watch] session22: armed at builder for OOM: "
                        "klass=0x%08x watch=[0x%08x,0x%08x)\n",
                        kls, g_watch_lo, g_watch_hi);
            }
        }
        
        if (strstr(nm, "exc_from_name_msg") && verbose) {
            const char *ns = (r2 && r2 < 0xf0000000u) ? ctx.mem.cstr(r2) : "?";
            const char *na = (r3 && r3 < 0xf0000000u) ? ctx.mem.cstr(r3) : "?";
            uint32_t msg = (regs[13] ? ctx.mem.read32(regs[13]) : 0);  /* 5th arg on stack */
            const char *ms = (msg && msg < 0xf0000000u) ? ctx.mem.cstr(msg) : nullptr;
            fprintf(stderr, "[detour]   exception = %s.%s  msg=%s\n",
                    ns ? ns : "?", na ? na : "?", ms ? ms : "(null)");
        }
        
        if (strstr(nm, "lockwrap") && verbose) {
            
            fprintf(stderr, "[detour]   lockwrap fp-chain (key=0x%08x):", r1);
            /* APCS: add r11,sp,#k  =>  [r11]=saved lr (ret), [r11-4]=prev fp. */
            uint32_t fp = regs[11];
            for (int i = 0; i < 24 && fp >= 0x40000000u && fp < 0x48000000u; ++i) {
                uint32_t ret = ctx.mem.read32(fp) & ~1u;
                fprintf(stderr, " %08x", ret);
                uint32_t nfp = ctx.mem.read32(fp - 4u);
                if (nfp <= fp) break;      /* must climb toward stacktop */
                fp = nfp;
            }
            fprintf(stderr, "\n");
        }
        /* For throw/raise, try to name the exception: obj->vtable->class->name. */
        if (verbose && (strstr(nm, "throw") || strstr(nm, "raise"))) {
            uint32_t cand[4] = { r0, r1, r2, r3 };
            for (int c = 0; c < 4; ++c) {
                uint32_t obj = cand[c];
                if (obj < 0x10000u || (obj & 3u)) continue;
                uint32_t vt  = ctx.mem.read32(obj);
                if (vt < 0x10000u || (vt & 3u)) continue;
                uint32_t kls = ctx.mem.read32(vt);
                if (kls < 0x10000u || (kls & 3u)) continue;
                for (uint32_t off = 0x28; off <= 0x44; off += 4) {
                    uint32_t p = ctx.mem.read32(kls + off);
                    const uint8_t *hp = p ? ctx.mem.ptr(p) : nullptr;
                    if (!hp) continue;
                    char buf[64]; size_t i = 0;
                    for (; i < sizeof(buf)-1 && hp[i] >= 0x20 && hp[i] < 0x7f; ++i)
                        buf[i] = (char)hp[i];
                    buf[i] = 0;
                    if (i >= 4)
                        fprintf(stderr, "[detour]   r%d=0x%08x class+0x%02x -> '%s'\n",
                                c, obj, off, buf);
                }
            }
        }
        
        if (strstr(nm, "GC_push_all_stack") && (r1 - r0) > 0x100000u) {
            static const int NH = 96;
            uint32_t hva[NH] = {}, hcnt[NH] = {};
            uint32_t scanned = 0, distinct = 0;
            uint32_t hi = r0 + 0x40000u < r1 ? r0 + 0x40000u : r1; /* 256KB window */
            for (uint32_t a = r0; a + 4 <= hi; a += 4) {
                uint32_t w = ctx.mem.read32(a) & ~1u;
                if (w < 0x20015b08u || w >= 0x202e40d0u) continue; /* libmono .text */
                ++scanned;
                int slot = -1;
                for (int i = 0; i < NH; ++i) if (hva[i] == w && hcnt[i]) { slot = i; break; }
                if (slot < 0) for (int i = 0; i < NH; ++i) if (!hcnt[i]) { slot = i; hva[i] = w; ++distinct; break; }
                if (slot >= 0) ++hcnt[slot];
            }
            fprintf(stderr, "[detour]   GC deep-stack histogram range=%uMB scanned=%u distinct>=%u; top:\n",
                    (r1 - r0) >> 20, scanned, distinct);
            for (int rank = 0; rank < 8; ++rank) {  /* print 8 most frequent */
                int best = -1;
                for (int i = 0; i < NH; ++i) if (hcnt[i] && (best < 0 || hcnt[i] > hcnt[best])) best = i;
                if (best < 0 || hcnt[best] < 3) break;
                fprintf(stderr, "[detour]     ret=0x%08x x%u\n", hva[best], hcnt[best]);
                hcnt[best] = 0;
            }
        }
        return; /* regs untouched */
    }

    auto call_runnable_run = [&](uint32_t runnable_h) {
        if (!runnable_h) return;
        if ((uintptr_t)runnable_h > 65536u) {
            fprintf(stderr, "[arm_jni] call_runnable_run: handle 0x%x out of bounds, skip\n",
                    runnable_h);
            return;
        }
        arm_exec_run_runnable(runnable_h);
    };

    /* Helpers for casting 32-bit ARM handles to host pointer types */
#define AS_CLASS(x)   ((jclass)   (uintptr_t)(x))
#define AS_OBJ(x)     ((jobject)  (uintptr_t)(x))
#define AS_MID(x)     ((jmethodID)(uintptr_t)(x))
#define AS_FID(x)     ((jfieldID) (uintptr_t)(x))
#define AS_STR(x)     ((jstring)  (uintptr_t)(x))
#define AS_ARR(x)     ((jarray)   (uintptr_t)(x))
#define RET_OBJ(v)    ret32((uint32_t)(uintptr_t)(v))
#define ARM_STR(va)   (ctx.mem.cstr(va))

    /* Log actual EGL (263-281) and GL (284-406) SVC calls only */
    if ((svc_no >= 263 && svc_no <= 281) || (svc_no >= 284 && svc_no <= 406)) {
        static uint64_t egl_gl_count = 0;
        if (egl_gl_count < 200)
            fprintf(stderr, "[svc-eglgl] svc=%u r0=0x%x r1=0x%x tid=%u #%llu\n",
                    svc_no, r0, r1, g_current_tid,
                    (unsigned long long)egl_gl_count);
        ++egl_gl_count;
    }
    /* Trace SVCs immediately after a mscorlib access() success */
    {
        static int g_trace_after_mscorlib = 0;
        static int g_mscorlib_trace_limit = 0;
        if (svc_no == SVC_ACCESS) {
            const char *_p = ctx.mem.cstr(r0);
            if (_p && strstr(_p, "mscorlib")) {
                /* will be traced in SVC_ACCESS handler; arm host access result */
                if (::access(_p, (int)r1) == 0) {
                    g_trace_after_mscorlib = 20;
                    g_mscorlib_trace_limit = 0;
                }
            }
        }
        if (g_trace_after_mscorlib > 0 && svc_no != SVC_ACCESS) {
            g_trace_after_mscorlib--;
            if (g_mscorlib_trace_limit++ < 100)
                fprintf(stderr, "[SVC_AFTER_MSCORLIB] svc=%u r0=0x%08x r1=0x%08x r2=0x%08x lr=0x%08x\n",
                        svc_no, r0, r1, r2, regs[14]);
        }
    }
    /* Cooperative scheduling fallback: specific SVCs (futex, sem_wait, nanosleep)
 * already schedule worker threads.  But when the main thread (tid=0) is in a
 * tight loop calling OTHER SVCs (e.g. localtime while waiting for a flag set
 * by a worker), workers never run until the next call to run_pending_threads(),
 * which only happens between frames.  Schedule them periodically here so any
 * worker-thread dependency can complete. */
    if (g_current_tid == 0 && !g_scheduling && !g_threads.empty()) {
        static uint32_t svc_sched_ctr = 0;
        if (++svc_sched_ctr >= 10'000u) {
            svc_sched_ctr = 0;
            schedule_threads(500'000ULL);
        }
    }

    switch (svc_no) {
    /* 0: ARM32 Linux raw syscall — SVC #0 with r7 = NR_xxx */
    case 0: {
        uint32_t nr = regs[7];
        switch (nr) {
        case 90:  /* __NR_mmap  (addr,len,prot,flags,fd in r0-r4, off in r5 bytes) */
        case 192: { /* __NR_mmap2 (same but r5 = offset in pages) */
            uint32_t fd_raw = regs[4];
            uint32_t off    = regs[5];
            if (nr == 192) off <<= 12;
            uint32_t addr;
            if (r0 && (r3 & 0x10u)) { /* MAP_FIXED */
                addr = r0;
            } else {
                addr = mmap_bump(r1);
                if (addr == ~0u) { ret32(~0u); break; }
            }
            if ((int32_t)fd_raw >= 0 && !(r3 & 0x20u)) { /* file-backed */
                if (pread((int)fd_raw, ctx.mem.ptr(addr), r1, (off_t)(int64_t)(uint64_t)off) < 0) {
                    if (getenv("LUNARIA_TRACE_MMAP"))
                        fprintf(stderr, "[mmap0] pread failed: fd=%d off=0x%llx errno=%d\n",
                                (int32_t)fd_raw, (unsigned long long)off, errno);
                }
            }
            if (getenv("LUNARIA_TRACE_MMAP"))
                fprintf(stderr, "[mmap0] nr=%u r1=%u prot=%u flags=%#x fd=%d off=%u -> %#x\n",
                        nr, r1, r2, r3, (int32_t)fd_raw, off, addr);
            ret32(addr);
            break;
        }
        case 91:  /* __NR_munmap */
            ret32(0); break;
        case 20:  /* __NR_getpid */
            ret32(1); break;
        case 5:   /* __NR_open(path, flags, mode) */
        case 322: { /* __NR_openat(dfd, path, flags, mode) */
            const char *path = ctx.mem.cstr(nr == 5 ? r0 : r1);
            int flags = (int)(nr == 5 ? r1 : r2);
            mode_t mode = (mode_t)(nr == 5 ? r2 : r3);
            int fd = guest_open_path(path, flags, mode);
            ret32((uint32_t)fd);
            break;
        }
        case 224: /* __NR_gettid */
            ret32(g_current_tid ? g_current_tid : 1u); break;
        case 0xf0002: { /* __ARM_NR_cacheflush(start=r0, end=r1, flags=r2)
 * mono JIT calls this after backpatching call sites.
 * Invalidate dynarmic's translation cache for the range
 * so the new instructions are picked up on next execution. */
            uint32_t flush_start = r0 & ~3u;
            uint32_t flush_end   = (r1 + 3u) & ~3u;
            {
                static uint64_t cf_count = 0;
                static uint32_t cf_last_start = 0, cf_last_end = 0;
                static uint64_t cf_repeat = 0;
                ++cf_count;
                if (flush_start == cf_last_start && flush_end == cf_last_end)
                    ++cf_repeat;
                else
                    cf_repeat = 0;
                cf_last_start = flush_start;
                cf_last_end   = flush_end;
                if (cf_count <= 20 || cf_count % 10000 == 0 || cf_repeat == 10)
                    fprintf(stderr, "[cacheflush] #%llu: [0x%08x,0x%08x) sz=%u repeat=%llu\n",
                            (unsigned long long)cf_count,
                            flush_start, flush_end, flush_end - flush_start,
                            (unsigned long long)cf_repeat);
            }
            if (flush_end > flush_start && flush_end - flush_start <= 0x100000u) {
                if (ctx.jit) ctx.jit->InvalidateCacheRange(flush_start, flush_end - flush_start);
                if (ctx.jit_aux)
                    ctx.jit_aux->InvalidateCacheRange(flush_start, flush_end - flush_start);
                /* jit_cb runs only static library comparators, not JIT output: skip */
            }
            ret32(0); break;
        }
        case 240: { /* __NR_futex: ARM raw-syscall layout r0=uaddr, r1=op, r2=val, r7=240
 * Mirror the SVC_SYSCALL futex path (which handles syscall() PLT calls)
 * but with ARM raw-svc argument registers instead of r0=nr,r1=uaddr. */
            uint32_t cmd = r1 & 0x7fu; /* strip FUTEX_PRIVATE_FLAG */
            if (cmd != 0u && cmd != 9u) { /* FUTEX_WAKE / FUTEX_REQUEUE etc. */
                uint32_t nwake = (r2 == 0u) ? 1u : std::min(r2, 256u);
                g_futex_wake_tokens[r0] += nwake;
                auto wa = g_futex_wait_addrs.find(r0);
                if (wa != g_futex_wait_addrs.end()) {
                    uint32_t woke = std::min(nwake, wa->second);
                    wa->second -= woke;
                    if (wa->second == 0u) g_futex_wait_addrs.erase(wa);
                }
                if (getenv("LUNARIA_TRACE_FUTEX")) {
                    static uint64_t wkr = 0;
                    if (wkr < 256)
                        fprintf(stderr, "[futex/raw] WAKE cmd=%u uaddr=0x%x word=0x%x nwake=%u tid=%u tokens=%u\n",
                                cmd, r0, ctx.mem.read32(r0), nwake, g_current_tid,
                                g_futex_wake_tokens[r0]);
                    ++wkr;
                }
                ret32(0); break;
            }
            if (ctx.mem.read32(r0) != r2) { ret32(0); break; } /* value already changed */
            /* Check pending wake tokens */
            {
                auto tok = g_futex_wake_tokens.find(r0);
                if (tok != g_futex_wake_tokens.end() && tok->second > 0u) {
                    --tok->second;
                    if (tok->second == 0u) g_futex_wake_tokens.erase(tok);
                    ret32(0); break;
                }
            }
            bool changed = false;
            if (g_current_tid == 0 && !g_scheduling && !g_threads.empty()) {
                int spin_limit = (int)env_ticks("LUNARIA_FUTEX_SPINS", 500);
                for (int spin = 0; spin < spin_limit && !changed; ++spin) {
                    schedule_threads(20'000'000ULL);
                    drive_aaudio_callbacks(ctx);
                drive_java_choreographer(ctx);
                    drive_java_choreographer(ctx);
                    if (ctx.mem.read32(r0) != r2) changed = true;
                    auto tok = g_futex_wake_tokens.find(r0);
                    if (!changed && tok != g_futex_wake_tokens.end() && tok->second > 0u) {
                        --tok->second;
                        if (tok->second == 0u) g_futex_wake_tokens.erase(tok);
                        changed = true;
                    }
                }
            } else {
                /* non-main worker: park until a matching FUTEX_WAKE (see the
 * SVC_SYSCALL futex path for the rationale). */
                g_futex_wait_addrs[r0]++;
                for (auto &t : g_threads) {
                    if (t.id != g_current_tid) continue;
                    t.waiting_futex = r0;
                    t.futex_val     = r2;
                    break;
                }
                g_yield_requested = true;
                ret32(0);
                break;
            }
            if (changed) { ret32(0); break; }
            uint32_t eva = errno_va(ctx, g_current_tid);
            if (eva) ctx.mem.write32(eva, 110u /* ETIMEDOUT */);
            ret32(~0u);
            break;
        }
        default: {
            static uint32_t raw_ctr = 0;
            if (raw_ctr < 40 || (raw_ctr % 100000 == 0))
                fprintf(stderr, "[svc0] unhandled raw syscall nr=%u r0=0x%08x "
                        "r1=0x%08x lr=0x%08x tid=%u (#%u)\n",
                        nr, r0, r1, regs[14], g_current_tid, raw_ctr);
            ++raw_ctr;
            ret32(0); break;
        }
        }
        break;
    }
    /* 1-3: reserved */
    case 1: case 2: case 3: ret32(0); break;

    /* 4: GetVersion */
    case 4: ret32((uint32_t)jvm->native.GetVersion(env)); break;

    /* 5: DefineClass — stub */
    case 5: ret32(0); break;

    /* 6: FindClass(env, name_va) */
    case 6: {
        const char *name = ARM_STR(r1);
        if (!name || name[0] == '\0') { ret32(0); break; }
        static int fc_count = 0;
        if (fc_count < 40) {
            fprintf(stderr, "[arm_jni] FindClass: %s\n", name);
            ++fc_count;
        }
        jobject cls = jvm->native.FindClass(env, name);
        RET_OBJ(cls);
        break;
    }

    /* 7: FromReflectedMethod */
    case 7: RET_OBJ(jvm->native.FromReflectedMethod(env, AS_OBJ(r1))); break;
    /* 8: FromReflectedField */
    case 8: RET_OBJ(jvm->native.FromReflectedField(env, AS_OBJ(r1))); break;
    /* 9: ToReflectedMethod */
    case 9: ret32(0); break;
    /* 10: GetSuperclass */
    case 10: RET_OBJ(jvm->native.GetSuperclass(env, AS_CLASS(r1))); break;
    /* 11: IsAssignableFrom */
    case 11: ret32((uint32_t)jvm->native.IsAssignableFrom(
                    env, AS_CLASS(r1), AS_CLASS(r2))); break;
    /* 12: ToReflectedField */
    case 12: ret32(0); break;

    /* 13: Throw */
    case 13: ret32(0); break;
    /* 14: ThrowNew */
    case 14: ret32(0); break;
    /* 15: ExceptionOccurred */
    case 15: RET_OBJ(jvm->native.ExceptionOccurred(env)); break;
    /* 16: ExceptionDescribe */
    case 16: jvm->native.ExceptionDescribe(env); break;
    /* 17: ExceptionClear */
    case 17: jvm->native.ExceptionClear(env); break;
    /* 18: FatalError */
    case 18: fprintf(stderr,"[arm_exec] FatalError: %s\n", ARM_STR(r1)); break;
    /* 19: PushLocalFrame */
    case 19: ret32(0); break;
    /* 20: PopLocalFrame */
    case 20: RET_OBJ(AS_OBJ(r1)); break;

    /* 21: NewGlobalRef */
    case 21: RET_OBJ(jvm->native.NewGlobalRef(env, AS_OBJ(r1))); break;
    /* 22: DeleteGlobalRef */
    case 22: jvm->native.DeleteGlobalRef(env, AS_OBJ(r1)); break;
    /* 23: DeleteLocalRef */
    case 23: jvm->native.DeleteLocalRef(env, AS_OBJ(r1)); break;
    /* 24: IsSameObject */
    case 24: {
        jboolean same = jvm->native.IsSameObject(env, AS_OBJ(r1), AS_OBJ(r2));
        static int iso_diag = 0;
        if (iso_diag++ < 40)
            fprintf(stderr, "[arm_jni] IsSameObject(0x%x, 0x%x) → %d\n",
                    r1, r2, (int)same);
        ret32((uint32_t)same);
        break;
    }
    /* 25: NewLocalRef */
    case 25: RET_OBJ(jvm->native.NewLocalRef(env, AS_OBJ(r1))); break;
    /* 26: EnsureLocalCapacity */
    case 26: ret32(0); break;

    /* 27: AllocObject */
    case 27: RET_OBJ(jvm->native.AllocObject(env, AS_CLASS(r1))); break;
    /* 28-30: NewObject/V/A — just AllocObject (Scanner ctor linked to its
 * InputStream by the asset bridge so boot.config can be read). */
    case 28: case 29: case 30: {
        if (try_asset_jni(ctx, svc_no, regs)) break;
        jobject no = jvm->native.AllocObject(env, AS_CLASS(r1));
        static int no_diag = 0;
        if (no_diag++ < 60) {
            const char *cn = ((uintptr_t)AS_CLASS(r1) <= 65536u && r1)
                ? jvm_get_class_name(jvm, AS_CLASS(r1)) : nullptr;
            fprintf(stderr, "[arm_jni] NewObject(%s) → 0x%x (svc=%u)\n",
                    cn ? cn : "?", (uint32_t)(uintptr_t)no, svc_no);
        }
        RET_OBJ(no); break;
    }

    /* 31: GetObjectClass */
    case 31: RET_OBJ(jvm->native.GetObjectClass(env, AS_OBJ(r1))); break;
    /* 32: IsInstanceOf */
    case 32: ret32((uint32_t)jvm->native.IsInstanceOf(
                    env, AS_OBJ(r1), AS_CLASS(r2))); break;

    /* 33: GetMethodID(env, class, name, sig) */
    case 33: {
        const char *mn = ARM_STR(r2);
        const char *ms = ARM_STR(r3);
        jmethodID gm = jvm->native.GetMethodID(env, AS_CLASS(r1), mn, ms);
        static int gm_diag = 0;
        bool gm_hot = mn && (strstr(mn, "Message") || strstr(mn, "Frame") ||
                             strstr(mn, "handle") || strstr(mn, "doFrame"));
        if (gm_diag++ < 120 || gm_hot)
            fprintf(stderr, "[arm_jni] GetMethodID(%s %s) cls=0x%x → 0x%x\n",
                    mn ? mn : "?", ms ? ms : "?", r1, (uint32_t)(uintptr_t)gm);
        RET_OBJ(gm);
        break;
    }

    /* 34-36: CallObjectMethod/V/A */
    case 34: case 35: case 36: {
        auto mid = AS_MID(r2);
        uintptr_t midx = (uintptr_t)mid;
        const char *mname = "?";
        if (midx > 0 && midx <= 65536) {
            auto &mo = jvm->objects[midx - 1];
            if (mo.type == jvm_object::JVM_OBJECT_METHOD && mo.method.name.data)
                mname = mo.method.name.data;
        }
        /* ClassLoader.findLibrary(name): Android maps a bare name to the APK's
 * native-lib path.  Returning null here makes Unity conclude the install
 * is broken and block forever on a fatal AlertDialog, so resolve against
 * the main library's directory. */
        if (!strcmp(mname, "findLibrary")) {
            uint32_t str_h = jni_arg_word(ctx, regs, (int)svc_no - 34, 0);
            const char *lname = jvm->native.GetStringUTFChars(env, AS_STR(str_h), nullptr);
            char lpath[4096]; lpath[0] = 0;
            if (lname && *lname && !g_main_lib_dir.empty())
                snprintf(lpath, sizeof(lpath), "%s/lib%s.so",
                         g_main_lib_dir.c_str(), lname);
            if (lpath[0] && access(lpath, F_OK) == 0) {
                fprintf(stderr, "[arm_jni] findLibrary(\"%s\") -> %s\n", lname, lpath);
                RET_OBJ(jvm->native.NewStringUTF(env, lpath));
            } else {
                fprintf(stderr, "[arm_jni] findLibrary(\"%s\") -> null (checked %s)\n",
                        lname ? lname : "?", lpath[0] ? lpath : "(no dir)");
                ret32(0);
            }
            break;
        }
        /* HandlerThread.getLooper() / Handler.obtainMessage(): UnityChoreographer
         * builds a Java HandlerThread and gives up (waiting forever on its init
         * cond) when getLooper() returns null.  We don't run Java threads, but a
         * plausible Looper/Message object lets the ctor proceed to the message
         * post, which we execute natively (see sendMessage handling). */
        /* java.lang.reflect.Method stubs created for JNIBridge.invoke carry
         * their method name in g_method_stub_names; answer getName() with it. */
        if (!strcmp(mname, "getName")) {
            auto ns = g_method_stub_names.find(r1);
            if (ns != g_method_stub_names.end()) {
                fprintf(stderr, "[jnibridge] Method.getName(0x%x) → \"%s\"\n",
                        r1, ns->second.c_str());
                RET_OBJ(jvm->native.NewStringUTF(env, ns->second.c_str()));
                break;
            }
        }
        if (!strcmp(mname, "getLooper")) {
            static jobject looper_stub = nullptr;
            if (!looper_stub)
                looper_stub = jvm->native.AllocObject(env,
                    jvm->native.FindClass(env, "android/os/Looper"));
            fprintf(stderr, "[arm_jni] getLooper -> stub 0x%x\n",
                    (uint32_t)(uintptr_t)looper_stub);
            RET_OBJ(looper_stub);
            break;
        }
        if (!strcmp(mname, "obtainMessage")) {
            /* obtainMessage(int what[, ...]): record what so the guest's
             * handleMessage sees the right dispatch code via GetIntField. */
            int32_t what = (int32_t)jni_arg_word(ctx, regs, (int)svc_no - 34, 0);
            jobject msg = jvm->native.AllocObject(env,
                jvm->native.FindClass(env, "android/os/Message"));
            uint32_t mh = (uint32_t)(uintptr_t)msg;
            if (mh && mh <= 65536u) g_message_what[mh] = what;
            fprintf(stderr, "[arm_jni] obtainMessage(what=%d) -> 0x%x "
                    "(svc=%u r3=0x%x [r3]=0x%x [r3+4]=0x%x)\n", what, mh,
                    svc_no, regs[3],
                    regs[3] ? ctx.mem.read32(regs[3]) : 0,
                    regs[3] ? ctx.mem.read32(regs[3] + 4) : 0);
            RET_OBJ(msg);
            break;
        }
        if (try_asset_jni(ctx, svc_no, regs)) break;
        /* Dispatch to the host JVM, which resolves the
 * method to its native handler (libjvm-*.c) via dlsym.  These calls
 * (getApplicationInfo, getSharedPreferences, getClass, …) now have
 * handlers and return a valid object handle (1..65536) or NULL, so the
 * common case is success and is logged only under LUNARIA_TRACE_JNI.  A
 * return outside the objects-array range IS broken, though — it would
 * later trip jvm_get_object's assert — so always surface that. */
        if (mid && r1) {
            uint32_t res = (uint32_t)(uintptr_t)
                jvm->native.CallObjectMethodA(env, AS_OBJ(r1), mid, nullptr);
            if (res > 65536u)
                fprintf(stderr, "[arm_jni] CallObjectMethodA: %s returned broken handle "
                        "0x%x (mid=0x%x obj=0x%x)\n", mname, res,
                        (unsigned)(uintptr_t)mid, r1);
            else if (getenv("LUNARIA_TRACE_JNI"))
                fprintf(stderr, "[arm_jni] CallObjectMethodA: %s -> 0x%x (mid=0x%x obj=0x%x)\n",
                        mname, res, (unsigned)(uintptr_t)mid, r1);
            ret32(res);
        } else {
            if (getenv("LUNARIA_TRACE_JNI"))
                fprintf(stderr, "[arm_jni] CallObjectMethod: null mid/obj for %s, returning 0\n", mname);
            ret32(0);
        }
        break;
    }
    /* 37-39: CallBooleanMethod/V/A */
    case 37: case 38: case 39: {
        /* Most boolean JNI calls default to false, but Unity's asset-delivery
 * bootstrap asks PlayAssetDeliveryUnityWrapper.playCoreApiMissing():
 * on this emulated, Play-Core-less environment that is true, which
 * makes Unity load assets straight from the APK instead of waiting on
 * a Play Core API we don't provide. */
        auto mid = AS_MID(r2);
        uintptr_t midx = (uintptr_t)mid;
        const char *mname = nullptr;
        if (midx > 0 && midx <= 65536) {
            auto &mo = jvm->objects[midx - 1];
            if (mo.type == jvm_object::JVM_OBJECT_METHOD && mo.method.name.data)
                mname = mo.method.name.data;
        }
        if (try_asset_jni(ctx, svc_no, regs)) break;
        /* Handler.post(Runnable) / Handler.postDelayed(Runnable, long) — the
 * Runnable is the first method argument.  For the plain/V variants
 * (svc 37/38) it is passed directly in r3; for the "A" variant (svc 39)
 * r3 is a `const jvalue*` array on the guest stack, so the Runnable
 * handle lives in args[0].l = *(u32*)r3.  Previously svc 39 fed the raw
 * jvalue-array pointer (a stack address) into call_runnable_run, which
 * then rejected it as out-of-bounds and silently dropped the post —
 * stalling any init/UI continuation Unity scheduled this way. */
        if (mname && (!strcmp(mname, "post") || !strcmp(mname, "postDelayed"))) {
            /* Locate the Runnable argument per JNI call variant:
 * svc 37  CallBooleanMethod   — varargs in regs: r3 = Runnable.
 * svc 38  CallBooleanMethodV  — r3 = va_list*  → first arg at *r3.
 * svc 39  CallBooleanMethodA  — r3 = jvalue*   → first arg at *r3.
 * The V/A forms put a guest *stack pointer* in r3, which previously
 * fell through as the handle and was dropped as out-of-bounds. */
            uint32_t runnable_h = ((svc_no == 38 || svc_no == 39) && r3)
                                      ? ctx.mem.read32(r3) : r3;
            static int post_diag = 0;
            if (post_diag++ < 4)
                fprintf(stderr, "[arm_jni] %s via svc %u: r3=0x%x -> runnable=0x%x\n",
                        mname, svc_no, r3, runnable_h);
            call_runnable_run(runnable_h);
            ret32(1); /* boolean true = posted */
            break;
        }
        /* ConcurrentLinkedQueue.add / offer — Unity 4 queues GL-thread Runnables here.
 * Defer execution until Activity.executeGLThreadJobs() drains the queue. */
        if (mname && (!strcmp(mname, "add") || !strcmp(mname, "offer"))) {
            uint32_t elem_h = ((svc_no == 38 || svc_no == 39) && r3)
                                  ? ctx.mem.read32(r3) : r3;
            if (elem_h > 0 && elem_h <= 65536u && r1) {
                jclass qcls = jvm->native.GetObjectClass(env, AS_OBJ(r1));
                const char *qname = ((uintptr_t)qcls <= 65536u)
                    ? jvm_get_class_name(jvm, qcls) : nullptr;
                if (qname && (strstr(qname, "Queue") || strstr(qname, "Linked"))) {
                    arm_exec_gl_job_enqueue(elem_h);
                    if (getenv("LUNARIA_TRACE_GL"))
                        fprintf(stderr, "[gl] %s: queued Runnable 0x%x on %s\n",
                                mname, elem_h, qname);
                    ret32(1);
                    break;
                }
            }
        }
        uint32_t bret = 0;
        if (mname && !strcmp(mname, "playCoreApiMissing")) bret = 1;
        /* Unity nativeRecreateGfxState checks Surface.isValid() before
 * ANativeWindow_fromSurface / EGL setup.  Stub Surfaces from AllocObject
 * have no native peer, so the default boolean 0 made Gfx init a no-op. */
        if (mname && !strcmp(mname, "isValid")) bret = 1;
        /* Handler.sendMessage*/
        if (mname && strstr(mname, "sendMessage")) bret = 1;
        {
            static std::set<std::string> bm_seen;
            if (bm_seen.emplace(mname ? mname : "?").second)
                fprintf(stderr, "[arm_jni] CallBooleanMethod: %s obj=0x%x → %u (svc=%u, first)\n",
                        mname ? mname : "?", r1, bret, svc_no);
        }
        ret32(bret);
        break;
    }
    /* 40-42: CallByteMethod/V/A */
    case 40: case 41: case 42: ret32(0); break;
    /* 43-45: CallCharMethod/V/A */
    case 43: case 44: case 45: ret32(0); break;
    /* 46-48: CallShortMethod/V/A */
    case 46: case 47: case 48: ret32(0); break;
    /* 49-51: CallIntMethod/V/A — asset InputStream.read/available land here */
    case 49: case 50: case 51: {
        if (try_asset_jni(ctx, svc_no, regs)) break;
        /* Returning 0 makes Unity abort(); return actual JVM string length. */
        auto mid = AS_MID(r2);
        uintptr_t midx = (uintptr_t)mid;
        if (midx > 0 && midx <= 65536) {
            auto &mo = jvm->objects[midx - 1];
            if (mo.type == jvm_object::JVM_OBJECT_METHOD && mo.method.name.data &&
                strcmp(mo.method.name.data, "length") == 0) {
                ret32((uint32_t)jvm->native.GetStringUTFLength(env, AS_STR(r1)));
                break;
            }
        }
        /* Forward to the host JVM (dlsym'd libjvm-*.c handlers), same as
 * CallObjectMethod above.  Always returning 0 here broke touch input:
 * Unity's nativeInjectEvent reads InputEvent.getSource() through a
 * cached methodID, got 0 instead of SOURCE_TOUCHSCREEN (0x1002), and
 * dropped every injected MotionEvent before reading its coordinates. */
        if (mid && r1) { ret32((uint32_t)jvm->native.CallIntMethodA(env, AS_OBJ(r1), mid, nullptr)); break; }
        ret32(0);
        break;
    }
    /* 52-54: CallLongMethod/V/A — asset InputStream.skip lands here */
    case 52: case 53: case 54:
        if (try_asset_jni(ctx, svc_no, regs)) break;
        if (r1 && r2) { ret64((uint64_t)jvm->native.CallLongMethodA(env, AS_OBJ(r1), AS_MID(r2), nullptr)); break; }
        ret64(0); break;
    /* 55-57: CallFloatMethod/V/A — MotionEvent.getX/getY land here */
    case 55: case 56: case 57: {
        if (r1 && r2) {
            jfloat v = jvm->native.CallFloatMethodA(env, AS_OBJ(r1), AS_MID(r2), nullptr);
            uint32_t bits; memcpy(&bits, &v, 4);
            ret32(bits);
            break;
        }
        ret32(0); break;
    }
    /* 58-60: CallDoubleMethod/V/A */
    case 58: case 59: case 60: {
        if (r1 && r2) {
            jdouble v = jvm->native.CallDoubleMethodA(env, AS_OBJ(r1), AS_MID(r2), nullptr);
            uint64_t bits; memcpy(&bits, &v, 8);
            ret64(bits);
            break;
        }
        ret64(0); break;
    }
    /* 61-63: CallVoidMethod/V/A */
    case 61: case 62: case 63: {
        if (try_asset_jni(ctx, svc_no, regs)) break;
        if (!r2) break;
        const char *mname = nullptr;
        uintptr_t midx = r2;
        if (midx > 0 && midx <= 65536) {
            auto &mo = jvm->objects[midx - 1];
            if (mo.type == jvm_object::JVM_OBJECT_METHOD && mo.method.name.data)
                mname = mo.method.name.data;
        }
        
        if (mname && !strcmp(mname, "runOnUiThread")) {
            /* Same JNI-variant handling as Handler.post: svc 62 (V) / 63 (A)
 * pass a guest stack pointer (va_list or jvalue array) in r3, so
 * the Runnable handle is at *r3; svc 61 passes it directly in r3. */
            uint32_t runnable_h = ((svc_no == 62 || svc_no == 63) && r3)
                                      ? ctx.mem.read32(r3) : r3;
            if (getenv("LUNARIA_TRACE_JNI"))
                fprintf(stderr, "[arm_jni] runOnUiThread: svc=%u r1=0x%x r3=0x%x *r3=0x%x -> h=0x%x\n",
                        svc_no, r1, r3, r3 ? ctx.mem.read32(r3) : 0u, runnable_h);
            if (runnable_h > 0 && runnable_h <= 65536)
                call_runnable_run(runnable_h);
            else if (runnable_h != 0)
                /* Non-zero but outside the objects array = a corrupt handle worth flagging. */
                fprintf(stderr, "[arm_jni] runOnUiThread: handle 0x%x out of bounds, skipping\n",
                        runnable_h);
            /* runnable_h == 0: Unity genuinely passed a null Runnable (verified
 * svc=62 CallVoidMethodV with *va_list==0); nothing to run. */
            break;
        }
        if (mname && !strcmp(mname, "executeGLThreadJobs")) {
            if (getenv("LUNARIA_TRACE_GL"))
                fprintf(stderr, "[gl] executeGLThreadJobs (JNI svc %u)\n", svc_no);
            arm_exec_drain_gl_thread_jobs();
            break;
        }
        if (!r1) {
            /* NULL receiver would trip the host JVM assert; guests (FMOD's
             * AudioTrack path) call through cached objects without checking. */
            fprintf(stderr, "[arm_jni] CallVoidMethod: NULL obj (mid=0x%08x "
                    "name=%s lr=0x%08x) — ignored\n",
                    r2, mname ? mname : "?", regs[14]);
            break;
        }
        /* Choreographer.postFrameCallback(cb): registration usually arrives
         * from handleMessage running on the callback JIT, so the doFrame
         * cannot be delivered inline — queue it for the drive loops. */
        if (mname && !strcmp(mname, "postFrameCallback")) {
            ++g_java_choreo_pending;
            static int pfc_n = 0;
            if (pfc_n++ < 8)
                fprintf(stderr, "[jnibridge] postFrameCallback queued (pending=%u)\n",
                        g_java_choreo_pending);
            break;
        }
        if (mname && !strcmp(mname, "removeFrameCallback")) {
            g_java_choreo_pending = 0;
            break;
        }
        /* Message.sendToTarget(): with no Java Looper thread to dispatch it,
         * re-enter the JNIBridge proxy's native invoke() directly, i.e. run
         * Handler$Callback.handleMessage(msg) on the caller's thread.  This is
         * what unblocks UnityChoreographer's init-done wait. */
        if (mname && !strcmp(mname, "sendToTarget") && g_jnibridge_last_ptr) {
            uint32_t fn = 0;
            for (auto &rn : g_ctx->natives)
                if (rn.name == "invoke" && strstr(rn.klass.c_str(), "JNIBridge"))
                    { fn = rn.fn_va; break; }
            if (fn) {
                static uint32_t br_cls = 0;
                if (!br_cls) br_cls = (uint32_t)(uintptr_t)jvm->native.FindClass(
                                          env, "bitter/jnibridge/JNIBridge");
                /* Pick each proxy by its recorded interfaces[0]: the native
                 * matcher chain in invoke() caches that exact class handle at
                 * proxy creation and compares with IsSameObject, so a fresh
                 * FindClass of the same name would not match.  GetMethodID on
                 * the recorded class + FromReflectedMethod (identity in the
                 * pseudo-JVM) then reproduces the jmethodID the matcher
                 * initialised through its __cxa_guard block. */
                uint64_t hm_ptr = 0, fc_ptr = 0;
                uint32_t hm_cls = 0, fc_cls = 0;
                for (auto &it : g_jnibridge_iface) {
                    for (auto &rec : it.second) {
                        if (rec.name.find("Handler$Callback") != std::string::npos)
                            { hm_ptr = it.first; hm_cls = rec.cls; }
                        if (rec.name.find("Choreographer$FrameCallback") != std::string::npos)
                            { fc_ptr = it.first; fc_cls = rec.cls; }
                    }
                }
                if (!hm_ptr) hm_ptr = g_jnibridge_last_ptr;
                if (hm_ptr && hm_cls) {
                    static uint32_t mth = 0;
                    if (!mth) {
                        mth = (uint32_t)(uintptr_t)jvm->native.GetMethodID(
                            env, (jclass)(uintptr_t)hm_cls, "handleMessage",
                            "(Landroid/os/Message;)Z");
                        if (mth && mth <= 65536u)
                            g_method_stub_names[mth] = "handleMessage";
                    }
                    fprintf(stderr, "[jnibridge] sendToTarget msg=0x%x → invoke(ptr=0x%llx, "
                            "cls=0x%x mid=0x%x) @0x%08x\n",
                            r1, (unsigned long long)hm_ptr, hm_cls, mth, fn);
                    /* invoke(JNIEnv, jclass, jlong ptr, jclass intf, jobject
                     * method, jobjectArray args): jlong → r2:r3, rest on the
                     * stack.  Runs on the standalone callback JIT — the main
                     * JIT is mid-Run() here.  args = { msg }. */
                    jobjectArray args = jvm->native.NewObjectArray(
                        env, 1, (jclass)(uintptr_t)hm_cls, nullptr);
                    jvm->native.SetObjectArrayElement(env, args, 0, AS_OBJ(r1));
                    uint32_t stk[3] = { hm_cls, mth, (uint32_t)(uintptr_t)args };
                    int32_t rc = call_guest_cb(ctx, fn, ENV_SLOT_BASE, br_cls,
                                               (uint32_t)hm_ptr,
                                               (uint32_t)(hm_ptr >> 32),
                                               stk, 3);
                    fprintf(stderr, "[jnibridge] invoke(handleMessage) returned 0x%x\n",
                            (uint32_t)rc);
                }
                /* The handler may have called Choreographer.postFrameCallback
                 * (deferred because the callback JIT was busy) — deliver the
                 * doFrame now that it is free again. */
                (void)fc_ptr; (void)fc_cls;
                drive_java_choreographer(ctx);
                break;
            }
        }
        {
            static std::set<std::string> vm_seen;
            if (vm_seen.emplace(mname ? mname : "?").second)
                fprintf(stderr, "[arm_jni] CallVoidMethod: %s obj=0x%x (svc=%u, first)\n",
                        mname ? mname : "?", r1, svc_no);
        }
        jvm->native.CallVoidMethodA(env, AS_OBJ(r1), AS_MID(r2), nullptr);
        break;
    }

    /* 64-93: CallNonvirtual* stubs */
    case 64 ... 90: ret32(0); break;
    case 91: case 92: case 93: break; /* CallNonvirtualVoidMethod/V/A */

    /* 94: GetFieldID(env, class, name, sig) */
    case 94: RET_OBJ(jvm->native.GetFieldID(env, AS_CLASS(r1),
                    ARM_STR(r2), ARM_STR(r3))); break;

    /* 95-103: GetXxxField */
    case 95:  RET_OBJ(jvm->native.GetObjectField(env,AS_OBJ(r1),AS_FID(r2))); break;
    case 96:  ret32((uint32_t)jvm->native.GetBooleanField(env,AS_OBJ(r1),AS_FID(r2))); break;
    case 97:  ret32((uint32_t)jvm->native.GetByteField(env,AS_OBJ(r1),AS_FID(r2))); break;
    case 98:  ret32((uint32_t)jvm->native.GetCharField(env,AS_OBJ(r1),AS_FID(r2))); break;
    case 99:  ret32((uint32_t)jvm->native.GetShortField(env,AS_OBJ(r1),AS_FID(r2))); break;
    case 100: {
        /* Message.what for stub Messages created by our obtainMessage */
        uintptr_t fidx = (uintptr_t)AS_FID(r2);
        if (fidx > 0 && fidx <= 65536) {
            auto &fo = jvm->objects[fidx - 1];
            if (fo.type == jvm_object::JVM_OBJECT_METHOD && fo.method.name.data &&
                !strcmp(fo.method.name.data, "what")) {
                auto wi = g_message_what.find(r1);
                if (wi != g_message_what.end()) {
                    fprintf(stderr, "[arm_jni] GetIntField(msg=0x%x, what) → %d\n",
                            r1, wi->second);
                    ret32((uint32_t)wi->second);
                    break;
                }
            }
        }
        ret32((uint32_t)jvm->native.GetIntField(env,AS_OBJ(r1),AS_FID(r2)));
        break;
    }
    case 101: ret64(0); break; /* GetLongField */
    case 102: ret32(0); break; /* GetFloatField */
    case 103: ret64(0); break; /* GetDoubleField */

    /* 104-112: SetXxxField stubs */
    case 104 ... 112: break;

    /* 113: GetStaticMethodID */
    case 113: RET_OBJ(jvm->native.GetStaticMethodID(env, AS_CLASS(r1),
                      ARM_STR(r2), ARM_STR(r3))); break;

    /* 114-143: CallStatic* stubs */
    /* CallStaticObjectMethod/V/A — dispatch special-cased static calls that need
 * argument marshaling, then fall back to generic jvm_wrap_method dispatch. */
    case 114: case 115: case 116: {
        if (try_asset_jni(ctx, svc_no, regs)) break;
        auto mid = AS_MID(r2);
        if (!mid) { ret32(0); break; }
        /* Peek at the method name to special-case forName and init */
        uintptr_t midx = (uintptr_t)mid;
        const char *mname = nullptr, *msig = nullptr;
        if (midx > 0 && midx <= 65536) {
            auto &mo = jvm->objects[midx - 1];
            if (mo.type == jvm_object::JVM_OBJECT_METHOD) {
                mname = mo.method.name.data;
                msig  = mo.method.signature.data;
            }
        }
        /* ReflectionHelper.newProxyInstance(UnityPlayer, J, Class|[Class) —
 * create a proxy object and remember its native InvocationHandler
 * pointer so run()/method calls can re-enter guest code via
 * nativeProxyInvoke (see arm_exec_run_runnable). */
        if (mname && !strcmp(mname, "newProxyInstance")) {
            uint64_t ptr = 0;
            uint32_t cls_h = 0;
            int variant = (int)svc_no - 114;
            if (variant == 1) {          /* V: r3 = va_list ap (guest VA) */
                uint32_t ap = regs[3] + 4;      /* skip player handle */
                ap = (ap + 7u) & ~7u;           /* va_arg(jlong): 8-byte align */
                ptr = (uint64_t)ctx.mem.read32(ap) |
                      ((uint64_t)ctx.mem.read32(ap + 4) << 32);
                cls_h = ctx.mem.read32(ap + 8);
            } else if (variant == 2) {   /* A: jvalue[] (8 bytes per slot) */
                uint32_t a = regs[3];
                ptr = (uint64_t)ctx.mem.read32(a + 8) |
                      ((uint64_t)ctx.mem.read32(a + 12) << 32);
                cls_h = ctx.mem.read32(a + 16);
            } else {                     /* variadic: player=r3, jlong 8-aligned on stack */
                uint32_t sp = regs[13];
                ptr = (uint64_t)ctx.mem.read32(sp) |
                      ((uint64_t)ctx.mem.read32(sp + 4) << 32);
                cls_h = ctx.mem.read32(sp + 8);
            }
            jobject prox = jvm->native.AllocObject(env,
                jvm->native.FindClass(env, "java/lang/reflect/Proxy"));
            uint32_t h = (uint32_t)(uintptr_t)prox;
            if (h && h <= 65536u) g_jproxy_ptrs[h] = ptr;
            fprintf(stderr, "[jproxy] newProxyInstance svc=%u ptr=0x%llx ifaces=0x%x -> h=0x%x\n",
                    svc_no, (unsigned long long)ptr, cls_h, h);
            RET_OBJ(prox);
            break;
        }
        /* JNIBridge.newInterfaceProxy(jlong ptr, Class[] interfaces) — Java
         * Proxy whose calls route to native JNIBridge.invoke(ptr, ...). */
        if (mname && !strcmp(mname, "newInterfaceProxy")) {
            uint64_t ptr = 0;
            uint32_t ifaces_h = 0;
            int variant = (int)svc_no - 114;
            if (variant == 1) {          /* V: r3 = va_list; jlong 8-aligned */
                uint32_t ap = (regs[3] + 7u) & ~7u;
                ptr = (uint64_t)ctx.mem.read32(ap) |
                      ((uint64_t)ctx.mem.read32(ap + 4) << 32);
                ifaces_h = ctx.mem.read32(ap + 8);
            } else if (variant == 2) {   /* A: jvalue[0].j, jvalue[1].l */
                ptr = (uint64_t)ctx.mem.read32(regs[3]) |
                      ((uint64_t)ctx.mem.read32(regs[3] + 4) << 32);
                ifaces_h = ctx.mem.read32(regs[3] + 8);
            } else {                     /* variadic: jlong lands on the stack */
                uint32_t sp = regs[13];
                ptr = (uint64_t)ctx.mem.read32(sp) |
                      ((uint64_t)ctx.mem.read32(sp + 4) << 32);
                ifaces_h = ctx.mem.read32(sp + 8);
            }
            /* Record every interface: the native matcher chain in invoke()
             * compares its cached class against these exact handles with
             * IsSameObject, so dispatch must reuse them — a later FindClass
             * of the same name may return a different handle. */
            std::vector<JniBridgeIface> recs;
            if (ifaces_h) {
                int n = (int)jvm->native.GetArrayLength(
                    env, (jarray)(uintptr_t)ifaces_h);
                for (int i = 0; i < n && i < 8; ++i) {
                    jobject icls = jvm->native.GetObjectArrayElement(
                        env, (jobjectArray)(uintptr_t)ifaces_h, i);
                    if (!icls) continue;
                    JniBridgeIface rec;
                    rec.cls = (uint32_t)(uintptr_t)icls;
                    const char *cn = jvm_get_class_name(jvm, icls);
                    if (cn) rec.name = cn;
                    recs.push_back(std::move(rec));
                }
            }
            jobject prox = jvm->native.AllocObject(env,
                jvm->native.FindClass(env, "java/lang/reflect/Proxy"));
            uint32_t h = (uint32_t)(uintptr_t)prox;
            if (h && h <= 65536u) {
                g_jnibridge_ptrs[h] = ptr;
                g_jnibridge_last_ptr = ptr;
                g_jnibridge_order.push_back(ptr);
                g_jnibridge_iface[ptr] = recs;
            }
            fprintf(stderr, "[jnibridge] newInterfaceProxy svc=%u ptr=0x%llx ifaces={",
                    svc_no, (unsigned long long)ptr);
            for (size_t i = 0; i < recs.size(); ++i)
                fprintf(stderr, "%s0x%x(%s)", i ? ", " : "", recs[i].cls,
                        recs[i].name.empty() ? "?" : recs[i].name.c_str());
            fprintf(stderr, "} -> h=0x%x\n", h);
            RET_OBJ(prox);
            break;
        }
        /* Class.forName(String) — read class name string from first arg */
        if (mname && !strcmp(mname, "forName")) {
            uint32_t str_h = jni_arg_word(ctx, regs, (int)svc_no - 114, 0);
            const char *utf = jvm->native.GetStringUTFChars(env, (jstring)(uintptr_t)str_h, nullptr);
            if (utf && *utf) {
                if (getenv("LUNARIA_TRACE_JNI"))
                    fprintf(stderr, "[forName] Class.forName(\"%s\")\n", utf);
                RET_OBJ(jvm->native.FindClass(env, utf));
            } else {
                ret32(0);
            }
            break;
        }
        /* PlayAssetDeliveryUnityWrapper.init(UnityPlayer, Context) — return stub */
        if (mname && !strcmp(mname, "init") && msig &&
            strstr(msig, "PlayAssetDeliveryUnityWrapper")) {
            static jobject pad_stub = nullptr;
            if (!pad_stub)
                pad_stub = jvm->native.AllocObject(env,
                    jvm->native.FindClass(env, "com/unity3d/player/PlayAssetDeliveryUnityWrapper"));
            RET_OBJ(pad_stub);
            break;
        }
        /* Generic dispatch with nullptr args (works for zero-arg static methods) */
        RET_OBJ(jvm->native.CallStaticObjectMethodA(env, AS_CLASS(r1), mid, nullptr));
        break;
    }
    case 117: case 118: case 119: {
        /* CallStaticBooleanMethod/V/A — playCoreApiMissing and other booleans */
        auto mid = AS_MID(r2);
        uintptr_t midx = (uintptr_t)mid;
        const char *mname = nullptr;
        if (midx > 0 && midx <= 65536) {
            auto &mo = jvm->objects[midx - 1];
            if (mo.type == jvm_object::JVM_OBJECT_METHOD && mo.method.name.data)
                mname = mo.method.name.data;
        }
        if (try_asset_jni(ctx, svc_no, regs)) break;
        uint32_t bret = 0;
        if (mname && !strcmp(mname, "playCoreApiMissing")) bret = 1;
        ret32(bret);
        break;
    }
    case 120: case 121: case 122: ret32(0); break;
    case 123: case 124: case 125: ret32(0); break;
    case 126: case 127: case 128: ret32(0); break;
    case 129: case 130: case 131: ret32(0); break; /* CallStaticIntMethod/V/A */
    case 132: case 133: case 134: ret64(0); break; /* CallStaticLongMethod/V/A */
    case 135: case 136: case 137: ret32(0); break;
    case 138: case 139: case 140: ret64(0); break;
    case 141: case 142: case 143: { /* CallStaticVoidMethod/V/A */
        /* Guests call through cached (class, methodID) pairs without checking
         * for NULL; forwarding NULL to the host JVM trips its assert. */
        const char *mname = nullptr;
        uintptr_t midx = (uintptr_t)AS_MID(r2);
        if (midx > 0 && midx <= 65536) {
            auto &mo = jvm->objects[midx - 1];
            if (mo.type == jvm_object::JVM_OBJECT_METHOD && mo.method.name.data)
                mname = mo.method.name.data;
        }
        if (!r1 || !r2) {
            fprintf(stderr, "[arm_jni] CallStaticVoidMethod: NULL class/mid "
                    "(cls=0x%08x mid=0x%08x name=%s lr=0x%08x) — ignored\n",
                    r1, r2, mname ? mname : "?", regs[14]);
            break;
        }
        jvm->native.CallStaticVoidMethodA(env, AS_CLASS(r1), AS_MID(r2), nullptr);
        break;
    }

    /* 144: GetStaticFieldID */
    case 144: RET_OBJ(jvm->native.GetStaticFieldID(env, AS_CLASS(r1),
                      ARM_STR(r2), ARM_STR(r3))); break;

    /* 145-162: GetStatic/SetStaticXxxField */
    case 145: RET_OBJ(jvm->native.GetStaticObjectField(env,AS_CLASS(r1),AS_FID(r2))); break;
    case 146: ret32((uint32_t)jvm->native.GetStaticBooleanField(env,AS_CLASS(r1),AS_FID(r2))); break;
    case 147: ret32(0); break;
    case 148: ret32(0); break;
    case 149: ret32(0); break;
    case 150: ret32((uint32_t)jvm->native.GetStaticIntField(env,AS_CLASS(r1),AS_FID(r2))); break;
    case 151: ret64(0); break;
    case 152: ret32(0); break;
    case 153: ret64(0); break;
    case 154 ... 162: break; /* SetStatic* stubs */

    /* 163: NewString(env, jchar* unicodeChars, jsize len) — UTF-16 → UTF-8 */
    case 163: {
        uint32_t len = r2;
        std::string utf8;
        if (r1 && len <= 8192u) {
            for (uint32_t i = 0; i < len; ++i) {
                auto *p = ctx.mem.ptr(r1 + i * 2u);
                if (!p) break;
                uint16_t c; memcpy(&c, p, 2);
                /* minimal UTF-16 (BMP) → UTF-8 */
                if (c < 0x80) utf8.push_back((char)c);
                else if (c < 0x800) {
                    utf8.push_back((char)(0xC0 | (c >> 6)));
                    utf8.push_back((char)(0x80 | (c & 0x3F)));
                } else {
                    utf8.push_back((char)(0xE0 | (c >> 12)));
                    utf8.push_back((char)(0x80 | ((c >> 6) & 0x3F)));
                    utf8.push_back((char)(0x80 | (c & 0x3F)));
                }
            }
        }
        static int nstr_count = 0;
        if (nstr_count < 80)
            fprintf(stderr, "[arm_jni] NewString(len=%u): %s\n", len, utf8.c_str());
        ++nstr_count;
        RET_OBJ(jvm->native.NewStringUTF(env, utf8.c_str()));
        break;
    }
    /* 164: GetStringLength */
    case 164: ret32((uint32_t)jvm->native.GetStringLength(env, AS_STR(r1))); break;
    /* 165: GetStringChars */
    case 165: ret32(0); break;
    /* 166: ReleaseStringChars */
    case 166: break;

    /* 167: NewStringUTF(env, utf_va) */
    case 167: {
        const char *ns = ARM_STR(r1);
        static int ns_count = 0;
        jstring ret_str = jvm->native.NewStringUTF(env, ns);
        if (ns_count < 40)
            fprintf(stderr, "[arm_jni] NewStringUTF: %s -> handle=0x%x\n",
                    ns ? ns : "(null)", (uint32_t)(uintptr_t)ret_str);
        ++ns_count;
        RET_OBJ(ret_str);
        break;
    }

    /* 168: GetStringUTFLength */
    case 168: ret32((uint32_t)jvm->native.GetStringUTFLength(env, AS_STR(r1))); break;

    /* 169: GetStringUTFChars(env, str, isCopy_va) */
    case 169: {
        jboolean isCopy = JNI_FALSE;
        const char *s = jvm->native.GetStringUTFChars(env, AS_STR(r1), &isCopy);
        if (getenv("LUNARIA_TRACE_JNI"))
            fprintf(stderr, "[arm_jni] GetStringUTFChars: handle=0x%x -> \"%s\"\n",
                    r1, s ? s : "(null)");
        if (s) {
            /* Allocate a unique ARM heap buffer per call so simultaneous
 * pointers held by Unity do not alias each other. */
            size_t len = std::min(strlen(s), (size_t)4095u);
            uint32_t addr = arm_malloc(ctx, (uint32_t)(len + 1));
            if (addr) {
                if (auto *p = ctx.mem.ptr(addr)) {
                    memcpy(p, s, len); p[len] = '\0';
                }
                ret32(addr);
            } else {
                /* Heap exhausted: fall back to shared scratch */
                if (auto *p = ctx.mem.ptr(STR_SCRATCH)) {
                    memcpy(p, s, len); p[len] = '\0';
                }
                ret32(STR_SCRATCH);
            }
        } else {
            ret32(0);
        }
        break;
    }
    /* 170: ReleaseStringUTFChars — no-op */
    case 170: break;

    /* 171: GetArrayLength */
    case 171: ret32((uint32_t)jvm->native.GetArrayLength(env, AS_ARR(r1))); break;

    /* 172-174: ObjectArray ops */
    case 172: {
        /* NewObjectArray(env, length, elementClass, initialElement) */
        uint32_t count = r1;
        if (count > 512u) {
            /* Unusually large array: return null so Unity's null-check skips it. */
            fprintf(stderr, "[arm_jni] NewObjectArray: count=%u > 512, returning null\n",
                    count);
            ret32(0u);
            break;
        }
        RET_OBJ(jvm->native.NewObjectArray(env, (jsize)count, AS_CLASS(r2), AS_OBJ(r3)));
        break;
    }
    case 173: RET_OBJ(jvm->native.GetObjectArrayElement(
                env,(jobjectArray)(uintptr_t)r1,(jsize)r2)); break;
    case 174: /* SetObjectArrayElement(env, array, index, value) */
        if (r1) jvm->native.SetObjectArrayElement(env, (jobjectArray)(uintptr_t)r1,
                                                   (jsize)r2, AS_OBJ(r3));
        break;

    /* 175-214: Array primitives — stubs */
    case 175 ... 214: ret32(0); break;

    /* 215: RegisterNatives(env, class, methods_va, count) */
    case 215:
        do_register_natives(ctx, r1, r2, (int32_t)r3);
        ret32(0);
        break;

    /* 216: UnregisterNatives */
    case 216: ret32(0); break;
    /* 217: MonitorEnter */
    case 217: ret32(0); break;
    /* 218: MonitorExit */
    case 218: ret32(0); break;

    /* 219: GetJavaVM(env, vm_pp_va) */
    case 219:
        ctx.mem.write32(r1, VM_SLOT_BASE);
        ret32(0);
        break;

    /* 220-221: String region ops */
    case 220: case 221: break;

    /* 222-223: WeakGlobalRef */
    case 222: RET_OBJ(jvm->native.NewGlobalRef(env, AS_OBJ(r1))); break;
    case 223: jvm->native.DeleteGlobalRef(env, AS_OBJ(r1)); break;

    /* 224: ExceptionCheck */
    case 224: ret32((uint32_t)jvm->native.ExceptionCheck(env)); break;

    /* 225-228: NIO / ref type stubs */
    case 225 ... 228: ret32(0); break;

    /* ---- JVM Invoke interface ---- */

    /* 229: JVM::GetEnv(vm, &env, version)
 * r0 = JavaVM* (VM_SLOT_BASE)
 * r1 = JNIEnv** (ARM VA to receive env pointer)
 * r2 = version */
    case SVC_JVM_GETENV:
        ctx.mem.write32(r1, ENV_SLOT_BASE);
        ret32(0); /* JNI_OK */
        break;

    /* 230: JVM::AttachCurrentThread */
    case SVC_JVM_ATTACH:
        ctx.mem.write32(r1, ENV_SLOT_BASE);
        ret32(0);
        break;

    /* 231: JVM::DestroyJavaVM */
    case SVC_JVM_DESTROY: ret32(0); break;

    /* ---- Android/POSIX stubs ---- */

    case SVC_LOG_PRINT: {
        const char *tag = ARM_STR(r1);
        ArmVarArgs ap{ctx, regs, 3, regs[13], false};
        std::string s = arm_vformat(ctx, ARM_STR(r2), ap);
        fprintf(stderr, "[android_log p=%u t=%s lr=0x%08x] %s\n",
                r0, tag?tag:"?", regs[14], s.c_str());
        wapi_trace_caller(ctx, regs, s.c_str());
        ret32(0);
        break;
    }
    case SVC_LOG_WRITE: {
        const char *tag = ARM_STR(r1);
        const char *msg = ARM_STR(r2);
        fprintf(stderr, "[android_log p=%u t=%s] %s\n",
                r0, tag?tag:"?", msg?msg:"");
        wapi_trace_caller(ctx, regs, msg ? msg : "");
        ret32(0);
        break;
    }
    case SVC_DLOPEN: {
        const char *dpath = ARM_STR(r0);
        fprintf(stderr, "[arm_exec] dlopen: %s (lr=0x%08x)\n", dpath ? dpath : "(null)", regs[14]);
        /* libandroid.so: Swappy's NDK Choreographer path dlopen()s this for
 * AChoreographer_* via dlsym.  There is no host file; return a sentinel
 * handle so dlsym can resolve our SVC trampolines instead of NULL. */
        if (dpath && strstr(dpath, "libandroid.so")) {
            ret32(ARM_LIBANDROID);
            break;
        }
        /* libaaudio.so: FMOD's audio output/recorder path resolves AAudio_*
         * via dlsym; return a sentinel handle backed by SVC trampolines. */
        if (dpath && strstr(dpath, "libaaudio.so")) {
            ret32(ARM_LIBAAUDIO);
            break;
        }
        /* libc.so: always loaded on a real device.  FMOD dlsym()s getauxval
         * from it for AT_HWCAP (NEON) detection. */
        if (dpath && strstr(dpath, "libc.so")) {
            ret32(ARM_LIBC);
            break;
        }
        /* libGLESv2/libEGL: Unity dlsym()s GLES3.x entry points from these;
         * dlsym resolves by name against the SVC map, so any sentinel works.
         * NULL made Unity silently disable the corresponding GL features. */
        if (dpath && (strstr(dpath, "libGLESv") || strstr(dpath, "libEGL.so"))) {
            ret32(ARM_LIBANDROID);
            break;
        }
        /* Return NULL for files that don't actually exist on the host so that
 * mono falls back to JIT mode instead of trying to use a missing
 * AOT-precompiled .dll.so that isn't included in this APK. */
        char lib_real[4096]; lib_real[0] = 0;
        if (dpath && access(dpath, F_OK) != 0) {
            /* Bare soname (no '/') e.g. "monobdwgc-2.0": Android linker convention
 * prepends "lib" and appends ".so" — but Unity also passes names that
 * already carry the prefix/suffix (dlopen("libswappywrapper") used to
 * probe "liblibswappywrapper.so" and fail even though the APK ships
 * libswappywrapper.so).  Try all plausible spellings. */
            bool resolved = false;
            if (!strchr(dpath, '/') && !g_main_lib_dir.empty()) {
                const char *dir = g_main_lib_dir.c_str();
                size_t dl = strlen(dpath);
                bool has_so = dl > 3 && !strcmp(dpath + dl - 3, ".so");
                const char *fmts[] = {
                    has_so ? "%s/%s"    : "%s/lib%s.so", /* canonical */
                    has_so ? "%s/lib%s" : "%s/%s.so",    /* alt prefix/suffix */
                    "%s/%s",                             /* exact name */
                };
                for (const char *f : fmts) {
                    char cand[4096];
                    snprintf(cand, sizeof(cand), f, dir, dpath);
                    if (access(cand, F_OK) == 0) {
                        fprintf(stderr, "[arm_exec] dlopen: resolved bare name → %s\n", cand);
                        snprintf(lib_real, sizeof(lib_real), "%s", cand);
                        resolved = true;
                        break;
                    }
                }
            }
            if (!resolved) {
                fprintf(stderr, "[arm_exec] dlopen: not found → NULL\n");
                ret32(0u);
                break;
            }
        } else if (dpath) {
            snprintf(lib_real, sizeof(lib_real), "%s", dpath);
        }
        /* Load a resolved .so that isn't in guest memory yet so its exports
         * resolve to real code via g_exported_syms.  Static ctors run on the
         * standalone callback JIT — the main JIT is mid-Run() in this SVC. */
        if (lib_real[0] && g_loaded_lib_paths.find(lib_real) == g_loaded_lib_paths.end()) {
            uint32_t onload_va = 0;
            uint32_t base = (g_lib_load_end + 0xffffu) & ~0xffffu;
            if (load_elf(ctx, lib_real, onload_va, base, /*ctors_via_cb=*/true)) {
                fprintf(stderr, "[arm_exec] dlopen: loaded %s at 0x%08x%s\n",
                        lib_real, base,
                        onload_va ? " (JNI_OnLoad not run)" : "");
            } else {
                fprintf(stderr, "[arm_exec] dlopen: load_elf failed for %s\n", lib_real);
                ret32(0u);
                break;
            }
        }
        ret32(1u);
        break;
    }
    case SVC_DLSYM: {
        const char *dsym = ARM_STR(r1);
        if (dsym) {
            /* Unity's per-frame re-init re-resolves the entire il2cpp_* export
 * set (~230 symbols) every frame, so this fires ~100k times/run.
 * The result is identical each time (the library loads only once),
 * so log each distinct symbol only the first time it is resolved;
 * the full per-call trace stays available under LUNARIA_TRACE_DLSYM. */
            static std::map<std::string, char> seen_dlsym;
            bool trace = seen_dlsym.emplace(dsym, 0).second || getenv("LUNARIA_TRACE_DLSYM");
            /* Prefer guest-side exports (e.g. mono symbols) over host stubs */
            auto exp_it = g_exported_syms.find(dsym);
            if (exp_it != g_exported_syms.end()) {
                if (trace)
                    fprintf(stderr, "[arm_exec] dlsym: %s -> guest 0x%08x\n",
                            dsym, exp_it->second);
                ret32(exp_it->second);
                break;
            }
            if (trace)
                fprintf(stderr, "[arm_exec] dlsym: %s (lr=0x%08x)\n", dsym, regs[14]);
            /* API 29+ vsync helpers need real FrameCallbackData; leave NULL so
 * Swappy falls back to postFrameCallbackDelayed (which we emulate). */
            if (!strcmp(dsym, "AChoreographer_postVsyncCallback") ||
                strstr(dsym, "AChoreographerFrameCallbackData_")) {
                ret32(0u);
                break;
            }
            if (uint32_t dva = lookup_symbol_direct_va(dsym)) { ret32(dva); break; }
            uint32_t found_svc = lookup_symbol_svc(dsym);
            if (found_svc != UINT32_MAX) {
                ret32(TRAMP_BASE + found_svc * TRAMP_STRIDE);
                break;
            }
            /* Unknown symbol → per-symbol stub trampoline (returns 0 and
             * logs the name when called).  Never hand out trampoline #0:
             * it is svc #0, which the SVC dispatcher interprets as a raw
             * Linux syscall through a stale r7. */
            auto slot_it = g_unknown_sym_slot.find(dsym);
            uint32_t slot;
            if (slot_it != g_unknown_sym_slot.end()) {
                slot = slot_it->second;
            } else {
                slot = (uint32_t)g_unknown_sym_names.size();
                uint32_t idx = SVC_TRAMP_TOTAL + slot;
                if (TRAMP_BASE + (idx + 1u) * TRAMP_STRIDE > JNI_TBL_BASE) {
                    ret32(TRAMP_BASE + SVC_RET0 * TRAMP_STRIDE); /* pool full */
                    break;
                }
                g_unknown_sym_names.push_back(dsym);
                g_unknown_sym_slot[dsym] = slot;
                ctx.mem.write32(TRAMP_BASE + idx * TRAMP_STRIDE,
                                0xEF000000u | SVC_UNKNOWN_SYM);
                ctx.mem.write32(TRAMP_BASE + idx * TRAMP_STRIDE + 4, 0xE12FFF1Eu);
            }
            ret32(TRAMP_BASE + (SVC_TRAMP_TOTAL + slot) * TRAMP_STRIDE);
        } else {
            ret32(TRAMP_BASE + SVC_RET0 * TRAMP_STRIDE); /* null sym → ret-0 stub */
        }
        break;
    }
    case SVC_DLCLOSE: ret32(0); break;

    /* ---- libc: memory / string ---- */

    case SVC_MALLOC: {
        uint32_t addr = arm_malloc(ctx, r0);
        ret32(addr);
        break;
    }
    case SVC_FREE: arm_free(ctx, r0); break;
    case SVC_CALLOC: {
        uint64_t sz64 = (uint64_t)r0 * (uint64_t)r1;
        if (sz64 > 0xFFFFFFFFull) { ret32(0); break; } /* overflow → fail */
        uint32_t sz = (uint32_t)sz64;
        uint32_t addr = arm_malloc(ctx, sz);
        /* free-list blocks may hold stale data, so zero explicitly */
        if (addr && sz) memset(ctx.mem.ptr(addr), 0, sz);
        ret32(addr);
        break;
    }
    case SVC_REALLOC:
        ret32(arm_realloc(ctx, r0, r1));
        break;
    case SVC_MEMCPY:
    case SVC_MEMMOVE: {
        /* dump top callers every 2^20 hits (find spin/copy hotspots) */
        static const bool trace_mm = getenv("LUNARIA_TRACE_MEMMOVE") != nullptr;
        if (trace_mm) {
            static std::map<uint32_t, std::pair<uint64_t, uint64_t>> by_lr;
            static uint64_t calls = 0;
            auto &e = by_lr[regs[14]];
            ++e.first; e.second += r2;
            if ((++calls & ((1u << 20) - 1)) == 0) {
                std::vector<std::pair<uint64_t, uint32_t>> top;
                for (auto &kv : by_lr) top.push_back(std::make_pair(kv.second.first, kv.first));
                std::sort(top.begin(), top.end(), std::greater<>());
                fprintf(stderr, "[memmove] %llu calls from %zu sites, top:\n",
                        (unsigned long long)calls, by_lr.size());
                for (size_t i = 0; i < top.size() && i < 8; ++i)
                    fprintf(stderr, "[memmove]   lr=0x%08x calls=%llu bytes=%llu\n",
                            top[i].second, (unsigned long long)top[i].first,
                            (unsigned long long)by_lr[top[i].second].second);
            }
        }
        if (r2) {
            wrange_log(r0, r1, r2, regs[15], regs[14],
                       svc_no == SVC_MEMCPY ? "memcpy" : "memmove");
            memmove(ctx.mem.ptr(r0), ctx.mem.ptr(r1), r2);
        }
        ret32(r0);
        break;
    }
    case SVC_MEMSET: {
        if (r2) {
            wrange_log(r0, r1 & 0xFF, r2, regs[15], regs[14], "memset");
            memset(ctx.mem.ptr(r0), (int)(r1 & 0xFF), r2);
        }
        ret32(r0);
        break;
    }
    case SVC_STRLEN: {
        const char *s = ctx.mem.cstr(r0);
        ret32(s ? (uint32_t)strlen(s) : 0);
        break;
    }
    case SVC_STRCPY: {
        if (r0 && r1) strcpy((char*)ctx.mem.ptr(r0), (const char*)ctx.mem.ptr(r1));
        ret32(r0);
        break;
    }
    case SVC_STRNCPY: {
        if (r0 && r1) strncpy((char*)ctx.mem.ptr(r0), (const char*)ctx.mem.ptr(r1), r2);
        ret32(r0);
        break;
    }
    case SVC_STRCMP: {
        const char *s1 = ctx.mem.cstr(r0);
        const char *s2 = ctx.mem.cstr(r1);
        if (!s1 && !s2) { ret32(0); break; }
        if (!s1) { ret32((uint32_t)-1); break; }
        if (!s2) { ret32(1); break; }
        ret32((uint32_t)strcmp(s1, s2));
        break;
    }
    case SVC_STRNCMP: {
        const char *s1 = ctx.mem.cstr(r0);
        const char *s2 = ctx.mem.cstr(r1);
        if (!s1 && !s2) { ret32(0); break; }
        if (!s1) { ret32((uint32_t)-1); break; }
        if (!s2) { ret32(1); break; }
        ret32((uint32_t)strncmp(s1, s2, (size_t)r2));
        break;
    }
    case SVC_STRDUP: {
        const char *s = ctx.mem.cstr(r0);
        if (!s) { ret32(0); break; }
        size_t len = strlen(s) + 1;
        uint32_t addr = arm_malloc(ctx, (uint32_t)len);
        if (addr) memcpy(ctx.mem.ptr(addr), s, len);
        ret32(addr);
        break;
    }
    case SVC_STRNDUP: {
        const char *s = ctx.mem.cstr(r0);
        if (!s) { ret32(0); break; }
        size_t len = strnlen(s, r1);
        uint32_t addr = arm_malloc(ctx, (uint32_t)(len + 1));
        if (addr) {
            memcpy(ctx.mem.ptr(addr), s, len);
            ctx.mem.ptr(addr)[len] = 0;
        }
        ret32(addr);
        break;
    }
    case SVC_STRCAT: {
        if (r0 && r1) strcat((char*)ctx.mem.ptr(r0), (const char*)ctx.mem.ptr(r1));
        ret32(r0);
        break;
    }
    case SVC_STRNCAT: {
        if (r0 && r1) strncat((char*)ctx.mem.ptr(r0), (const char*)ctx.mem.ptr(r1), r2);
        ret32(r0);
        break;
    }
    case SVC_ABORT:
        /* A real abort() never returns.  The old stub returned 0, so a failed
 * g_assert (mono's `do { abort(); } while (1)` style) re-entered the
 * same assertion forever, flooding the log with millions of lines at
 * 100% CPU.  Instead, unwind the current run (like a longjmp out of the
 * aborting call): set the yield flag so CallSVC halts this JIT.  The
 * loader's render loop then re-enters cleanly, bounding the spew to one
 * abort per frame instead of an unbounded busy-loop. */
        {
            static uint64_t aborts = 0;
            bool first = (aborts == 0);
            if (aborts < 4 || (aborts & 0x3ff) == 0)
                fprintf(stderr, "[arm_exec] abort() called (#%llu) lr=0x%08x tid=%u — halting run\n",
                        (unsigned long long)aborts, regs[14], g_current_tid);
            ++aborts;
            /* On first abort, dump the guest stack to reveal the failing mono function. */
            if (first) {
                uint32_t sp = regs[13];
                fprintf(stderr, "[arm_exec] abort stack dump (sp=0x%08x r0=0x%08x r1=0x%08x):\n",
                        sp, r0, r1);
                for (uint32_t off = 0; off < 0x200; off += 4) {
                    uint32_t w = ctx.mem.read32(sp + off);
                    if (w >= 0x20000000u && w < 0x30000000u)
                        fprintf(stderr, "[arm_exec]   stack+0x%03x -> 0x%08x\n", off, w);
                }
            }
        }
        ret32(0);
        g_yield_requested = true;
        break;
    case SVC_PTHREAD_KEY:
        /* mutex/cond/sem family: succeed immediately, but periodically give
 * worker threads time — the caller may be spin-waiting on a predicate
 * that only a worker thread can change. */
        if (getenv("LUNARIA_TRACE_PTHREAD"))
            fprintf(stderr, "[arm_exec] SVC_PTHREAD_KEY svc=%u lr=0x%08x r0=0x%08x r1=0x%08x\n",
                    svc_no, regs[14], r0, r1);
        maybe_schedule_on_wait();
        ret32(0);
        break;
    case SVC_WAIT:
        /* always schedule before returning success */
        if (!g_threads.empty())
            schedule_threads(2'000'000ULL);
        ret32(0);
        break;

    
    case SVC_SEM_INIT: {
        g_sems[r0] = (int32_t)r2;
        static int si_log = 0;
        if (si_log++ < 40)
            fprintf(stderr, "[sem] init sem=0x%08x val=%d lr=0x%08x tid=%u\n",
                    r0, (int32_t)r2, regs[14], g_current_tid);
        ret32(0); break;
    }
    case SVC_SEM_POST:
        ++g_sems[r0];
        if (getenv("LUNARIA_TRACE_SEM")) {
            static int pn = 0;
            if (pn++ < 200)
                fprintf(stderr, "[sem] post sem=%#x val=%d tid=%u lr=0x%08x\n",
                        r0, g_sems[r0], g_current_tid, regs[14]);
        }
        ret32(0); break;
    case SVC_SEM_DESTROY:
        g_sems.erase(r0);
        ret32(0); break;
    case SVC_SEM_GETVALUE:
        if (r1) ctx.mem.write32(r1, (uint32_t)(g_sems.count(r0) ? g_sems[r0] : 0));
        ret32(0); break;
    case SVC_SEM_TRYWAIT: {
        int32_t &c = g_sems[r0];
        if (c > 0) { --c; ret32(0); break; }
        uint32_t eva = errno_va(ctx, g_current_tid);
        if (eva) ctx.mem.write32(eva, 11 /* EAGAIN */);
        ret32(~0u);
        break;
    }
    case SVC_SEM_WAIT: case SVC_SEM_TIMEDWAIT: {
        /* call_guest_cb context: sem_wait must not block (we're inside a signal
 * handler call). Return 0 immediately so GC_suspend_handler can complete
 * its sem_post(GC_ack_sem) + sem_wait(GC_resume_sem) sequence without
 * hanging the emulator. GC_resume_sem will be posted by GC_start_world
 * before the suspended thread's next quantum. */
        if (g_in_cb) { ret32(0); break; }
        
        if (g_current_tid == 0 && !g_scheduling && !g_threads.empty()) {
            int spin_limit = (int)env_ticks("LUNARIA_FUTEX_SPINS", 500);
            for (int spin = 0; g_sems[r0] <= 0 && spin < spin_limit; ++spin) {
                schedule_threads(20'000'000ULL);
                drive_aaudio_callbacks(ctx);
                drive_java_choreographer(ctx);
            }
        }
        int32_t &c = g_sems[r0];
        if (c > 0) { --c; ret32(0); break; }
        if (g_current_tid != 0) {
            
            for (auto &t : g_threads) {
                if (t.id != g_current_tid) continue;
                t.waiting_sem     = r0;
                t.sem_timed       = (svc_no == SVC_SEM_TIMEDWAIT);
                t.sem_skip_passes = 0;
                break;
            }
            g_yield_requested = true;
            ret32(0); 
            break;
        }
        if (getenv("LUNARIA_TRACE_SEM")) {
            static int wn = 0;
            if (wn++ < 200)
                fprintf(stderr, "[sem] wait-fail sem=%#x tid=%u lr=0x%08x threads=%zu\n",
                        r0, g_current_tid, regs[14], g_threads.size());
        }
        /* let caller retry */
        uint32_t eva = errno_va(ctx, g_current_tid);
        if (eva)
            ctx.mem.write32(eva, svc_no == SVC_SEM_WAIT ? 4u /* EINTR */
                                                        : 110u /* ETIMEDOUT */);
        ret32(~0u);
        break;
    }

    /* pthread_create: capture a full guest-thread context with its own stack;
 * arm_exec_run_pending_threads() schedules it cooperatively. */
    case SVC_PTHREAD_CREATE: {
        /* pthread_create(thread_t*, attr*, fn_va, arg) */
        if (r2 && g_thread_stack_next + THREAD_STACK_SIZE <= HEAP_BASE) {
            ArmThread t;
            
            t.id = (uint32_t)g_threads.size() + 2;
            uint32_t stack_top = g_thread_stack_next + THREAD_STACK_SIZE - 16;
            t.stack_base = g_thread_stack_next;
            g_thread_stack_next += THREAD_STACK_SIZE;
            t.regs[0]  = r3;            /* arg */
            t.regs[13] = stack_top;
            t.regs[14] = SENTINEL_ADDR;
            t.regs[15] = r2 & ~1u;
            t.entry_pc = r2 & ~1u;
            t.cpsr = (r2 & 1u) ? 0x30u : 0x10u;
            g_threads.push_back(t);
            if (r0) ctx.mem.write32(r0, t.id);
            fprintf(stderr, "[arm_exec] pthread_create tid=%u fn=0x%08x arg=0x%08x sp=0x%08x out=0x%08x\n",
                    t.id, r2, r3, stack_top, r0);
            if (getenv("LUNARIA_TRACE_JOBS") && r3 >= 0x1000u) {
                /* dump the thread-arg object: Baselib job-worker layout has
                 * jobsem@+0x48 (count@+0x88), donesem@+0xc8 (count@+0x108),
                 * fn/arg@+0x148, flag@+0x150 */
                fprintf(stderr, "  [arg dump] +0x48=0x%08x +0x88=0x%08x +0xc8=0x%08x "
                        "+0x108=0x%08x +0x148=0x%08x +0x14c=0x%08x +0x150=0x%08x\n",
                        ctx.mem.read32(r3+0x48), ctx.mem.read32(r3+0x88),
                        ctx.mem.read32(r3+0xc8), ctx.mem.read32(r3+0x108),
                        ctx.mem.read32(r3+0x148), ctx.mem.read32(r3+0x14c),
                        ctx.mem.read32(r3+0x150));
            }
            
            bool immediate = false;
            if (g_mono_base) {
                uint32_t fn = r2 & ~1u;
                uint32_t lr = regs[14];
                if ((fn >= g_mono_base && fn < g_mono_base + 0x800000u) ||
                    (lr >= g_mono_base && lr < g_mono_base + 0x800000u))
                    immediate = true;
            }
            if (immediate)
                schedule_threads(env_ticks("LUNARIA_THREAD_TICKS", 200'000'000ULL));
        } else if (r0) {
            ctx.mem.write32(r0, 0);
        }
        ret32(0);
        break;
    }

    /* ---- ANativeWindow ---- */

    case SVC_ANW_FROM_SURFACE: {
        /* ANativeWindow_fromSurface(env, surface) → ANativeWindow*
 * Unity blocks graphics-device init until it has a non-null window, so
 * return the fake window whenever we have ANY working host GL backend —
 * either a GLFW/X11 window or a headless surfaceless EGL context.
 * (Previously this required g_glfw, which is null in surfaceless mode,
 * leaving Unity windowless → it re-ran engine init every frame.) */
        ensure_glfw_window();
        bool have_gl = g_glfw || g_egl_ctx != EGL_NO_CONTEXT || init_host_egl();
        uint32_t anw = have_gl ? ensure_fake_anative_window(ctx) : 0u;
        static int anw_count = 0;
        if (getenv("LUNARIA_TRACE_EGL") && anw_count < 10)
            fprintf(stderr, "[egl] ANativeWindow_fromSurface → %#x (have_gl=%d)\n",
                    anw, have_gl);
        ++anw_count;
        ret32(anw);
        break;
    }
    case SVC_ANW_ACQUIRE: case SVC_ANW_RELEASE: case SVC_ANW_TOSURFACE:
        ret32(0u); break;
    case SVC_ANW_GETWIDTH: {
        int w = 1280;
        if (g_glfw) glfwGetWindowSize(g_glfw, &w, nullptr);
        ret32((uint32_t)w); break;
    }
    case SVC_ANW_GETHEIGHT: {
        int h = 720;
        if (g_glfw) glfwGetWindowSize(g_glfw, nullptr, &h);
        ret32((uint32_t)h); break;
    }
    case SVC_ANW_SETBUFGEO: ret32(0u); break;

    /* ---- ALooper ---- */

    case SVC_ALOOPER_FORTHREAD: case SVC_ALOOPER_PREPARE:
        /* Return a non-NULL sentinel so Unity doesn't skip frame processing */
        ret32(ARM_ALOOPER); break;
    case SVC_ALOOPER_POLLONCE: case SVC_ALOOPER_POLLALL: {
        /* ALooper_pollOnce(timeoutMillis, outFd, outEvents, outData)
         * r0 = timeoutMillis: 0=poll immediately, -1=block forever, >0=timed wait
         * Returns: >= 0 = fd with event, POLL_CALLBACK(-1), POLL_TIMEOUT(-3), POLL_WAKE(-2)
         *
         * With timeout==0 return POLL_TIMEOUT immediately (poll mode — caller continues).
         * With timeout!=0 the caller expects to block; if we return immediately the
         * caller loops back and calls us again, creating an infinite spin that
         * prevents worker threads from ever running.  Yield to workers so they can
         * make progress (e.g. deliver a wake via SVC_ALOOPER_WAKE). */
        int32_t timeout_ms = (int32_t)r0;
        static uint32_t alooper_poll_count = 0;
        if (alooper_poll_count++ < 4)
            fprintf(stderr, "[alooper] pollOnce/All timeout=%d tid=%u #%u\n",
                    timeout_ms, g_current_tid, alooper_poll_count - 1);
        if (timeout_ms == 0) {
            /* Immediate poll: return POLL_TIMEOUT, caller continues normally */
            ret32((uint32_t)-3);
        } else if (g_current_tid == 0 && !g_scheduling && !g_threads.empty()) {
            /* Main thread blocking wait: spin schedule_threads so workers can run,
             * check for wake signal.  Same pattern as futex_wait on main thread. */
            int spin_limit = (int)env_ticks("LUNARIA_FUTEX_SPINS", 500);
            bool woken = false;
            for (int s = 0; s < spin_limit && !woken; ++s) {
                schedule_threads(20'000'000ULL);
                drive_aaudio_callbacks(ctx);
                drive_java_choreographer(ctx);
                if (g_alooper_wake_pending.exchange(false)) woken = true;
            }
            ret32(woken ? (uint32_t)-2u : (uint32_t)-3u); /* POLL_WAKE or POLL_TIMEOUT */
        } else {
            /* Worker thread or no workers: yield this thread's slice */
            if (g_current_tid != 0) g_yield_requested = true;
            ret32((uint32_t)-3);
        }
        break;
    }
    case SVC_ALOOPER_WAKE:
        g_alooper_wake_pending.store(true);
        ret32(0u); break;
    case SVC_ALOOPER_ADDFD:
        /* ALooper_addFd/removeFd return 1 on success, -1 on error.  We don't
         * drive fd callbacks from a real looper; callers that poll the fd
         * themselves (NdkLooperHandler reads its pipe) still work. */
        ret32(1u); break;

    /* ---- AAudio (see SVC constant block for scope) ---- */

    case SVC_AAUDIO_CREATE_BUILDER:
        /* AAudio_createStreamBuilder(AAudioStreamBuilder **builder) */
        fprintf(stderr, "[arm_exec] AAudio_createStreamBuilder tid=%u\n", g_current_tid);
        if (!r0) { ret32((uint32_t)-1); break; }
        g_aaudio_builder = AAudioStreamState{};   /* reset accumulated config */
        ctx.mem.write32(r0, ARM_AAUDIO_BUILDER);
        ret32(0u); break;
    case SVC_AAUDIO_SET_DIRECTION:
        g_aaudio_builder.input = (r1 == 1u);      /* AAUDIO_DIRECTION_INPUT */
        ret32(0u); break;
    case SVC_AAUDIO_SET_DATA_CB:
        g_aaudio_builder.data_cb   = r1;
        g_aaudio_builder.user_data = r2;
        ret32(0u); break;
    case SVC_AAUDIO_SET_FORMAT:
        if ((int32_t)r1 > 0) g_aaudio_builder.format = r1;
        ret32(0u); break;
    case SVC_AAUDIO_SET_CHANNELS:
        if ((int32_t)r1 > 0 && r1 <= 8u) g_aaudio_builder.channels = r1;
        ret32(0u); break;
    case SVC_AAUDIO_SET_RATE:
        if ((int32_t)r1 > 0) g_aaudio_builder.sample_rate = r1;
        ret32(0u); break;
    case SVC_AAUDIO_OPEN_STREAM: {
        /* AAudioStreamBuilder_openStream(builder, AAudioStream **stream) */
        if (!r1) { ret32((uint32_t)-1); break; }
        uint32_t handle = ARM_AAUDIO_STREAM + 16u * g_aaudio_stream_count++;
        g_aaudio_streams[handle] = g_aaudio_builder;
        fprintf(stderr, "[arm_exec] AAudio openStream → 0x%08x dir=%s cb=0x%08x "
                "ch=%u fmt=%u rate=%u tid=%u\n",
                handle, g_aaudio_builder.input ? "in" : "out",
                g_aaudio_builder.data_cb, g_aaudio_builder.channels,
                g_aaudio_builder.format, g_aaudio_builder.sample_rate, g_current_tid);
        ctx.mem.write32(r1, handle);
        ret32(0u); break;
    }
    case SVC_AAUDIO_START: {
        auto it = g_aaudio_streams.find(r0);
        if (it != g_aaudio_streams.end()) it->second.started = true;
        fprintf(stderr, "[arm_exec] AAudioStream_requestStart 0x%08x %s tid=%u\n",
                r0, it != g_aaudio_streams.end() ? "ok" : "unknown", g_current_tid);
        ret32(0u); break;
    }
    case SVC_AAUDIO_STOP: {
        auto it = g_aaudio_streams.find(r0);
        if (it != g_aaudio_streams.end()) it->second.started = false;
        ret32(0u); break;
    }
    case SVC_AAUDIO_GET_FPB:
        /* 480 frames/burst = 10 ms @ 48 kHz, a typical device value */
        ret32(480u); break;
    case SVC_AAUDIO_GET_BUFSIZE:
        ret32(1920u); break;
    case SVC_AAUDIO_SET_BUFSIZE:
        /* returns the actual size set (or negative error) */
        ret32(r1); break;
    case SVC_AAUDIO_GET_BUFCAP:
        ret32(3840u); break;
    case SVC_AAUDIO_WAIT_STATE: {
        /* AAudioStream_waitForStateChange(stream, inputState, *nextState, timeoutNs).
         * Transitional states resolve to their settled state immediately. */
        uint32_t in = r1, next;
        switch (in) {
        case 3u:  next = 4u;  break;   /* STARTING → STARTED */
        case 5u:  next = 6u;  break;   /* PAUSING → PAUSED */
        case 7u:  next = 8u;  break;   /* FLUSHING → FLUSHED */
        case 9u:  next = 10u; break;   /* STOPPING → STOPPED */
        case 11u: next = 12u; break;   /* CLOSING → CLOSED */
        default:  next = 2u;  break;   /* → OPEN */
        }
        if (r2) ctx.mem.write32(r2, next);
        ret32(0u); break;
    }
    case SVC_GETAUXVAL: {
        /* getauxval(type) — ARMv7 NEON device profile.  Reading the host's
         * /proc/self/auxv would misreport x86 HWCAP bits in 64-bit entries. */
        uint32_t v = 0;
        switch (r0) {
        case 16u: v = 0x0017B0D7u; break; /* AT_HWCAP: SWP|HALF|THUMB|FAST_MULT|
                                           * VFP|EDSP|NEON|VFPv3|TLS|VFPv4|
                                           * IDIVA|IDIVT */
        case 26u: v = 0u; break;          /* AT_HWCAP2 */
        case 6u:  v = 4096u; break;       /* AT_PAGESZ */
        default:  v = 0u; break;
        }
        fprintf(stderr, "[arm_exec] getauxval(%u) → 0x%08x\n", r0, v);
        ret32(v);
        break;
    }

    /* ---- EGL ---- */

    case SVC_EGL_GETDISPLAY: {
        /* eglGetDisplay(display_type) */
        if (g_egl_dpy == EGL_NO_DISPLAY) {
            ensure_glfw_window();
            Display *x11 = glfwGetX11Display();
            g_egl_dpy = eglGetDisplay(x11 ? (EGLNativeDisplayType)x11
                                           : EGL_DEFAULT_DISPLAY);
            fprintf(stderr, "[arm_exec] eglGetDisplay → %p (x11=%p)\n",
                    g_egl_dpy, x11);
        }
        ret32(g_egl_dpy != EGL_NO_DISPLAY ? ARM_EGL_DISPLAY : 0u);
        break;
    }
    case SVC_EGL_INITIALIZE: {
        /* eglInitialize(dpy, major*, minor*) */
        EGLint major = 0, minor = 0;
        EGLBoolean ok = (g_egl_dpy != EGL_NO_DISPLAY)
            ? eglInitialize(g_egl_dpy, &major, &minor) : EGL_FALSE;
        fprintf(stderr, "[arm_exec] eglInitialize → ok=%d %d.%d\n", ok, major, minor);
        if (r1) ctx.mem.write32(r1, (uint32_t)major);
        if (r2) ctx.mem.write32(r2, (uint32_t)minor);
        ret32((uint32_t)ok);
        break;
    }
    case SVC_EGL_CHOOSECONFIG: {
        /* eglChooseConfig(dpy, attribs, configs, config_size, num_configs)
 * r0=dpy r1=attribs_va r2=configs_va r3=config_size [sp]=num_va */
        uint32_t num_va = regs[13] ? ctx.mem.read32(regs[13]) : 0u;
        uint32_t cfg_size = r3;
        EGLint   num    = 0;
        EGLBoolean ok   = EGL_FALSE;
        EGLConfig host_cfgs[64] = {};
        if (g_egl_dpy != EGL_NO_DISPLAY) {
            auto attrs = read_egl_attribs(ctx, r1);
            
            EGLint want = r2 ? (EGLint)std::min<uint32_t>(cfg_size, 64u) : 0;
            ok = eglChooseConfig(g_egl_dpy, attrs.data(),
                                 r2 ? host_cfgs : nullptr, want, &num);
            if (ok != EGL_TRUE || num == 0) {
                
                const EGLint fallback[] = {
                    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
                    EGL_SURFACE_TYPE,    g_glfw ? (EGL_WINDOW_BIT | EGL_PBUFFER_BIT)
                                                : EGL_PBUFFER_BIT,
                    EGL_RED_SIZE,   8, EGL_GREEN_SIZE, 8,
                    EGL_BLUE_SIZE,  8, EGL_DEPTH_SIZE, 16,
                    EGL_NONE
                };
                num = 0;
                ok = eglChooseConfig(g_egl_dpy, fallback,
                                     r2 ? host_cfgs : nullptr,
                                     r2 ? 64 : 0, &num);
                if (ok == EGL_TRUE && num > 0)
                    fprintf(stderr, "[arm_exec] eglChooseConfig: strict attribs "
                            "unmatched on host — using relaxed fallback config\n");
            }
        }
        fprintf(stderr, "[arm_exec] eglChooseConfig → ok=%d num=%d (size=%u)\n",
                ok, num, cfg_size);
        if (r2 && num > 0) {
            uint32_t n_out = std::min<uint32_t>((uint32_t)num, cfg_size);
            for (uint32_t i = 0; i < n_out; ++i) {
                
                uint32_t idx = ~0u;
                for (uint32_t k = 0; k < g_egl_cfg_tab.size(); ++k)
                    if (g_egl_cfg_tab[k] == host_cfgs[i]) { idx = k; break; }
                if (idx == ~0u) {
                    idx = (uint32_t)g_egl_cfg_tab.size();
                    g_egl_cfg_tab.push_back(host_cfgs[i]);
                }
                ctx.mem.write32(r2 + i * 4u, ARM_EGL_CFGTAB_BASE + idx);
            }
            num = (EGLint)n_out;
        }
        if (num_va)        ctx.mem.write32(num_va, (uint32_t)num);
        ret32((uint32_t)ok);
        break;
    }
    case SVC_EGL_CREATEWSURF: {
        /* eglCreateWindowSurface(dpy, cfg, window, attribs) */
        if (g_egl_dpy == EGL_NO_DISPLAY || !g_egl_cfg) {
            ensure_glfw_window();
            init_host_egl();
        }
        if (g_egl_dpy == EGL_NO_DISPLAY || !g_egl_cfg)
            { ret32(0u); break; }
        if (g_egl_surf == EGL_NO_SURFACE && g_glfw) {
            Window x11win = glfwGetX11Window(g_glfw);
            g_egl_surf = eglCreateWindowSurface(
                g_egl_dpy, g_egl_cfg, (EGLNativeWindowType)x11win, nullptr);
        }
        if (g_egl_surf == EGL_NO_SURFACE) {
            static const EGLint pb_attribs[] = {
                EGL_WIDTH, 1280, EGL_HEIGHT, 720, EGL_NONE
            };
            g_egl_surf = eglCreatePbufferSurface(g_egl_dpy, g_egl_cfg, pb_attribs);
        }
        fprintf(stderr, "[arm_exec] eglCreateWindowSurface → %p (glfw=%d)\n",
                g_egl_surf, g_glfw != nullptr);
        ret32(g_egl_surf != EGL_NO_SURFACE ? ARM_EGL_SURFACE : 0u);
        break;
    }
    case SVC_EGL_CREATEPBUF: {
        /* eglCreatePbufferSurface — reuse window surface as fallback */
        ret32(g_egl_surf != EGL_NO_SURFACE ? ARM_EGL_SURFACE : 0u);
        break;
    }
    case SVC_EGL_CREATECTX: {
        /* eglCreateContext(dpy, cfg, share, attribs) */
        if (g_egl_dpy == EGL_NO_DISPLAY || !g_egl_cfg)
            { ret32(0u); break; }
        if (g_egl_ctx == EGL_NO_CONTEXT) {
            auto attrs = read_egl_attribs(ctx, r3);
            /* Try with Unity's requested attribs first, fall back to GLES2 */
            g_egl_ctx = eglCreateContext(
                g_egl_dpy, g_egl_cfg, EGL_NO_CONTEXT, attrs.data());
            if (g_egl_ctx == EGL_NO_CONTEXT) {
                static const EGLint fallback[] =
                    {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
                g_egl_ctx = eglCreateContext(
                    g_egl_dpy, g_egl_cfg, EGL_NO_CONTEXT, fallback);
            }
        }
        fprintf(stderr, "[arm_exec] eglCreateContext → %p\n", g_egl_ctx);
        ret32(g_egl_ctx != EGL_NO_CONTEXT ? ARM_EGL_CONTEXT : 0u);
        break;
    }
    case SVC_EGL_MAKECURRENT: {
        /* eglMakeCurrent(dpy, draw, read, ctx) */
        EGLSurface draw = (r1 && r1 != ~0u) ? g_egl_surf : EGL_NO_SURFACE;
        EGLSurface read_s = (r2 && r2 != ~0u) ? g_egl_surf : EGL_NO_SURFACE;
        EGLContext ectx = (r3 && r3 != ~0u) ? g_egl_ctx  : EGL_NO_CONTEXT;
        /* Unity 4.x releases the context (NO_CTX) expecting GLSurfaceView to re-bind
 * each frame.  In the emulated environment there is no GLSurfaceView thread to
 * re-bind it, so honour the unbind only if the ARM guest explicitly passes
 * EGL_NO_CONTEXT; otherwise keep the context current so GL calls succeed. */
        if (ectx == EGL_NO_CONTEXT) {
            ret32(EGL_TRUE);
            break;
        }
        EGLBoolean ok = (g_egl_dpy != EGL_NO_DISPLAY)
            ? eglMakeCurrent(g_egl_dpy, draw, read_s, ectx) : EGL_FALSE;
        ret32((uint32_t)ok);
        break;
    }
    case SVC_EGL_SWAPBUF: {
        /* eglSwapBuffers(dpy, surface) — also pump GLFW events.
 * Return EGL_FALSE if no real EGL context exists yet (during JNI_OnLoad),
 * so Unity knows EGL isn't ready and proceeds to initialize it properly. */
        ++g_guest_egl_swap_count;
        if (getenv("LUNARIA_TRACE_EGL")) {
            static uint64_t n = 0;
            if (n < 20 || (n % 500) == 0)
                fprintf(stderr, "[egl] guest eglSwapBuffers #%llu tid=%u egl_ready=%d\n",
                        (unsigned long long)n, g_current_tid, g_egl_dpy != EGL_NO_DISPLAY);
            ++n;
        } else {
            static uint64_t n = 0;
            if (n < 5)
                fprintf(stderr, "[egl] ARM eglSwapBuffers called tid=%u #%llu egl_ready=%d\n",
                        g_current_tid, (unsigned long long)n, g_egl_dpy != EGL_NO_DISPLAY);
            ++n;
        }
        if (g_egl_dpy == EGL_NO_DISPLAY || g_egl_surf == EGL_NO_SURFACE || g_egl_ctx == EGL_NO_CONTEXT) {
            /* No real EGL context — return failure so Unity initializes EGL */
            ret32((uint32_t)EGL_FALSE);
            break;
        }
        g_arm_did_swap = true;
        if (g_glfw) glfwPollEvents();
        
        {
            static const std::vector<uint64_t> dump_frames = [] {
                std::vector<uint64_t> v;
                if (const char *e = getenv("LUNARIA_DUMP_FRAME")) {
                    char *dup = strdup(e);
                    for (char *tok = strtok(dup, ","); tok; tok = strtok(nullptr, ","))
                        v.push_back(strtoull(tok, nullptr, 0));
                    free(dup);
                }
                return v;
            }();
            if (!dump_frames.empty()) {
                static uint64_t sw = 0;
                for (uint64_t want : dump_frames) {
                    if (sw != want) continue;
                    EGLint w = 0, h = 0;
                    eglQuerySurface(g_egl_dpy, g_egl_surf, EGL_WIDTH, &w);
                    eglQuerySurface(g_egl_dpy, g_egl_surf, EGL_HEIGHT, &h);
                    if (w > 0 && h > 0) {
                        std::vector<uint8_t> px((size_t)w * h * 4);
                        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
                        const char *dir = getenv("LUNARIA_DUMP_DIR") ?: "/tmp";
                        char path[512];
                        snprintf(path, sizeof path, "%s/frame%llu.ppm", dir,
                                 (unsigned long long)sw);
                        if (FILE *fp = fopen(path, "wb")) {
                            fprintf(fp, "P6\n%d %d\n255\n", w, h);
                            for (int y = h - 1; y >= 0; --y)       
                                for (int x = 0; x < w; ++x)
                                    fwrite(&px[((size_t)y * w + x) * 4], 1, 3, fp);
                            fclose(fp);
                            fprintf(stderr, "[egl] dumped frame %llu -> %s (%dx%d)\n",
                                    (unsigned long long)sw, path, w, h);
                        }
                    }
                }
                ++sw;
            }
        }
        EGLBoolean ok = eglSwapBuffers(g_egl_dpy, g_egl_surf);
        {
            
            static uint64_t last_draw = 0, last_clear = 0, sn = 0;
            if (sn < 40 || (sn % 50) == 0)
                fprintf(stderr, "[egl] swap#%llu draws=+%llu clears=+%llu "
                        "viewport=%d,%d,%dx%d\n",
                        (unsigned long long)sn,
                        (unsigned long long)(g_gl_draw_count - last_draw),
                        (unsigned long long)(g_guest_gl_clear_count - last_clear),
                        g_gl_last_viewport[0], g_gl_last_viewport[1],
                        g_gl_last_viewport[2], g_gl_last_viewport[3]);
            last_draw = g_gl_draw_count; last_clear = g_guest_gl_clear_count;
            ++sn;
        }
        ret32((uint32_t)ok);
        break;
    }
    case SVC_EGL_DESTROYSURF: ret32(EGL_TRUE); break;
    case SVC_EGL_DESTROYCTX:  ret32(EGL_TRUE); break;
    case SVC_EGL_TERMINATE:   ret32(EGL_TRUE); break;
    case SVC_EGL_SWAPINTERVAL: {
        EGLBoolean ok = (g_egl_dpy != EGL_NO_DISPLAY)
            ? eglSwapInterval(g_egl_dpy, (EGLint)r1) : EGL_FALSE;
        ret32((uint32_t)ok);
        break;
    }
    case SVC_EGL_GETPROC: {
        /* Look up the proc name in kSymbolSvcMap; return its SVC trampoline.
 * Return 0 (null) for unknown functions so Unity does not call through
 * a fake pointer and accidentally write to ARM address 0, corrupting
 * the code segment (which caused ExceptionRaised pc=0x4 + stack overflow). */
        const char *procname = ARM_STR(r0);
        uint32_t tramp_addr = 0u; /* default: null (function not available) */
        if (procname) {
            uint32_t s = lookup_symbol_svc(procname);
            if (s != UINT32_MAX) tramp_addr = TRAMP_BASE + s * TRAMP_STRIDE;
            static int gp_count = 0;
            if (gp_count < 300)
                fprintf(stderr, "[arm_exec] eglGetProcAddress(%s) -> 0x%08x tid=%u\n",
                        procname, tramp_addr, g_current_tid);
            ++gp_count;
        }
        ret32(tramp_addr);
        break;
    }
    case SVC_EGL_QUERYSURF: {
        /* eglQuerySurface(dpy, surface, attr, value) */
        EGLint val = 0;
        if (g_egl_dpy != EGL_NO_DISPLAY && g_egl_surf != EGL_NO_SURFACE)
            eglQuerySurface(g_egl_dpy, g_egl_surf, (EGLint)r2, &val);
        {
            static int qs_log = 0;
            if (qs_log++ < 24)
                fprintf(stderr, "[egl] QuerySurface(attr=0x%x) -> %d (surf=%p)\n",
                        r2, (int)val, g_egl_surf);
        }
        if (r3) ctx.mem.write32(r3, (uint32_t)val);
        ret32(EGL_TRUE);
        break;
    }
    case SVC_EGL_GETERROR:
        ret32((uint32_t)(g_egl_dpy != EGL_NO_DISPLAY
            ? eglGetError() : EGL_SUCCESS));
        break;
    case SVC_EGL_GETCFGATTRIB: {
        
        EGLint val = 0;
        EGLConfig hc = resolve_egl_config(r1);
        if (g_egl_dpy != EGL_NO_DISPLAY && hc)
            eglGetConfigAttrib(g_egl_dpy, hc, (EGLint)r2, &val);
        if (getenv("LUNARIA_TRACE_EGL")) {
            static int cfg_log = 0;
            if (cfg_log++ < 64)
                fprintf(stderr, "[egl] GetConfigAttrib(cfg=%#x attr=0x%x) -> %d\n",
                        r1, r2, (int)val);
        }
        if (r3) ctx.mem.write32(r3, (uint32_t)val);
        ret32(EGL_TRUE);
        break;
    }
    case SVC_EGL_QUERYSTR: {
        /* eglQueryString(dpy, name) */
        const char *s = (g_egl_dpy != EGL_NO_DISPLAY)
            ? eglQueryString(g_egl_dpy, (EGLint)r1) : nullptr;
        if (s) {
            size_t len = std::min(strlen(s), (size_t)4095u);
            if (auto *p = ctx.mem.ptr(STR_SCRATCH)) {
                memcpy(p, s, len); p[len] = '\0';
            }
            ret32(STR_SCRATCH);
        } else {
            ret32(0u);
        }
        break;
    }
    case SVC_EGL_SURFACEATTRIB: ret32(EGL_TRUE); break;
    case SVC_EGL_GETCURCTX:
        ret32(g_egl_ctx != EGL_NO_CONTEXT ? ARM_EGL_CONTEXT : 0u); break;
    case SVC_EGL_GETCURSURF:
        ret32(g_egl_surf != EGL_NO_SURFACE ? ARM_EGL_SURFACE : 0u); break;

    /* eglGetSystemTimeFrequencyNV() → uint64_t ticks-per-second
 * On real Tegra hardware this is ~1e9 (nanosecond resolution).
 * Return value in r0:r1 (ARM32 AAPCS u64 calling convention). */
    case SVC_EGL_SYSTIME_FREQ: {
        static int stf_log = 0;
        if (stf_log++ < 3)
            fprintf(stderr, "[egl] eglGetSystemTimeFrequencyNV() → 1000000000\n");
        regs[0] = 1000000000u; /* low word = 1e9 */
        regs[1] = 0u;          /* high word = 0 */
        break;
    }
    /* eglGetSystemTimeNV() → uint64_t nanoseconds since boot */
    case SVC_EGL_SYSTIME: {
        struct timespec ts = {};
        clock_gettime(CLOCK_BOOTTIME, &ts);
        uint64_t ns = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
        regs[0] = (uint32_t)(ns & 0xffffffffull);
        regs[1] = (uint32_t)(ns >> 32);
        break;
    }

    /* sigsuspend(mask): in normal context just return -1/EINTR.
 * In call_guest_cb context (GC_suspend_handler loop), the Boehm GC
 * code in libmono.so spins: clear struct->signal, call sigsuspend,
 * check struct->signal == SIGXCPU(24), loop if not.
 * The struct is at [fp - 0x14] (fp = r11), signal field at +0xc.
 * Write 24 so the loop exits on the very first iteration. */
    case SVC_SIGSUSPEND: {
        static int ss_log = 0;
        if (g_in_cb) {
            /* GC_suspend_handler_inner do-while loop:
 * do { sigsuspend(&mask); } while (me->signal != SIGXCPU);
 * Our SVC trampoline is just "svc #n; bx lr" (no push/pop).
 * The return address is in LR (regs[14]).  After the SVC, the
 * trampoline executes "bx lr", jumping to wherever LR points.
 * The loop exit in GC_suspend_handler_inner is LR + 0x10:
 * LR+0x00: ldr r3, [r11, #-0x14]   (reload me)
 * LR+0x04: ldr r3, [r3, #0xc]       (read me->signal)
 * LR+0x08: cmp r3, #24               (compare to SIGXCPU)
 * LR+0x0c: bne <loop_top>            (branch if !=)
 * LR+0x10: ...                       (loop exit) ← target
 * Set LR += 0x10 so "bx lr" jumps directly to the loop exit,
 * bypassing the condition check entirely. */
            if (ss_log++ < 5)
                fprintf(stderr, "[sigsuspend] in_cb: skip loop cond lr=0x%08x→0x%08x "
                        "tid=%u\n", regs[14], regs[14] + 0x10u, g_current_tid);
            regs[14] += 0x10u; /* BX LR will jump to loop exit */
            ret32(0); break;
        }
        if (ss_log++ < 5)
            fprintf(stderr, "[sigsuspend] r0=0x%08x tid=%u → -1/EINTR\n", r0, g_current_tid);
        uint32_t eva = errno_va(ctx, g_current_tid);
        if (eva) ctx.mem.write32(eva, 4 /* EINTR */);
        ret32(~0u); break;
    }

    /* dl_unwind_find_exidx(pc, *pcount):
 * Return the real .ARM.exidx of the module containing pc — libc++abi's
 * unwinder walks it to find C++ catch handlers, so IL2CPP's thrown
 * Il2CppExceptionWrapper can reach its catch instead of std::terminate.
 * Fall back to a one-entry CANTUNWIND table for unknown PCs so a stack
 * walk through stub/trampoline code still terminates cleanly. */
    case SVC_DL_UNWIND_EXIDX: {
        static int dl_call_count = 0;
        uint32_t hit_va = 0, hit_count = 0;
        for (const auto &m : g_module_exidx)
            if (r0 >= m.lo && r0 < m.hi)
                { hit_va = m.exidx_va; hit_count = m.exidx_count; break; }
        if (dl_call_count < 8)
            fprintf(stderr, "[arm_exec] dl_unwind_find_exidx pc=0x%08x → "
                    "exidx=0x%08x count=%u (call %d)\n",
                    r0, hit_va, hit_count, ++dl_call_count);
        else ++dl_call_count;
        if (hit_va) {
            if (r1) ctx.mem.write32(r1, hit_count);
            ret32(hit_va);
            break;
        }
        /* Lazy-allocate a 2-word ARM buffer for the fake EXIDX entry */
        static uint32_t fake_exidx_va = 0;
        if (!fake_exidx_va) {
            fake_exidx_va = arm_malloc(ctx, 8);
            /* word0: PREL31(entry_addr → 0).  Any PC ≥ 0 will match this entry. */
            uint32_t prel31 = (uint32_t)(-(int32_t)fake_exidx_va) & 0x7FFFFFFFu;
            ctx.mem.write32(fake_exidx_va,     prel31);
            ctx.mem.write32(fake_exidx_va + 4, 0x00000001u); /* CANTUNWIND */
        }
        if (r1) ctx.mem.write32(r1, 1u); /* pcount = 1 */
        ret32(fake_exidx_va);
        break;
    }

    /* ---- OpenGL ES 2.0 ---- */
    /* Helper to read host pointer from ARM VA (for data buffers) */
#define ARM_PTR(va) ((va) ? (void*)ctx.mem.ptr(va) : nullptr)
#define ARM_CPTR(va) ((va) ? (const void*)ctx.mem.ptr(va) : nullptr)
#define ARM_STR_GL(va) ((va) ? ctx.mem.cstr(va) : nullptr)

    case SVC_GL_Viewport:
        { static int n=0; if(n<3) fprintf(stderr,"[gl] glViewport(%d,%d,%d,%d) tid=%u\n",(int)r0,(int)r1,(int)r2,(int)r3,g_current_tid); ++n; }
        g_gl_last_viewport[0]=(int)r0; g_gl_last_viewport[1]=(int)r1;
        g_gl_last_viewport[2]=(int)r2; g_gl_last_viewport[3]=(int)r3;
        if (pfn_glViewport) pfn_glViewport((GLint)r0,(GLint)r1,(GLsizei)r2,(GLsizei)r3);
        break;
    case SVC_GL_Clear:
        ++g_guest_gl_clear_count;
        if (getenv("LUNARIA_TRACE_EGL") || getenv("LUNARIA_TRACE_GL")) {
            static int n = 0;
            if (n < 20 || getenv("LUNARIA_TRACE_EGL"))
                fprintf(stderr, "[gl] glClear(0x%x) tid=%u #%d\n", r0, g_current_tid, n);
            ++n;
        }
        if (pfn_glClear) pfn_glClear((GLbitfield)r0); break;
    case SVC_GL_ClearColor:
        if (pfn_glClearColor) pfn_glClearColor(rf(r0),rf(r1),rf(r2),rf(r3)); break;
    case SVC_GL_ClearDepthf:
        if (pfn_glClearDepthf) pfn_glClearDepthf(rf(r0)); break;
    case SVC_GL_ClearStencil:
        if (pfn_glClearStencil) pfn_glClearStencil((GLint)r0); break;
    case SVC_GL_Enable:
        if (pfn_glEnable) pfn_glEnable((GLenum)r0); break;
    case SVC_GL_Disable:
        if (pfn_glDisable) pfn_glDisable((GLenum)r0); break;
    case SVC_GL_DepthFunc:
        if (pfn_glDepthFunc) pfn_glDepthFunc((GLenum)r0); break;
    case SVC_GL_DepthMask:
        if (pfn_glDepthMask) pfn_glDepthMask((GLboolean)r0); break;
    case SVC_GL_ColorMask:
        if (pfn_glColorMask) pfn_glColorMask((GLboolean)r0,(GLboolean)r1,(GLboolean)r2,(GLboolean)r3); break;
    case SVC_GL_Scissor:
        if (pfn_glScissor) pfn_glScissor((GLint)r0,(GLint)r1,(GLsizei)r2,(GLsizei)r3); break;
    case SVC_GL_FrontFace:
        if (pfn_glFrontFace) pfn_glFrontFace((GLenum)r0); break;
    case SVC_GL_CullFace:
        if (pfn_glCullFace) pfn_glCullFace((GLenum)r0); break;
    case SVC_GL_BlendFuncSeparate:
        if (pfn_glBlendFuncSeparate) pfn_glBlendFuncSeparate((GLenum)r0,(GLenum)r1,(GLenum)r2,(GLenum)r3); break;
    case SVC_GL_BlendEquationSeparate:
        if (pfn_glBlendEquationSeparate) pfn_glBlendEquationSeparate((GLenum)r0,(GLenum)r1); break;
    case SVC_GL_GetError:
        ret32(pfn_glGetError ? (uint32_t)pfn_glGetError() : 0u); break;
    case SVC_GL_GetString: {
        /* Stash GL string in scratch, return ARM VA */
        const char *s = pfn_glGetString ? (const char*)pfn_glGetString((GLenum)r0) : nullptr;
        if (s) {
            size_t len = std::min(strlen(s),(size_t)4095u);
            if (auto *p = ctx.mem.ptr(STR_SCRATCH)) { memcpy(p,s,len); p[len]='\0'; }
            ret32(STR_SCRATCH);
        } else ret32(0u);
        break;
    }
    case SVC_GL_GetIntegerv:
        if (pfn_glGetIntegerv && r1) pfn_glGetIntegerv((GLenum)r0,(GLint*)ctx.mem.ptr(r1)); break;
    case SVC_GL_PixelStorei:
        if (pfn_glPixelStorei) pfn_glPixelStorei((GLenum)r0,(GLint)r1); break;
    case SVC_GL_ReadPixels:
        /* r0=x r1=y r2=w r3=h sp[0]=fmt sp[1]=type sp[2]=pixels_va */
        if (pfn_glReadPixels) {
            uint32_t fmt  = ctx.mem.read32(regs[13]);
            uint32_t type = ctx.mem.read32(regs[13]+4);
            uint32_t pva  = ctx.mem.read32(regs[13]+8);
            pfn_glReadPixels((GLint)r0,(GLint)r1,(GLsizei)r2,(GLsizei)r3,(GLenum)fmt,(GLenum)type,ARM_PTR(pva));
        }
        break;
    case SVC_GL_Flush:  if (pfn_glFlush)  pfn_glFlush();  break;
    case SVC_GL_Finish: if (pfn_glFinish) pfn_glFinish(); break;

    
    case SVC_GL_GenBuffers:
        if (pfn_glGenBuffers && r1) pfn_glGenBuffers((GLsizei)r0,(GLuint*)ctx.mem.ptr(r1)); break;
    case SVC_GL_BindBuffer:
        if (r0 == 0x8892u) g_gl_bound_array_buf = r1;
        else if (r0 == 0x8893u) g_gl_bound_elem_buf = r1;
        if (pfn_glBindBuffer) pfn_glBindBuffer((GLenum)r0,(GLuint)r1); break;
    case SVC_GL_BufferData:
        if (pfn_glBufferData) pfn_glBufferData((GLenum)r0,(GLsizeiptr)r1,ARM_CPTR(r2),(GLenum)r3); break;
    case SVC_GL_BufferSubData:
        /* 4 args → data pointer arrives in r3 per AAPCS (was wrongly read
 * from [sp], uploading stack garbage into VBOs) */
        if (pfn_glBufferSubData) pfn_glBufferSubData((GLenum)r0,(GLintptr)r1,(GLsizeiptr)r2,ARM_CPTR(r3)); break;
    case SVC_GL_DeleteBuffers:
        if (r1) {
            /* GL semantics: deleting the currently bound buffer unbinds it */
            for (uint32_t i = 0; i < r0; ++i) {
                uint32_t b = ctx.mem.read32(r1 + i*4);
                if (b == g_gl_bound_array_buf) g_gl_bound_array_buf = 0;
                if (b == g_gl_bound_elem_buf)  g_gl_bound_elem_buf  = 0;
            }
        }
        if (pfn_glDeleteBuffers && r1) pfn_glDeleteBuffers((GLsizei)r0,(const GLuint*)ctx.mem.ptr(r1)); break;

    
    case SVC_GL_GenTextures:
        if (pfn_glGenTextures && r1) pfn_glGenTextures((GLsizei)r0,(GLuint*)ctx.mem.ptr(r1)); break;
    case SVC_GL_BindTexture:
        if (pfn_glBindTexture) pfn_glBindTexture((GLenum)r0,(GLuint)r1); break;
    case SVC_GL_ActiveTexture:
        if (pfn_glActiveTexture) pfn_glActiveTexture((GLenum)r0); break;
    case SVC_GL_DeleteTextures:
        if (pfn_glDeleteTextures && r1) pfn_glDeleteTextures((GLsizei)r0,(const GLuint*)ctx.mem.ptr(r1)); break;
    case SVC_GL_TexParameteri:
        if (pfn_glTexParameteri) pfn_glTexParameteri((GLenum)r0,(GLenum)r1,(GLint)r2); break;
    case SVC_GL_TexImage2D: {
        /* r0=target r1=level r2=internalfmt r3=width sp[0]=height sp[1]=border
 * sp[2]=format sp[3]=type sp[4]=pixels_va */
        if (pfn_glTexImage2D) {
            GLsizei h   = (GLsizei)ctx.mem.read32(regs[13]);
            GLint   brd = (GLint)  ctx.mem.read32(regs[13]+4);
            GLenum  fmt = (GLenum) ctx.mem.read32(regs[13]+8);
            GLenum  typ = (GLenum) ctx.mem.read32(regs[13]+12);
            uint32_t pva= ctx.mem.read32(regs[13]+16);
            pfn_glTexImage2D((GLenum)r0,(GLint)r1,(GLint)r2,(GLsizei)r3,h,brd,fmt,typ,ARM_CPTR(pva));
        }
        break;
    }
    case SVC_GL_TexSubImage2D: {
        /* r0=target r1=level r2=xoff r3=yoff sp[0]=w sp[1]=h sp[2]=fmt sp[3]=type sp[4]=pva */
        if (pfn_glTexSubImage2D) {
            GLsizei w   = (GLsizei)ctx.mem.read32(regs[13]);
            GLsizei h   = (GLsizei)ctx.mem.read32(regs[13]+4);
            GLenum  fmt = (GLenum) ctx.mem.read32(regs[13]+8);
            GLenum  typ = (GLenum) ctx.mem.read32(regs[13]+12);
            uint32_t pva= ctx.mem.read32(regs[13]+16);
            pfn_glTexSubImage2D((GLenum)r0,(GLint)r1,(GLint)r2,(GLint)r3,w,h,fmt,typ,ARM_CPTR(pva));
        }
        break;
    }
    case SVC_GL_CopyTexSubImage2D: {
        /* r0=target r1=level r2=xoff r3=yoff sp[0]=x sp[1]=y sp[2]=w sp[3]=h */
        if (pfn_glCopyTexSubImage2D) {
            GLint x=(GLint)ctx.mem.read32(regs[13]), y=(GLint)ctx.mem.read32(regs[13]+4);
            GLsizei w=(GLsizei)ctx.mem.read32(regs[13]+8), h=(GLsizei)ctx.mem.read32(regs[13]+12);
            pfn_glCopyTexSubImage2D((GLenum)r0,(GLint)r1,(GLint)r2,(GLint)r3,x,y,w,h);
        }
        break;
    }
    case SVC_GL_CompressedTexImage2D: {
        /* r0=target r1=level r2=internalfmt r3=width sp[0]=h sp[1]=border sp[2]=size sp[3]=pva */
        if (pfn_glCompressedTexImage2D) {
            GLsizei h  = (GLsizei)ctx.mem.read32(regs[13]);
            GLint   brd= (GLint)  ctx.mem.read32(regs[13]+4);
            GLsizei sz = (GLsizei)ctx.mem.read32(regs[13]+8);
            uint32_t pva = ctx.mem.read32(regs[13]+12);
            pfn_glCompressedTexImage2D((GLenum)r0,(GLint)r1,(GLenum)r2,(GLsizei)r3,h,brd,sz,ARM_CPTR(pva));
        }
        break;
    }
    case SVC_GL_CompressedTexSubImage2D: {
        /* r0=target r1=level r2=xoff r3=yoff sp[0]=w sp[1]=h sp[2]=fmt sp[3]=size sp[4]=pva */
        if (pfn_glCompressedTexSubImage2D) {
            GLsizei w  = (GLsizei)ctx.mem.read32(regs[13]);
            GLsizei h  = (GLsizei)ctx.mem.read32(regs[13]+4);
            GLenum  fmt= (GLenum) ctx.mem.read32(regs[13]+8);
            GLsizei sz = (GLsizei)ctx.mem.read32(regs[13]+12);
            uint32_t pva = ctx.mem.read32(regs[13]+16);
            pfn_glCompressedTexSubImage2D((GLenum)r0,(GLint)r1,(GLint)r2,(GLint)r3,w,h,fmt,sz,ARM_CPTR(pva));
        }
        break;
    }
    case SVC_GL_GenerateMipmap:
        if (pfn_glGenerateMipmap) pfn_glGenerateMipmap((GLenum)r0); break;

    
    case SVC_GL_GenFramebuffers:
        if (pfn_glGenFramebuffers && r1) pfn_glGenFramebuffers((GLsizei)r0,(GLuint*)ctx.mem.ptr(r1)); break;
    case SVC_GL_BindFramebuffer:
        if (pfn_glBindFramebuffer) pfn_glBindFramebuffer((GLenum)r0,(GLuint)r1); break;
    case SVC_GL_DeleteFramebuffers:
        if (pfn_glDeleteFramebuffers && r1) pfn_glDeleteFramebuffers((GLsizei)r0,(const GLuint*)ctx.mem.ptr(r1)); break;
    case SVC_GL_CheckFramebufferStatus:
        ret32(pfn_glCheckFramebufferStatus ? (uint32_t)pfn_glCheckFramebufferStatus((GLenum)r0) : 0x8CD5u); break;
    case SVC_GL_FramebufferTexture2D:
        if (pfn_glFramebufferTexture2D) pfn_glFramebufferTexture2D((GLenum)r0,(GLenum)r1,(GLenum)r2,(GLuint)r3,(GLint)ctx.mem.read32(regs[13])); break;
    case SVC_GL_FramebufferRenderbuffer:
        if (pfn_glFramebufferRenderbuffer) pfn_glFramebufferRenderbuffer((GLenum)r0,(GLenum)r1,(GLenum)r2,(GLuint)r3); break;
    case SVC_GL_GetFramebufferAttachmentParameteriv:
        if (pfn_glGetFramebufferAttachmentParameteriv && r3)
            pfn_glGetFramebufferAttachmentParameteriv((GLenum)r0,(GLenum)r1,(GLenum)r2,(GLint*)ctx.mem.ptr(r3)); break;

    
    case SVC_GL_GenRenderbuffers:
        if (pfn_glGenRenderbuffers && r1) pfn_glGenRenderbuffers((GLsizei)r0,(GLuint*)ctx.mem.ptr(r1)); break;
    case SVC_GL_BindRenderbuffer:
        if (pfn_glBindRenderbuffer) pfn_glBindRenderbuffer((GLenum)r0,(GLuint)r1); break;
    case SVC_GL_DeleteRenderbuffers:
        if (pfn_glDeleteRenderbuffers && r1) pfn_glDeleteRenderbuffers((GLsizei)r0,(const GLuint*)ctx.mem.ptr(r1)); break;
    case SVC_GL_RenderbufferStorage:
        if (pfn_glRenderbufferStorage) pfn_glRenderbufferStorage((GLenum)r0,(GLenum)r1,(GLsizei)r2,(GLsizei)r3); break;

    
    case SVC_GL_CreateShader:
        ret32(pfn_glCreateShader ? (uint32_t)pfn_glCreateShader((GLenum)r0) : 0u); break;
    case SVC_GL_ShaderSource: {
        /* r0=shader r1=count r2=strings_va r3=lengths_va */
        if (pfn_glShaderSource && r2) {
            /* Build host string array from ARM VAs */
            GLsizei cnt = (GLsizei)r1;
            std::vector<const char*> strs(cnt);
            std::vector<GLint> lens(cnt, -1);
            for (GLsizei i = 0; i < cnt; ++i) {
                uint32_t sva = ctx.mem.read32(r2 + (uint32_t)(i*4));
                strs[i] = ctx.mem.cstr(sva);
                if (r3) {
                    GLint l; uint32_t lv = ctx.mem.read32(r3 + (uint32_t)(i*4));
                    l = (GLint)lv; lens[i] = l;
                }
            }
            
            std::string src;
            for (GLsizei i = 0; i < cnt; ++i) {
                if (!strs[i]) continue;
                if (r3 && lens[i] >= 0) src.append(strs[i], (size_t)lens[i]);
                else src.append(strs[i]);
            }
            if (src.find("#extension") != std::string::npos) {
                std::string ver, ext, body;
                size_t pos = 0;
                while (pos < src.size()) {
                    size_t eol = src.find('\n', pos);
                    if (eol == std::string::npos) eol = src.size(); else ++eol;
                    std::string line = src.substr(pos, eol - pos);
                    size_t ws = line.find_first_not_of(" \t");
                    if (ws != std::string::npos &&
                        line.compare(ws, 10, "#extension") == 0)
                        ext += line;
                    else if (ws != std::string::npos && body.empty() &&
                             line.compare(ws, 8, "#version") == 0)
                        ver += line;
                    else
                        body += line;
                    pos = eol;
                }
                if (!ext.empty() && ext.back() != '\n') ext += '\n';
                src = ver + ext + body;
            }
            const char *one = src.c_str();
            GLint onelen = (GLint)src.size();
            pfn_glShaderSource((GLuint)r0, 1, &one, &onelen);
        }
        break;
    }
    case SVC_GL_CompileShader:
        if (pfn_glCompileShader) pfn_glCompileShader((GLuint)r0);
        if (pfn_glGetShaderiv) {
            GLint ok_c = 0;
            pfn_glGetShaderiv((GLuint)r0, 0x8B81 /* GL_COMPILE_STATUS */, &ok_c);
            static int shlog = 0;
            if (!ok_c && shlog++ < 8) {
                char info[512] = {};
                if (pfn_glGetShaderInfoLog) {
                    GLsizei len = 0;
                    pfn_glGetShaderInfoLog((GLuint)r0, sizeof info - 1, &len, info);
                }
                fprintf(stderr, "[gl] shader %u compile FAILED: %s\n", r0, info);
            }
        }
        break;
    case SVC_GL_DeleteShader:
        if (pfn_glDeleteShader) pfn_glDeleteShader((GLuint)r0); break;
    case SVC_GL_GetShaderiv:
        if (pfn_glGetShaderiv && r2) pfn_glGetShaderiv((GLuint)r0,(GLenum)r1,(GLint*)ctx.mem.ptr(r2)); break;
    case SVC_GL_GetShaderInfoLog: {
        /* r0=shader r1=bufSize r2=length_va r3=infoLog_va */
        if (pfn_glGetShaderInfoLog && r3) {
            GLsizei len = 0;
            std::vector<char> buf(r1 + 1);
            pfn_glGetShaderInfoLog((GLuint)r0,(GLsizei)r1,&len,buf.data());
            if (auto *p = ctx.mem.ptr(r3)) { memcpy(p,buf.data(),std::min((uint32_t)len+1,r1)); }
            if (r2) ctx.mem.write32(r2,(uint32_t)len);
        }
        break;
    }
    case SVC_GL_GetShaderSource: {
        if (pfn_glGetShaderSource && r3) {
            GLsizei len = 0;
            std::vector<char> buf(r1 + 1);
            pfn_glGetShaderSource((GLuint)r0,(GLsizei)r1,&len,buf.data());
            if (auto *p = ctx.mem.ptr(r3)) memcpy(p,buf.data(),std::min((uint32_t)len+1,r1));
            if (r2) ctx.mem.write32(r2,(uint32_t)len);
        }
        break;
    }

    
    case SVC_GL_CreateProgram:
        ret32(pfn_glCreateProgram ? (uint32_t)pfn_glCreateProgram() : 0u); break;
    case SVC_GL_AttachShader:
        if (pfn_glAttachShader) pfn_glAttachShader((GLuint)r0,(GLuint)r1); break;
    case SVC_GL_LinkProgram:
        if (pfn_glLinkProgram) pfn_glLinkProgram((GLuint)r0);
        if (pfn_glGetProgramiv) {
            GLint ok_l = 0;
            pfn_glGetProgramiv((GLuint)r0, 0x8B82 /* GL_LINK_STATUS */, &ok_l);
            static int lnlog = 0;
            if (!ok_l && lnlog++ < 8) {
                char info[512] = {};
                if (pfn_glGetProgramInfoLog) {
                    GLsizei len = 0;
                    pfn_glGetProgramInfoLog((GLuint)r0, sizeof info - 1, &len, info);
                }
                fprintf(stderr, "[gl] program %u link FAILED: %s\n", r0, info);
            }
        }
        break;
    case SVC_GL_UseProgram:
        if (pfn_glUseProgram) pfn_glUseProgram((GLuint)r0); break;
    case SVC_GL_DeleteProgram:
        if (pfn_glDeleteProgram) pfn_glDeleteProgram((GLuint)r0); break;
    case SVC_GL_GetProgramiv:
        if (pfn_glGetProgramiv && r2) pfn_glGetProgramiv((GLuint)r0,(GLenum)r1,(GLint*)ctx.mem.ptr(r2)); break;
    case SVC_GL_GetProgramInfoLog: {
        if (pfn_glGetProgramInfoLog && r3) {
            GLsizei len = 0;
            std::vector<char> buf(r1 + 1);
            pfn_glGetProgramInfoLog((GLuint)r0,(GLsizei)r1,&len,buf.data());
            if (auto *p = ctx.mem.ptr(r3)) memcpy(p,buf.data(),std::min((uint32_t)len+1,r1));
            if (r2) ctx.mem.write32(r2,(uint32_t)len);
        }
        break;
    }
    case SVC_GL_GetAttribLocation:
        ret32(pfn_glGetAttribLocation ? (uint32_t)pfn_glGetAttribLocation((GLuint)r0,ARM_STR_GL(r1)) : ~0u); break;
    case SVC_GL_GetUniformLocation:
        ret32(pfn_glGetUniformLocation ? (uint32_t)pfn_glGetUniformLocation((GLuint)r0,ARM_STR_GL(r1)) : ~0u); break;
    case SVC_GL_GetActiveAttrib: {
        /* r0=prog r1=idx r2=bufSize r3=length_va sp[0]=size_va sp[1]=type_va sp[2]=name_va */
        if (pfn_glGetActiveAttrib) {
            GLsizei len=0; GLint sz=0; GLenum tp=0;
            uint32_t size_va = ctx.mem.read32(regs[13]);
            uint32_t type_va = ctx.mem.read32(regs[13]+4);
            uint32_t name_va = ctx.mem.read32(regs[13]+8);
            std::vector<char> buf(r2+1);
            pfn_glGetActiveAttrib((GLuint)r0,(GLuint)r1,(GLsizei)r2,&len,&sz,&tp,buf.data());
            if (r3) ctx.mem.write32(r3,(uint32_t)len);
            if (size_va) ctx.mem.write32(size_va,(uint32_t)sz);
            if (type_va) ctx.mem.write32(type_va,(uint32_t)tp);
            if (name_va) if (auto *p = ctx.mem.ptr(name_va)) memcpy(p,buf.data(),std::min((uint32_t)len+1,r2));
        }
        break;
    }
    case SVC_GL_GetActiveUniform: {
        if (pfn_glGetActiveUniform) {
            GLsizei len=0; GLint sz=0; GLenum tp=0;
            uint32_t size_va = ctx.mem.read32(regs[13]);
            uint32_t type_va = ctx.mem.read32(regs[13]+4);
            uint32_t name_va = ctx.mem.read32(regs[13]+8);
            std::vector<char> buf(r2+1);
            pfn_glGetActiveUniform((GLuint)r0,(GLuint)r1,(GLsizei)r2,&len,&sz,&tp,buf.data());
            if (r3) ctx.mem.write32(r3,(uint32_t)len);
            if (size_va) ctx.mem.write32(size_va,(uint32_t)sz);
            if (type_va) ctx.mem.write32(type_va,(uint32_t)tp);
            if (name_va) if (auto *p = ctx.mem.ptr(name_va)) memcpy(p,buf.data(),std::min((uint32_t)len+1,r2));
        }
        break;
    }
    case SVC_GL_BindAttribLocation:
        if (pfn_glBindAttribLocation) pfn_glBindAttribLocation((GLuint)r0,(GLuint)r1,ARM_STR_GL(r2)); break;

    
    case SVC_GL_Uniform1i:
        if (pfn_glUniform1i) pfn_glUniform1i((GLint)r0,(GLint)r1); break;
    case SVC_GL_Uniform1iv:
        if (pfn_glUniform1iv && r2) pfn_glUniform1iv((GLint)r0,(GLsizei)r1,(const GLint*)ctx.mem.ptr(r2)); break;
    case SVC_GL_Uniform2iv:
        if (pfn_glUniform2iv && r2) pfn_glUniform2iv((GLint)r0,(GLsizei)r1,(const GLint*)ctx.mem.ptr(r2)); break;
    case SVC_GL_Uniform3iv:
        if (pfn_glUniform3iv && r2) pfn_glUniform3iv((GLint)r0,(GLsizei)r1,(const GLint*)ctx.mem.ptr(r2)); break;
    case SVC_GL_Uniform4iv:
        if (pfn_glUniform4iv && r2) pfn_glUniform4iv((GLint)r0,(GLsizei)r1,(const GLint*)ctx.mem.ptr(r2)); break;
    case SVC_GL_Uniform1fv:
        if (pfn_glUniform1fv && r2) pfn_glUniform1fv((GLint)r0,(GLsizei)r1,(const GLfloat*)ctx.mem.ptr(r2)); break;
    case SVC_GL_Uniform2fv:
        if (pfn_glUniform2fv && r2) pfn_glUniform2fv((GLint)r0,(GLsizei)r1,(const GLfloat*)ctx.mem.ptr(r2)); break;
    case SVC_GL_Uniform3fv:
        if (pfn_glUniform3fv && r2) pfn_glUniform3fv((GLint)r0,(GLsizei)r1,(const GLfloat*)ctx.mem.ptr(r2)); break;
    case SVC_GL_Uniform4fv:
        if (pfn_glUniform4fv && r2) pfn_glUniform4fv((GLint)r0,(GLsizei)r1,(const GLfloat*)ctx.mem.ptr(r2)); break;
    /* glUniformMatrix{3,4}fv(location, count, transpose, value) — 4 args, so
 * `value` arrives in r3 per AAPCS, NOT on the stack.  Reading [sp] here
 * passed random stack garbage as the matrix (all-zero / NaN MVPs → no
 * visible geometry). */
    case SVC_GL_UniformMatrix3fv: {
        if (pfn_glUniformMatrix3fv && r3) pfn_glUniformMatrix3fv((GLint)r0,(GLsizei)r1,(GLboolean)r2,(const GLfloat*)ctx.mem.ptr(r3)); break;
    }
    case SVC_GL_UniformMatrix4fv: {
        if (pfn_glUniformMatrix4fv && r3) {
            const GLfloat *m = (const GLfloat*)ctx.mem.ptr(r3);
            static int mm4log = 0;
            if (getenv("LUNARIA_TRACE_GLDRAW") && mm4log++ < 16) {
                fprintf(stderr, "[gl] UniformMatrix4fv loc=%d cnt=%u tr=%u:\n", (int)r0, r1, r2);
                for (int row = 0; row < 4; ++row)
                    fprintf(stderr, "  %9.4f %9.4f %9.4f %9.4f\n",
                            m[row], m[row+4], m[row+8], m[row+12]);
            }
            pfn_glUniformMatrix4fv((GLint)r0,(GLsizei)r1,(GLboolean)r2,m);
        }
        break;
    }

    
    case SVC_GL_EnableVertexAttribArray:
        if (pfn_glEnableVertexAttribArray) pfn_glEnableVertexAttribArray((GLuint)r0); break;
    case SVC_GL_DisableVertexAttribArray:
        if (pfn_glDisableVertexAttribArray) pfn_glDisableVertexAttribArray((GLuint)r0); break;
    case SVC_GL_VertexAttribPointer:
        /* r0=idx r1=size r2=type r3=norm sp[0]=stride sp[1]=ptr_va */
        if (pfn_glVertexAttribPointer) {
            GLsizei stride = (GLsizei)ctx.mem.read32(regs[13]);
            uint32_t pva   = ctx.mem.read32(regs[13]+4);
            /* VBO bound → pva is a byte offset, pass untranslated */
            const void *p = g_gl_bound_array_buf
                ? (const void*)(uintptr_t)pva : ARM_CPTR(pva);
            pfn_glVertexAttribPointer((GLuint)r0,(GLint)r1,(GLenum)r2,(GLboolean)r3,stride,p);
        }
        break;
    case SVC_GL_GetVertexAttribiv:
        if (pfn_glGetVertexAttribiv && r2) pfn_glGetVertexAttribiv((GLuint)r0,(GLenum)r1,(GLint*)ctx.mem.ptr(r2)); break;
    case SVC_GL_GetVertexAttribPointerv: break; /* stub */

    
    case SVC_GL_DrawArrays:
        ++g_gl_draw_count;
        if (pfn_glDrawArrays) pfn_glDrawArrays((GLenum)r0,(GLint)r1,(GLsizei)r2);
        if (g_gl_draw_count <= 12 && pfn_glGetError) {
            GLenum e = pfn_glGetError();
            fprintf(stderr, "[gl] DrawArrays#%llu mode=0x%x first=%d count=%d err=0x%x\n",
                    (unsigned long long)g_gl_draw_count, r0, (int)r1, (int)r2, e);
        }
        break;
    case SVC_GL_DrawElements:
        /* r0=mode r1=count r2=type r3=indices_va (IBO bound → byte offset) */
        ++g_gl_draw_count;
        if (pfn_glDrawElements) pfn_glDrawElements((GLenum)r0,(GLsizei)r1,(GLenum)r2,
            g_gl_bound_elem_buf ? (const void*)(uintptr_t)r3 : ARM_CPTR(r3));
        if (g_gl_draw_count <= 12 && pfn_glGetError) {
            GLenum e = pfn_glGetError();
            fprintf(stderr, "[gl] DrawElements#%llu mode=0x%x count=%d type=0x%x err=0x%x\n",
                    (unsigned long long)g_gl_draw_count, r0, (int)r1, r2, e);
        }
        if (getenv("LUNARIA_TRACE_GLDRAW") && g_gl_draw_count <= 12 && pfn_glGetIntegerv) {
            GLint prog = 0, vbo = 0, ibo = 0, depth = 0, cull = 0;
            pfn_glGetIntegerv(0x8B8D /* CURRENT_PROGRAM */, &prog);
            pfn_glGetIntegerv(0x8894 /* ARRAY_BUFFER_BINDING */, &vbo);
            pfn_glGetIntegerv(0x8895 /* ELEMENT_ARRAY_BUFFER_BINDING */, &ibo);
            if (pfn_glIsEnabled) {
                depth = pfn_glIsEnabled(0x0B71 /* DEPTH_TEST */);
                cull  = pfn_glIsEnabled(0x0B44 /* CULL_FACE */);
            }
            fprintf(stderr, "[gl]   state: prog=%d vbo=%d ibo=%d depth=%d cull=%d "
                    "(guest vbo=%u ibo=%u)\n", prog, vbo, ibo, depth, cull,
                    g_gl_bound_array_buf, g_gl_bound_elem_buf);
        }
        break;

    
    case SVC_GL_StencilFunc:
        if (pfn_glStencilFunc) pfn_glStencilFunc((GLenum)r0,(GLint)r1,(GLuint)r2); break;
    case SVC_GL_StencilFuncSeparate:
        if (pfn_glStencilFuncSeparate) pfn_glStencilFuncSeparate((GLenum)r0,(GLenum)r1,(GLint)r2,(GLuint)r3); break;
    case SVC_GL_StencilMask:
        if (pfn_glStencilMask) pfn_glStencilMask((GLuint)r0); break;
    case SVC_GL_StencilOp:
        if (pfn_glStencilOp) pfn_glStencilOp((GLenum)r0,(GLenum)r1,(GLenum)r2); break;
    case SVC_GL_StencilOpSeparate:
        if (pfn_glStencilOpSeparate) pfn_glStencilOpSeparate((GLenum)r0,(GLenum)r1,(GLenum)r2,(GLenum)r3); break;

    
    case SVC_GL_BlendFunc:
        if (pfn_glBlendFunc) pfn_glBlendFunc((GLenum)r0,(GLenum)r1); break;
    case SVC_GL_TexParameterf:
        if (pfn_glTexParameterf) pfn_glTexParameterf((GLenum)r0,(GLenum)r1,rf(r2)); break;
    case SVC_GL_DepthRangef:
        if (pfn_glDepthRangef) pfn_glDepthRangef(rf(r0),rf(r1)); break;
    case SVC_GL_PolygonOffset:
        if (pfn_glPolygonOffset) pfn_glPolygonOffset(rf(r0),rf(r1)); break;
    case SVC_GL_LineWidth:
        if (pfn_glLineWidth) pfn_glLineWidth(rf(r0)); break;
    case SVC_GL_SampleCoverage:
        if (pfn_glSampleCoverage) pfn_glSampleCoverage(rf(r0),(GLboolean)r1); break;

    
    case SVC_GL_Uniform1f:
        if (pfn_glUniform1f) pfn_glUniform1f((GLint)r0, rf(r1)); break;
    case SVC_GL_Uniform2f:
        if (pfn_glUniform2f) pfn_glUniform2f((GLint)r0, rf(r1), rf(r2)); break;
    case SVC_GL_Uniform3f:
        if (pfn_glUniform3f) pfn_glUniform3f((GLint)r0, rf(r1), rf(r2), rf(r3)); break;
    case SVC_GL_Uniform4f: {
        uint32_t v3 = ctx.mem.read32(regs[13]);
        if (pfn_glUniform4f) pfn_glUniform4f((GLint)r0, rf(r1), rf(r2), rf(r3), rf(v3));
        break;
    }
    
    case SVC_GL_VertexAttrib1f:
        if (pfn_glVertexAttrib1f) pfn_glVertexAttrib1f((GLuint)r0, rf(r1)); break;
    case SVC_GL_VertexAttrib2f:
        if (pfn_glVertexAttrib2f) pfn_glVertexAttrib2f((GLuint)r0, rf(r1), rf(r2)); break;
    case SVC_GL_VertexAttrib3f:
        if (pfn_glVertexAttrib3f) pfn_glVertexAttrib3f((GLuint)r0, rf(r1), rf(r2), rf(r3)); break;
    case SVC_GL_VertexAttrib4f: {
        uint32_t v3 = ctx.mem.read32(regs[13]);
        if (pfn_glVertexAttrib4f) pfn_glVertexAttrib4f((GLuint)r0, rf(r1), rf(r2), rf(r3), rf(v3));
        break;
    }
    case SVC_GL_VertexAttrib4fv:
        if (pfn_glVertexAttrib4fv && r1) pfn_glVertexAttrib4fv((GLuint)r0,(const GLfloat*)ctx.mem.ptr(r1)); break;
    case SVC_GL_VertexAttrib1fv:
        if (pfn_glVertexAttrib1fv && r1) pfn_glVertexAttrib1fv((GLuint)r0,(const GLfloat*)ctx.mem.ptr(r1)); break;
    case SVC_GL_VertexAttrib2fv:
        if (pfn_glVertexAttrib2fv && r1) pfn_glVertexAttrib2fv((GLuint)r0,(const GLfloat*)ctx.mem.ptr(r1)); break;
    case SVC_GL_VertexAttrib3fv:
        if (pfn_glVertexAttrib3fv && r1) pfn_glVertexAttrib3fv((GLuint)r0,(const GLfloat*)ctx.mem.ptr(r1)); break;
    
    case SVC_GL_GetFloatv:
        if (pfn_glGetFloatv && r1) pfn_glGetFloatv((GLenum)r0,(GLfloat*)ctx.mem.ptr(r1)); break;
    case SVC_GL_GetBooleanv:
        if (pfn_glGetBooleanv && r1) pfn_glGetBooleanv((GLenum)r0,(GLboolean*)ctx.mem.ptr(r1)); break;
    case SVC_GL_IsEnabled:
        ret32(pfn_glIsEnabled ? (uint32_t)pfn_glIsEnabled((GLenum)r0) : 0u); break;
    case SVC_GL_IsProgram:
        ret32(pfn_glIsProgram ? (uint32_t)pfn_glIsProgram((GLuint)r0) : 0u); break;
    case SVC_GL_IsShader:
        ret32(pfn_glIsShader ? (uint32_t)pfn_glIsShader((GLuint)r0) : 0u); break;
    case SVC_GL_IsTexture:
        ret32(pfn_glIsTexture ? (uint32_t)pfn_glIsTexture((GLuint)r0) : 0u); break;
    case SVC_GL_IsBuffer:
        ret32(pfn_glIsBuffer ? (uint32_t)pfn_glIsBuffer((GLuint)r0) : 0u); break;
    case SVC_GL_IsFramebuffer:
        ret32(pfn_glIsFramebuffer ? (uint32_t)pfn_glIsFramebuffer((GLuint)r0) : 0u); break;
    case SVC_GL_IsRenderbuffer:
        ret32(pfn_glIsRenderbuffer ? (uint32_t)pfn_glIsRenderbuffer((GLuint)r0) : 0u); break;
    
    case SVC_GL_BlendEquation:
        if (pfn_glBlendEquation) pfn_glBlendEquation((GLenum)r0); break;
    case SVC_GL_BlendColor:
        if (pfn_glBlendColor) pfn_glBlendColor(rf(r0),rf(r1),rf(r2),rf(r3)); break;
    case SVC_GL_ReleaseShaderCompiler:
        if (pfn_glReleaseShaderCompiler) pfn_glReleaseShaderCompiler(); break;
    case SVC_GL_GetShaderPrecisionFormat: {
        /* r0=shadertype r1=precisiontype r2=range_va r3=precision_va */
        if (pfn_glGetShaderPrecisionFormat)
            pfn_glGetShaderPrecisionFormat((GLenum)r0,(GLenum)r1,
                r2 ? (GLint*)ctx.mem.ptr(r2) : nullptr,
                r3 ? (GLint*)ctx.mem.ptr(r3) : nullptr);
        break;
    }
    case SVC_GL_UniformMatrix2fv: {
        uint32_t pva = ctx.mem.read32(regs[13]);
        if (pfn_glUniformMatrix2fv && pva) pfn_glUniformMatrix2fv((GLint)r0,(GLsizei)r1,(GLboolean)r2,(const GLfloat*)ctx.mem.ptr(pva)); break;
    }

#undef ARM_PTR
#undef ARM_CPTR
#undef ARM_STR_GL

    /* ---- ARM EABI integer division helpers ---- */
    case SVC_AEABI_UIDIV:
        ret32(r1 ? r0 / r1 : 0u); break;
    case SVC_AEABI_UIDIVMOD:
        if (r1) { regs[0] = r0 / r1; regs[1] = r0 % r1; }
        else    { regs[0] = 0; regs[1] = 0; }
        break;
    case SVC_AEABI_IDIV: {
        int32_t a=(int32_t)r0, b=(int32_t)r1;
        if (b == 0 || (a == INT32_MIN && b == -1)) { ret32(0); break; }
        regs[0] = (uint32_t)(a/b); regs[1] = (uint32_t)(a%b); break;
    }
    case SVC_AEABI_LDIVMOD: {
        /* signed 64-bit division: numerator r0:r1, denominator r2:r3 */
        int64_t n = (int64_t)((uint64_t)r0 | ((uint64_t)r1 << 32));
        int64_t d = (int64_t)((uint64_t)r2 | ((uint64_t)r3 << 32));
        if (d == 0 || (n == INT64_MIN && d == -1)) {
            regs[0]=regs[1]=regs[2]=regs[3]=0;
        } else {
            int64_t q = n / d, m = n % d;
            regs[0]=(uint32_t)(uint64_t)q; regs[1]=(uint32_t)((uint64_t)q>>32);
            regs[2]=(uint32_t)(uint64_t)m; regs[3]=(uint32_t)((uint64_t)m>>32);
        }
        break;
    }
    case SVC_AEABI_ULDIVMOD: {
        /* unsigned 64-bit division: numerator in r0:r1, denominator in r2:r3 */
        uint64_t n = (uint64_t)r0 | ((uint64_t)r1 << 32);
        uint64_t d = (uint64_t)r2 | ((uint64_t)r3 << 32);
        if (d) { uint64_t q=n/d, m=n%d;
                 regs[0]=(uint32_t)q; regs[1]=(uint32_t)(q>>32);
                 regs[2]=(uint32_t)m; regs[3]=(uint32_t)(m>>32); }
        else { regs[0]=regs[1]=regs[2]=regs[3]=0; }
        break;
    }

    /* ---- Host-passthrough file I/O ---- */
    /* File descriptors and FILE* are stored directly in ARM 32-bit registers.
 * Host fds are small numbers; FILE* are stored as a table index. */
    case SVC_LIBC_OPEN: {
        const char *path = ctx.mem.cstr(r0);
        if (const char *subst = synthetic_proc_path(path)) path = subst;
        int fd = guest_open_path(path, (int)r1, (mode_t)r2);
        if (getenv("LUNARIA_TRACE_OPEN"))
            fprintf(stderr, "[open] %s -> fd=%d\n", path ? path : "(null)", fd);
        ret32((uint32_t)fd); break;
    }
    case SVC_LIBC_CLOSE:
        if ((int)r0 > 2) ret32((uint32_t)close((int)r0));
        else ret32(0);
        break;
    case SVC_LIBC_READ: {
        void *buf = ctx.mem.ptr(r1);
        if (!buf) { ret32(~0u); break; }
        int fd = (int)r0;
        /* For pipe/socket reads that might block (Unity render thread sync),
 * schedule cooperative threads first so the write-side can produce data.
 * We check if the fd is non-regular (pipe/socket) via poll with timeout=0. */
        {
            struct pollfd pfd = { fd, POLLIN, 0 };
            int poll0 = poll(&pfd, 1, 0);
            if (poll0 == 0) /* only log blocking reads */
                fprintf(stderr, "[read-block] fd=%d len=%u tid=%u lr=0x%08x\n",
                        fd, r2, g_current_tid, regs[14]);
            if (poll0 == 0 && !g_threads.empty()) {
                /* No data yet — run worker threads (render thread may write to pipe) */
                for (int spin = 0; spin < 20 && poll(&pfd, 1, 0) == 0; ++spin)
                    schedule_threads(20'000'000ULL);
            }
        }
        ssize_t nr = read(fd, buf, (size_t)r2);
        if (getenv("LUNARIA_TRACE_READ")) {
            static uint64_t rc = 0;
            char lp[64] = {0};
            if (rc < 2000) {
                char proc[64]; snprintf(proc, sizeof proc, "/proc/self/fd/%d", fd);
                ssize_t ll = readlink(proc, lp, sizeof(lp) - 1); if (ll > 0) lp[ll] = '\0';
                fprintf(stderr, "[read] fd=%d req=%u nr=%zd off=%ld path=%s lr=0x%08x\n",
                        fd, r2, nr, (long)lseek(fd, 0, SEEK_CUR), lp, regs[14]);
            }
            ++rc;
        }
        ret32((uint32_t)nr); break;
    }
    case SVC_LIBC_WRITE: {
        const void *buf = ctx.mem.ptr(r1);
        ret32(buf ? (uint32_t)write((int)r0, buf, (size_t)r2) : ~0u); break;
    }
    case SVC_LIBC_LSEEK: {
        /* lseek(fd, off32, whence): r0=fd r1=offset(32bit signed) r2=whence */
        off_t off = (off_t)(int32_t)r1;
        off_t res = lseek((int)r0, off, (int)r2);
        if (getenv("LUNARIA_TRACE_LSEEK")) {
            char lp[64] = {0}; char proc[64];
            snprintf(proc, sizeof proc, "/proc/self/fd/%d", (int)r0);
            ssize_t ll = readlink(proc, lp, sizeof(lp)-1); if (ll > 0) lp[ll] = '\0';
            static const char *wnames[] = {"SET","CUR","END"};
            fprintf(stderr, "[lseek] fd=%d off=%ld whence=%s -> %ld  path=%s lr=0x%08x\n",
                    (int)r0, (long)off, (int)r2<3?wnames[r2]:"?", (long)res, lp, regs[14]);
        }
        ret32((uint32_t)res); break;
    }
    case SVC_LIBC_LSEEK64: {
        /* off64_t lseek64(int fd, off64_t offset, int whence)
 * AAPCS (non-variadic): the 64-bit offset must start in an even
 * register, so r1 is skipped as padding → offset in r2:r3, and whence
 * spills to the stack at sp[0].  (Reading r1:r2 as the offset made
 * Unity's AsyncReadManager seek to a garbage padding value, so
 * globalgamemanagers read 0 bytes → "Unknown error occurred while
 * loading".) */
        int64_t off = (int64_t)((uint64_t)r2 | ((uint64_t)r3 << 32));
        uint32_t whence = regs[13] ? ctx.mem.read32(regs[13]) : 0u;
        off_t res = lseek((int)r0, (off_t)off, (int)whence);
        if (getenv("LUNARIA_TRACE_LSEEK")) {
            char lp[64] = {0}; char proc[64];
            snprintf(proc, sizeof proc, "/proc/self/fd/%d", (int)r0);
            ssize_t ll = readlink(proc, lp, sizeof(lp)-1); if (ll > 0) lp[ll] = '\0';
            static const char *wnames[] = {"SET","CUR","END"};
            fprintf(stderr, "[lseek64] fd=%d off=%lld whence=%s -> %lld  path=%s lr=0x%08x\n",
                    (int)r0, (long long)off, whence<3?wnames[whence]:"?", (long long)res, lp, regs[14]);
        }
        
        if (res == (off_t)-1) {
            regs[0] = regs[1] = ~0u;
        } else {
            regs[0] = (uint32_t)(uint64_t)res;
            regs[1] = (uint32_t)((uint64_t)res >> 32);
        }
        break;
    }
    case SVC_LIBC_FOPEN: {
        const char *path = ctx.mem.cstr(r0);
        const char *mode = ctx.mem.cstr(r1);
        /* mono try_open_assembly passes "file:///abs/path" to fopen; strip scheme. */
        if (path && strncmp(path, "file://", 7) == 0) {
            path += 7;
            if (*path != '/') {
                const char *sl = strchr(path, '/');
                if (sl) path = sl;
            }
        }
        if (getenv("LUNARIA_TRACE_OPEN"))
            fprintf(stderr, "[fopen] %s mode=%s\n", path ? path : "(null)", mode ? mode : "(null)");
        if (const char *subst = synthetic_proc_path(path)) path = subst;
        FILE *f = (path && mode) ? fopen(path, mode) : nullptr;
        if (!f && path && mode) {
            char cache[PATH_MAX];
            if (apk_extract_to_cache(path, cache, sizeof cache))
                f = fopen(cache, mode);
        }
        ret32(register_file(ctx, f)); break;
    }
    case SVC_LIBC_FCLOSE: {
        uint32_t idx = (r0 && r0 < 256u) ? r0 : (r0 ? ctx.mem.read32(r0) : 0u);
        if (getenv("LUNARIA_TRACE_OPEN") && r0)
            fprintf(stderr, "[fclose] r0=%#x idx=%u tab=%p lr=%#x\n",
                    r0, idx, (idx > 0 && idx < 256u) ? (void*)g_file_tab[idx] : nullptr, regs[14]);
        if (idx > 0 && idx < 256u && g_file_tab[idx]) {
            fclose(g_file_tab[idx]); g_file_tab[idx] = nullptr;
        }
        if (r0 >= 256u) arm_free(ctx, r0); /* free the shim */
        ret32(0); break;
    }
    case SVC_FDOPEN: {
        int fd = (int)(int32_t)r0;
        const char *mode = ctx.mem.cstr(r1);
        FILE *f = (fd >= 0 && mode) ? fdopen(dup(fd), mode) : nullptr;
        ret32(register_file(ctx, f)); break;
    }
    case SVC_STRERROR_R: {
        
        int errnum = (int)(int32_t)r0;
        uint8_t *buf = ctx.mem.ptr(r1);
        size_t buflen = (size_t)r2;
        if (buf && buflen) {
            const char *msg = strerror(errnum);
            if (msg) {
                strncpy((char*)buf, msg, buflen - 1);
                buf[buflen - 1] = '\0';
                ret32(0);
            } else {
                snprintf((char*)buf, buflen, "Unknown error %d", errnum);
                ret32(0);
            }
        } else {
            ret32(22u /* EINVAL */);
        }
        break;
    }
    case SVC_LIBC_FREAD: {
        void *buf = ctx.mem.ptr(r0);
        FILE *f = resolve_file(ctx, r3);
        ret32((buf && f) ? (uint32_t)fread(buf, (size_t)r1, (size_t)r2, f) : 0u); break;
    }
    case SVC_LIBC_FWRITE: {
        const void *buf = ctx.mem.ptr(r0);
        FILE *f = resolve_file(ctx, r3);
        ret32((buf && f) ? (uint32_t)fwrite(buf, (size_t)r1, (size_t)r2, f) : 0u); break;
    }
    case SVC_LIBC_FSEEK: {
        FILE *f = resolve_file(ctx, r0);
        ret32(f ? (uint32_t)fseek(f, (long)(int32_t)r1, (int)r2) : ~0u); break;
    }
    case SVC_LIBC_FTELL: {
        FILE *f = resolve_file(ctx, r0);
        ret32(f ? (uint32_t)ftell(f) : ~0u); break;
    }
    case SVC_LIBC_STAT: {
        struct stat st;
        const char *path = ctx.mem.cstr(r0);
        if (getenv("LUNARIA_TRACE_OPEN"))
            fprintf(stderr, "[stat] %s\n", path ? path : "(null)");
        int rc = path ? stat(path, &st) : -1;
        if (rc != 0 && path) {
            char cache[PATH_MAX];
            if (apk_extract_to_cache(path, cache, sizeof cache))
                rc = stat(cache, &st);
        }
        /* Path is a directory *inside* the APK (App Bundle base/ prefix): the
 * zip has no explicit dir entry, so synthesize one.  Unity's IL2CPP
 * data installer stat()s assets/bin/Data/Managed before extracting it;
 * a missing source made it abort with "Not enough storage space …". */
        if (rc != 0 && path && apk_path_is_dir(path)) {
            memset(&st, 0, sizeof st);
            st.st_mode = S_IFDIR | 0755;
            st.st_nlink = 2;
            rc = 0;
        }
        if (rc != 0) { ret32(~0u); break; }
        if (r1) write_guest_stat(ctx, r1, st);
        ret32(0); break;
    }
    case SVC_LIBC_FSTAT: {
        struct stat st;
        if (getenv("LUNARIA_TRACE_OPEN"))
            fprintf(stderr, "[fstat] fd=%d\n", (int)r0);
        if (fstat((int)r0, &st) != 0) { ret32(~0u); break; }
        if (r1) write_guest_stat(ctx, r1, st);
        ret32(0); break;
    }
    /* statfs/fstatfs(path|fd, buf): report a large free filesystem so Unity's
 * resource-extraction storage check passes instead of aborting with the
 * "Not enough storage space to install required resources." dialog.
 * bionic ARM32 struct statfs layout (84 bytes, u64 8-byte aligned). */
    case SVC_STATFS: {
        uint32_t buf = r1;
        if (buf) {
            auto w64 = [&](uint32_t off, uint64_t v) {
                ctx.mem.write32(buf + off, (uint32_t)v);
                ctx.mem.write32(buf + off + 4, (uint32_t)(v >> 32));
            };
            const uint64_t blocks = 0x4000000ull; /* 64M × 4096 = 256 GB */
            ctx.mem.write32(buf + 0,  0x858458f6u); /* f_type = EXT4_SUPER_MAGIC */
            ctx.mem.write32(buf + 4,  4096u);       /* f_bsize */
            w64(8,  blocks);                        /* f_blocks */
            w64(16, blocks);                        /* f_bfree */
            w64(24, blocks);                        /* f_bavail */
            w64(32, 0x100000ull);                   /* f_files */
            w64(40, 0x100000ull);                   /* f_ffree */
            w64(48, 0);                             /* f_fsid */
            ctx.mem.write32(buf + 56, 255u);        /* f_namelen */
            ctx.mem.write32(buf + 60, 4096u);       /* f_frsize */
            ctx.mem.write32(buf + 64, 0u);          /* f_flags */
            for (uint32_t k = 0; k < 4; ++k) ctx.mem.write32(buf + 68 + k*4, 0u);
        }
        ret32(0); break;
    }
    /* statvfs/fstatvfs(path|fd, buf): same intent, POSIX layout (92 bytes). */
    case SVC_STATVFS: {
        uint32_t buf = r1;
        if (buf) {
            auto w64 = [&](uint32_t off, uint64_t v) {
                ctx.mem.write32(buf + off, (uint32_t)v);
                ctx.mem.write32(buf + off + 4, (uint32_t)(v >> 32));
            };
            const uint64_t blocks = 0x4000000ull; /* 64M × 4096 = 256 GB */
            ctx.mem.write32(buf + 0,  4096u);       /* f_bsize */
            ctx.mem.write32(buf + 4,  4096u);       /* f_frsize */
            w64(8,  blocks);                        /* f_blocks */
            w64(16, blocks);                        /* f_bfree */
            w64(24, blocks);                        /* f_bavail */
            w64(32, 0x100000ull);                   /* f_files */
            w64(40, 0x100000ull);                   /* f_ffree */
            w64(48, 0x100000ull);                   /* f_favail */
            ctx.mem.write32(buf + 56, 0u);          /* f_fsid */
            ctx.mem.write32(buf + 60, 0u);          /* f_flag */
            ctx.mem.write32(buf + 64, 255u);        /* f_namemax */
            for (uint32_t k = 0; k < 6; ++k) ctx.mem.write32(buf + 68 + k*4, 0u);
        }
        ret32(0); break;
    }
    case SVC_CHDIR: {
        const char *path = ctx.mem.cstr(r0);
        int rc = path ? chdir(path) : -1;
        if (rc == 0)
            fprintf(stderr, "[chdir] -> %s\n", path);
        ret32((uint32_t)rc); break;
    }
    case SVC_LIBC_MMAP: case SVC_LIBC_MMAP2: {
        /* mmap(addr, len, prot, flags, fd@sp[0], off@sp[1]) */
        uint32_t fd  = regs[13] ? ctx.mem.read32(regs[13])     : ~0u;
        uint32_t off = regs[13] ? ctx.mem.read32(regs[13] + 4) : 0u;
        if (svc_no == SVC_LIBC_MMAP2) off <<= 12;
        if (getenv("LUNARIA_TRACE_MMAP"))
            fprintf(stderr, "[mmap] r1=%u prot=%u flags=%#x fd=%d off=%u\n",
                    r1, r2, r3, (int32_t)fd, off);
        uint32_t addr;
        if (r0 && (r3 & 0x10u)) { /* MAP_FIXED */
            addr = r0;
        } else {
            addr = mmap_bump(r1);
            if (addr == ~0u) {
                if (getenv("LUNARIA_TRACE_MMAP"))
                    fprintf(stderr, "[mmap] -> MAP_FAILED (cap/OOM)\n");
                ret32(~0u); break;
            }
        }
        if ((int32_t)fd >= 0 && !(r3 & 0x20u /* MAP_ANONYMOUS */)) {
            
            if (pread((int)fd, ctx.mem.ptr(addr), r1, (off_t)(int64_t)(uint64_t)off) < 0) {
                if (getenv("LUNARIA_TRACE_MMAP"))
                    fprintf(stderr, "[mmap] pread failed: fd=%d off=0x%llx errno=%d\n",
                            (int32_t)fd, (unsigned long long)off, errno);
                /* fall through: already zero-filled, guest treats as anon */
            }
        }
        if (getenv("LUNARIA_TRACE_MMAP"))
            fprintf(stderr, "[mmap] -> %#x\n", addr);
        ret32(addr);
        break;
    }
    case SVC_LIBC_MUNMAP:
        ret32(0); break;

    /* ---- time / environment / misc ---- */

    case SVC_CLOCK_GETTIME: {
        /* clock_gettime(clk, struct timespec*) — ARM32: {u32 sec; u32 nsec} */
        struct timespec ts;
        clockid_t clk = (r0 == 1 /* CLOCK_MONOTONIC */) ? CLOCK_MONOTONIC : CLOCK_REALTIME;
        clock_gettime(clk, &ts);
        if (r1) {
            ctx.mem.write32(r1,     (uint32_t)ts.tv_sec);
            ctx.mem.write32(r1 + 4, (uint32_t)ts.tv_nsec);
        }
        ret32(0);
        break;
    }
    case SVC_GETTIMEOFDAY: {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        if (r0) {
            ctx.mem.write32(r0,     (uint32_t)tv.tv_sec);
            ctx.mem.write32(r0 + 4, (uint32_t)tv.tv_usec);
        }
        ret32(0);
        break;
    }
    case SVC_TIME: {
        uint32_t t = (uint32_t)time(nullptr);
        if (r0) ctx.mem.write32(r0, t);
        ret32(t);
        break;
    }
    case SVC_NANOSLEEP:
    case SVC_USLEEP:
        /* Cooperative model: main thread schedules workers; background threads
 * yield their slice so other threads get to run. */
        if (g_current_tid != 0)
            g_yield_requested = true;
        else
            maybe_schedule_on_wait();
        ret32(0);
        break;
    case SVC_GETENV: {
        const char *name = ctx.mem.cstr(r0);
        const char *val = name ? getenv(name) : nullptr;
        if (getenv("LUNARIA_TRACE_GETENV"))
            fprintf(stderr, "[getenv] %s -> %s\n", name ? name : "(null)", val ? val : "(null)");
        if (val) {
            size_t len = strlen(val) + 1;
            uint32_t addr = arm_malloc(ctx, (uint32_t)len);
            if (addr) { memcpy(ctx.mem.ptr(addr), val, len); ret32(addr); break; }
        }
        ret32(0);
        break;
    }
    case SVC_GETPID: ret32((uint32_t)getpid()); break;
    case SVC_GETTID: ret32(1000u + g_current_tid); break;
    case SVC_SCHED_YIELD: {
        /* Diagnostic: count sched_yield calls by LR range to find spin loops */
        if (getenv("LUNARIA_TRACE_YIELD")) {
            static uint64_t ycount = 0;
            uint32_t ylr = regs[14] & ~1u;
            if ((ycount & 0x3ff) == 0)
                fprintf(stderr, "[yield] #%llu lr=0x%08x tid=%u\n",
                        (unsigned long long)ycount, ylr, g_current_tid);
            ++ycount;
        }
        /* Background threads spinning on sched_yield must yield their slice so
 * other threads (e.g. the one they're waiting on) actually get to run.
 * Without this, a spin like GC's readiness poll burns the entire slice
 * and the predicate it waits for never changes. */
        if (g_current_tid != 0)
            g_yield_requested = true;
        else
            maybe_schedule_on_wait();
        ret32(0);
        break;
    }
    case SVC_GETPAGESIZE: ret32(4096u); break;
    case SVC_SYSCONF: {
        /* bionic: _SC_PAGESIZE=39, _SC_NPROCESSORS_CONF=96, _SC_NPROCESSORS_ONLN=97,
 * _SC_CLK_TCK=6 */
        switch (r0) {
        case 39: ret32(4096u); break;
        case 96: case 97: ret32(4u); break;
        case 6:  ret32(100u); break;
        default: ret32(1u); break;
        }
        break;
    }
    case SVC_RET0: ret32(0); break;

    case SVC_UNKNOWN_SYM: {
        /* Stub for a dlsym'd-but-unimplemented symbol; the trampoline slot
         * (recovered from the PC inside the trampoline window) names it. */
        uint32_t idx = (regs[15] - TRAMP_BASE) / TRAMP_STRIDE;
        const char *nm = "?";
        if (idx >= SVC_TRAMP_TOTAL &&
            idx - SVC_TRAMP_TOTAL < g_unknown_sym_names.size())
            nm = g_unknown_sym_names[idx - SVC_TRAMP_TOTAL].c_str();
        static std::map<std::string, uint32_t> seen;
        uint32_t &n = seen[nm];
        if (n < 3 || (n % 100000u) == 0)
            fprintf(stderr, "[arm_exec] unimplemented %s(r0=0x%08x r1=0x%08x "
                    "r2=0x%08x r3=0x%08x) lr=0x%08x tid=%u → 0 (call %u)\n",
                    nm, r0, r1, r2, r3, regs[14], g_current_tid, n);
        ++n;
        ret32(0);
        break;
    }

    /* ---- GLES 3.x ---- */
#define ARM_PTR(va) ((va) ? (void*)ctx.mem.ptr(va) : nullptr)
#define ARM_CPTR(va) ((va) ? (const void*)ctx.mem.ptr(va) : nullptr)
    case SVC_GL3_GetStringi: {
        const char *s = pfn_glGetStringi
            ? (const char*)pfn_glGetStringi((GLenum)r0, (GLuint)r1) : nullptr;
        if (s) {
            size_t len = std::min(strlen(s), (size_t)4095u);
            if (auto *p = ctx.mem.ptr(STR_SCRATCH)) { memcpy(p,s,len); p[len]='\0'; }
            ret32(STR_SCRATCH);
        } else ret32(0u);
        break;
    }
    case SVC_GL3_GetIntegeri_v:
        if (r2) ctx.mem.write32(r2, 0u);
        if (pfn_glGetIntegeri_v)
            pfn_glGetIntegeri_v((GLenum)r0,(GLuint)r1,(GLint*)ARM_PTR(r2));
        break;
    case SVC_GL3_GetInternalformativ: {
        uint32_t params = ctx.mem.read32(regs[13]);
        if (params && r3) ctx.mem.write32(params, 0u);
        if (pfn_glGetInternalformativ)
            pfn_glGetInternalformativ((GLenum)r0,(GLenum)r1,(GLenum)r2,
                                      (GLsizei)r3,(GLint*)ARM_PTR(params));
        break;
    }
    case SVC_GL3_GetProgramInterfaceiv:
        /* Zero the out-param first: Unity sizes its resource-enumeration
         * loops from it, and stale stack garbage once produced a 52M-
         * iteration introspection loop. */
        if (r3) ctx.mem.write32(r3, 0u);
        if (pfn_glGetProgramInterfaceiv)
            pfn_glGetProgramInterfaceiv((GLuint)r0,(GLenum)r1,(GLenum)r2,
                                        (GLint*)ARM_PTR(r3));
        break;
    case SVC_GL3_GetProgramResourceiv: {
        /* (prog, iface, index, propCount, props*, count, length*, params*) */
        uint32_t props  = ctx.mem.read32(regs[13]);
        uint32_t count  = ctx.mem.read32(regs[13]+4);
        uint32_t length = ctx.mem.read32(regs[13]+8);
        uint32_t params = ctx.mem.read32(regs[13]+12);
        if (length) ctx.mem.write32(length, 0u);
        for (uint32_t i = 0; params && i < count; ++i)
            ctx.mem.write32(params + 4u*i, 0u);
        if (pfn_glGetProgramResourceiv)
            pfn_glGetProgramResourceiv((GLuint)r0,(GLenum)r1,(GLuint)r2,
                (GLsizei)r3,(const GLenum*)ARM_CPTR(props),(GLsizei)count,
                (GLsizei*)ARM_PTR(length),(GLint*)ARM_PTR(params));
        break;
    }
    case SVC_GL3_GetProgramResourceName: {
        /* (prog, iface, index, bufSize, length*, name*) */
        uint32_t length = ctx.mem.read32(regs[13]);
        uint32_t name   = ctx.mem.read32(regs[13]+4);
        if (length) ctx.mem.write32(length, 0u);
        if (name && r3) ctx.mem.ptr(name)[0] = '\0';
        if (pfn_glGetProgramResourceName)
            pfn_glGetProgramResourceName((GLuint)r0,(GLenum)r1,(GLuint)r2,
                (GLsizei)r3,(GLsizei*)ARM_PTR(length),(GLchar*)ARM_PTR(name));
        break;
    }
    case SVC_GL3_GenVertexArrays:
        if (pfn_glGenVertexArrays)
            pfn_glGenVertexArrays((GLsizei)r0,(GLuint*)ARM_PTR(r1));
        else for (uint32_t i = 0; r1 && i < r0; ++i) {
            static uint32_t fake_vao = 1;
            ctx.mem.write32(r1 + 4u*i, fake_vao++);
        }
        break;
    case SVC_GL3_BindVertexArray:
        if (pfn_glBindVertexArray) pfn_glBindVertexArray((GLuint)r0);
        break;
    case SVC_GL3_DeleteVertexArrays:
        if (pfn_glDeleteVertexArrays)
            pfn_glDeleteVertexArrays((GLsizei)r0,(const GLuint*)ARM_CPTR(r1));
        break;
    case SVC_GL3_IsVertexArray:
        ret32(pfn_glIsVertexArray ? (uint32_t)pfn_glIsVertexArray((GLuint)r0) : 0u);
        break;
    case SVC_GL3_BindSampler:
        if (pfn_glBindSampler) pfn_glBindSampler((GLuint)r0,(GLuint)r1);
        break;
    case SVC_GL3_BindBufferBase:
        if (pfn_glBindBufferBase)
            pfn_glBindBufferBase((GLenum)r0,(GLuint)r1,(GLuint)r2);
        break;
    case SVC_GL3_BindBufferRange: {
        uint32_t off = ctx.mem.read32(regs[13]);
        uint32_t sz  = ctx.mem.read32(regs[13]+4);
        if (pfn_glBindBufferRange)
            pfn_glBindBufferRange((GLenum)r0,(GLuint)r1,(GLuint)r2,
                                  (intptr_t)off,(intptr_t)sz);
        break;
    }
    case SVC_GL3_MapBufferRange: {
        /* Host pointers can't be handed to the guest; shadow the mapping in
         * a guest buffer and copy back on flush/unmap. */
        uint32_t off = r1, len = r2, access = r3;
        void *host = pfn_glMapBufferRange
            ? pfn_glMapBufferRange((GLenum)r0,(intptr_t)off,(intptr_t)len,
                                   (GLbitfield)access) : nullptr;
        uint32_t gva = arm_malloc(ctx, len ? len : 4u);
        if (!gva) { ret32(0u); break; }
        if (host && (access & 0x0001u) /* GL_MAP_READ_BIT */)
            memcpy(ctx.mem.ptr(gva), host, len);
        g_gl_mapped[r0] = { host, gva, len, access };
        ret32(gva);
        break;
    }
    case SVC_GL3_UnmapBuffer: {
        auto it = g_gl_mapped.find(r0);
        if (it != g_gl_mapped.end()) {
            auto &m = it->second;
            /* GL_MAP_WRITE_BIT without GL_MAP_FLUSH_EXPLICIT_BIT */
            if (m.host && (m.access & 0x0002u) && !(m.access & 0x0010u))
                memcpy(m.host, ctx.mem.ptr(m.gva), m.len);
            arm_free(ctx, m.gva);
            g_gl_mapped.erase(it);
        }
        ret32(pfn_glUnmapBuffer ? (uint32_t)pfn_glUnmapBuffer((GLenum)r0) : 1u);
        break;
    }
    case SVC_GL3_FlushMappedBufferRange: {
        auto it = g_gl_mapped.find(r0);
        if (it != g_gl_mapped.end()) {
            auto &m = it->second;
            uint32_t off = r1, len = r2;
            if (m.host && off + len <= m.len)
                memcpy((char*)m.host + off, ctx.mem.ptr(m.gva + off), len);
            if (pfn_glFlushMappedBufferRange)
                pfn_glFlushMappedBufferRange((GLenum)r0,(intptr_t)off,(intptr_t)len);
        }
        break;
    }
    case SVC_GL3_TexStorage2D: {
        uint32_t h = ctx.mem.read32(regs[13]);
        if (pfn_glTexStorage2D)
            pfn_glTexStorage2D((GLenum)r0,(GLsizei)r1,(GLenum)r2,(GLsizei)r3,
                               (GLsizei)h);
        break;
    }
    case SVC_GL3_TexStorage3D: {
        uint32_t h = ctx.mem.read32(regs[13]);
        uint32_t d = ctx.mem.read32(regs[13]+4);
        if (pfn_glTexStorage3D)
            pfn_glTexStorage3D((GLenum)r0,(GLsizei)r1,(GLenum)r2,(GLsizei)r3,
                               (GLsizei)h,(GLsizei)d);
        break;
    }
    case SVC_GL3_TexSubImage3D: {
        /* (target, level, xoff, yoff | zoff, w, h, d, format, type, pixels) */
        uint32_t sp = regs[13];
        if (pfn_glTexSubImage3D)
            pfn_glTexSubImage3D((GLenum)r0,(GLint)r1,(GLint)r2,(GLint)r3,
                (GLint)ctx.mem.read32(sp),      (GLsizei)ctx.mem.read32(sp+4),
                (GLsizei)ctx.mem.read32(sp+8),  (GLsizei)ctx.mem.read32(sp+12),
                (GLenum)ctx.mem.read32(sp+16),  (GLenum)ctx.mem.read32(sp+20),
                ARM_CPTR(ctx.mem.read32(sp+24)));
        break;
    }
    case SVC_GL3_ProgramParameteri:
        if (pfn_glProgramParameteri)
            pfn_glProgramParameteri((GLuint)r0,(GLenum)r1,(GLint)r2);
        break;
    case SVC_GL3_GetProgramBinary: {
        /* (program, bufSize, length*, binaryFormat*, binary*) */
        uint32_t bin = ctx.mem.read32(regs[13]);
        if (r2) ctx.mem.write32(r2, 0u);
        if (r3) ctx.mem.write32(r3, 0u);
        if (pfn_glGetProgramBinary)
            pfn_glGetProgramBinary((GLuint)r0,(GLsizei)r1,(GLsizei*)ARM_PTR(r2),
                                   (GLenum*)ARM_PTR(r3),ARM_PTR(bin));
        break;
    }
    case SVC_GL3_ProgramBinary:
        if (pfn_glProgramBinary)
            pfn_glProgramBinary((GLuint)r0,(GLenum)r1,ARM_CPTR(r2),(GLsizei)r3);
        break;
    case SVC_GL3_FenceSync: {
        void *sync = pfn_glFenceSync
            ? pfn_glFenceSync((GLenum)r0,(GLbitfield)r1) : nullptr;
        g_gl_syncs.push_back(sync);
        ret32((uint32_t)g_gl_syncs.size());   /* handle = index + 1 */
        break;
    }
    case SVC_GL3_ClientWaitSync: {
        /* (sync, flags, timeout64 in r2:r3) */
        void *sync = (r0 && r0 <= g_gl_syncs.size()) ? g_gl_syncs[r0-1] : nullptr;
        uint64_t timeout = (uint64_t)r2 | ((uint64_t)r3 << 32);
        uint32_t rc = 0x911Au; /* GL_ALREADY_SIGNALED */
        if (sync && pfn_glClientWaitSync)
            rc = (uint32_t)pfn_glClientWaitSync(sync,(GLbitfield)r1,timeout);
        ret32(rc);
        break;
    }
    case SVC_GL3_DeleteSync:
        if (r0 && r0 <= g_gl_syncs.size() && g_gl_syncs[r0-1]) {
            if (pfn_glDeleteSync) pfn_glDeleteSync(g_gl_syncs[r0-1]);
            g_gl_syncs[r0-1] = nullptr;
        }
        break;
    case SVC_GL3_InvalidateFramebuffer:
        if (pfn_glInvalidateFramebuffer)
            pfn_glInvalidateFramebuffer((GLenum)r0,(GLsizei)r1,
                                        (const GLenum*)ARM_CPTR(r2));
        break;
    case SVC_GL3_DetachShader:
        if (pfn_glDetachShader) pfn_glDetachShader((GLuint)r0,(GLuint)r1);
        break;
    case SVC_GL3_DrawBuffers:
        if (pfn_glDrawBuffers)
            pfn_glDrawBuffers((GLsizei)r0,(const GLenum*)ARM_CPTR(r1));
        break;
#undef ARM_PTR
#undef ARM_CPTR

    case SVC_SIGACTION: {
        /* sigaction(signum=r0, new_act=r1, old_act=r2)
 * ARM32 struct sigaction: { sa_handler(4), sa_flags(4), sa_restorer(4), sa_mask[2](8) }
 * Record handler VA so pthread_kill can simulate GC signal delivery. */
        int signum = (int)r0;
        uint32_t new_act_va = r1;
        if (new_act_va) {
            uint32_t sa_handler = ctx.mem.read32(new_act_va);  /* offset 0 = sa_handler */
            if (sa_handler > 1u /* not SIG_DFL(0) or SIG_IGN(1) */) {
                g_sighandlers[signum] = sa_handler;
                static int sa_log = 0;
                if (sa_log++ < 20)
                    fprintf(stderr, "[sigaction] sig=%d handler=0x%08x\n", signum, sa_handler);
            }
        }
        ret32(0); break;
    }

    case SVC_BSD_SIGNAL: {
        /* bsd_signal(signum=r0, handler=r1) → returns old handler (or 0) */
        int signum = (int)r0;
        uint32_t handler_va = r1;
        uint32_t old_handler = 0;
        auto it = g_sighandlers.find(signum);
        if (it != g_sighandlers.end()) old_handler = it->second;
        if (handler_va > 1u /* not SIG_DFL/SIG_IGN */) {
            g_sighandlers[signum] = handler_va;
            static int bsd_log = 0;
            if (bsd_log++ < 20)
                fprintf(stderr, "[bsd_signal] sig=%d handler=0x%08x\n", signum, handler_va);
        }
        ret32(old_handler); break;
    }

    case SVC_PTHREAD_KILL: {
        /* pthread_kill(thread_id=r0, sig=r1) / kill(pid=r0, sig=r1)
 * Boehm GC stop-the-world: GC thread sends SIGPWR(30) to each guest thread.
 * libmono.so registers 0x202e1429 (Mono signal dispatch fn) via sigaction.
 * That fn looks up the real handler (GC_suspend_handler) from an internal
 * BSS table and calls it.  GC_suspend_handler posts GC_ack_sem then calls
 * sem_wait(GC_resume_sem).  Since we're inside call_guest_cb (g_in_cb=true),
 * SVC_SEM_WAIT returns immediately so the handler completes without blocking. */
        uint32_t target_tid = r0;
        int      sig        = (int)r1;
        static int pkill_log = 0;
        if (pkill_log++ < 20)
            fprintf(stderr, "[pthread_kill] target=0x%08x sig=%d lr=0x%08x\n",
                    target_tid, sig, regs[14]);

        uint32_t handler_va = 0;
        auto it = g_sighandlers.find(sig);
        if (it != g_sighandlers.end() && it->second > 1u)
            handler_va = it->second;

        if (handler_va > 1u) {
            if (pkill_log <= 5)
                fprintf(stderr, "[pthread_kill] calling handler=0x%08x for sig=%d\n",
                        handler_va, sig);
            /* Pretend we are the target thread so pthread_self() inside the
 * signal handler (GC_suspend_handler) returns the correct TID.
 * GC_lookup_thread(tid) then finds the right per-thread GC struct. */
            uint32_t saved_tid = g_current_tid;
            g_current_tid = target_tid;
            (void)call_guest_cb(ctx, handler_va, (uint32_t)sig, 0);
            g_current_tid = saved_tid;
        } else if (pkill_log <= 5)
            fprintf(stderr, "[pthread_kill] no handler for sig=%d, skipping\n", sig);
        ret32(0); break;
    }
    case SVC_SYSCALL: {
        /* Generic syscall() shim. The only syscall that must behave correctly
 * for mono's runtime startup is futex (__NR_futex == 240 on ARM):
 * mono drains a coop event/semaphore with
 * futex(uaddr, FUTEX_WAIT, val, timeout)
 * on the main thread, expecting a worker thread (spawned via
 * pthread_create) to store a non-zero word and FUTEX_WAKE it.  The old
 * no-op stub returned 0 ("woken") immediately *without scheduling the
 * worker threads*, so `while (*uaddr == val) futex_wait(...)` spun
 * forever and mono_jit_init never completed — the engine never reached
 * nativeRender, hence the black screen.  Here FUTEX_WAIT runs the
 * cooperative worker threads so they can change *uaddr, honours EAGAIN
 * when the word already changed, and only reports ETIMEDOUT when no
 * progress is possible (so the caller re-evaluates its predicate
 * instead of treating 0 as a spurious wake). */
        switch (r0) {
        case 240: { /* __NR_futex(uaddr=r1, op=r2, val=r3, timeout=[sp]) */
            uint32_t cmd = r2 & 0x7fu; /* strip FUTEX_PRIVATE_FLAG / CLOCK_REALTIME */
            if (cmd != 0u && cmd != 9u) { /* FUTEX_WAKE / FUTEX_REQUEUE etc. */
                /* Record pending wake tokens so next futex_wait on this addr returns 0 */
                uint32_t nwake = (r2 == 0u) ? 1u : std::min(r2, 256u);
                g_futex_wake_tokens[r1] += nwake;
                /* Remove address from waiters map (threads are being woken) */
                auto wa = g_futex_wait_addrs.find(r1);
                if (wa != g_futex_wait_addrs.end()) {
                    uint32_t woke = std::min(nwake, wa->second);
                    wa->second -= woke;
                    if (wa->second == 0u) g_futex_wait_addrs.erase(wa);
                }
                if (getenv("LUNARIA_TRACE_FUTEX")) {
                    static uint64_t wk = 0;
                    if (wk < 256)
                        fprintf(stderr, "[futex] WAKE cmd=%u uaddr=0x%x word=0x%x nwake=%u tid=%u tokens=%u (#%llu)\n",
                                cmd, r1, ctx.mem.read32(r1), nwake, g_current_tid,
                                g_futex_wake_tokens[r1], (unsigned long long)wk);
                    ++wk;
                }
                ret32(0);
                break;
            }
            if (ctx.mem.read32(r1) != r3) { ret32(0); break; } /* word already changed */
            /* Check pending wake tokens before blocking */
            {
                auto tok = g_futex_wake_tokens.find(r1);
                if (tok != g_futex_wake_tokens.end() && tok->second > 0u) {
                    --tok->second;
                    if (tok->second == 0u) g_futex_wake_tokens.erase(tok);
                    ret32(0); break;
                }
            }
            bool changed = false;
            if (g_current_tid == 0 && !g_scheduling && !g_threads.empty()) {
                /* Give worker threads up to LUNARIA_FUTEX_SPINS×20M ticks to change
 * the futex word.  Default 500 spins = 10B ticks so that deep
 * Mono init chains (JIT compile → GC init → corlib load) finish
 * before the main thread gives up and returns ETIMEDOUT. */
                int spin_limit = (int)env_ticks("LUNARIA_FUTEX_SPINS", 500);
                for (int spin = 0; spin < spin_limit && !changed; ++spin) {
                    schedule_threads(20'000'000ULL);
                    drive_aaudio_callbacks(ctx);
                drive_java_choreographer(ctx);
                    drive_java_choreographer(ctx);
                    if (ctx.mem.read32(r1) != r3) changed = true;
                    /* re-check wake tokens after scheduling */
                    auto tok = g_futex_wake_tokens.find(r1);
                    if (!changed && tok != g_futex_wake_tokens.end() && tok->second > 0u) {
                        --tok->second;
                        if (tok->second == 0u) g_futex_wake_tokens.erase(tok);
                        changed = true;
                    }
                }
            } else {
                /* non-main worker: park until a matching FUTEX_WAKE token
 * arrives (schedule_threads skips t.waiting_futex).  Returning to a
 * busy re-run made the Baselib semaphore acquire read a phantom
 * "available" state and dispatch a NULL job. */
                g_futex_wait_addrs[r1]++;
                for (auto &t : g_threads) {
                    if (t.id != g_current_tid) continue;
                    t.waiting_futex = r1;
                    t.futex_val     = r3;
                    break;
                }
                g_yield_requested = true;
                ret32(0);
                break;
            }
            if (changed) {
                static uint64_t woken = 0;
                if (getenv("LUNARIA_TRACE_FUTEX") && woken < 64)
                    fprintf(stderr, "[futex] WAIT woken uaddr=0x%x (#%llu)\n",
                            r1, (unsigned long long)woken);
                ++woken;
                ret32(0);
                break;
            }
            {
                static uint64_t timed = 0;
                if (getenv("LUNARIA_TRACE_FUTEX") && timed < 256)
                    fprintf(stderr, "[futex] WAIT timeout uaddr=0x%x val=%u tid=%u (#%llu)\n",
                            r1, r3, g_current_tid, (unsigned long long)timed);
                ++timed;
            }
            uint32_t eva = errno_va(ctx, g_current_tid);
            if (eva) ctx.mem.write32(eva, 110u /* ETIMEDOUT */);
            if (g_current_tid != 0) g_yield_requested = true; /* worker: yield slice */
            ret32(~0u);
            break;
        }
        case 224: /* __NR_gettid */
            ret32(g_current_tid ? g_current_tid : 1u);
            break;
        case 5: { /* __NR_open(path, flags, mode) */
            int fd = guest_open_path(ctx.mem.cstr(r1), (int)r2, (mode_t)r3);
            ret32((uint32_t)fd);
            break;
        }
        case 322: { /* __NR_openat(dfd, path, flags, mode) */
            int fd = guest_open_path(ctx.mem.cstr(r2), (int)r3, (mode_t)regs[3]);
            ret32((uint32_t)fd);
            break;
        }
        default:
            ret32(0);
            break;
        }
        break;
    }
    case SVC_SBRK: {
        /* sbrk(increment): return old brk, advance by increment.
 * sbrk(0) = query current brk.  Used by Boehm GC's GC_scratch_alloc. */
        int32_t incr = (int32_t)r0;
        uint32_t old = g_brk;
        if (incr > 0) {
            if ((uint64_t)g_brk + (uint32_t)incr > BRK_END) { ret32(~0u); break; }
            g_brk += (uint32_t)incr;
        } else if (incr < 0) {
            uint32_t dec = (uint32_t)(-incr);
            g_brk = (g_brk > BRK_BASE + dec) ? g_brk - dec : BRK_BASE;
        }
        ret32(old);
        break;
    }
    case SVC_BRK: {
        /* brk(addr): set brk to addr (or current if addr=0). */
        if (r0 == 0 || r0 < BRK_BASE || r0 > BRK_END) {
            ret32(g_brk); break;
        }
        g_brk = r0;
        ret32(r0);
        break;
    }
    case SVC_ERRNO_ADDR:
        ret32(errno_va(ctx, g_current_tid));
        break;
    case SVC_SYSPROP_GET: {
        /* __system_property_get(name, value) → strlen(value).
         * FMOD reads ro.build.version.sdk to decide whether AAudio (API 26+)
         * is available; an empty reply made it fall back to no audio output,
         * which later crashed AudioManager (NULL output recorder).  Keep the
         * SDK level in sync with jni_stubs.c Build.VERSION.SDK_INT (31) and
         * AConfiguration_getSdkVersion. */
        const char *name = ARM_STR(r0);
        const char *val  = "";
        if (name) {
            static const std::pair<const char *, const char *> props[] = {
                {"ro.build.version.sdk",     "31"},
                {"ro.build.version.release", "12.0"},
                {"ro.product.cpu.abi",       "armeabi-v7a"},
                {"ro.product.cpu.abi2",      ""},
                {"ro.product.cpu.abilist",   "armeabi-v7a,armeabi"},
                {"ro.product.manufacturer",  "Lunaria"},
                {"ro.product.model",         "Lunaria"},
                {"ro.product.board",         "lunaria"},
                {"ro.hardware",              "lunaria"},
            };
            for (auto &p : props)
                if (!strcmp(name, p.first)) { val = p.second; break; }
            fprintf(stderr, "[arm_exec] __system_property_get(\"%s\") → \"%s\"\n",
                    name, val);
        }
        size_t n = strlen(val);
        if (n > 91) n = 91;          /* PROP_VALUE_MAX = 92 incl. NUL */
        if (r1) {
            memcpy(ctx.mem.ptr(r1), val, n);
            ctx.mem.ptr(r1)[n] = '\0';
        }
        ret32((uint32_t)n);
        break;
    }

    /* ---- pthread TLS ---- */

    case SVC_PTHREAD_SELF: ret32(g_current_tid ? g_current_tid : 1u); break;
    case SVC_PTHREAD_KEY_CREATE: {
        uint32_t key = g_tls_next_key++;
        if (r0) ctx.mem.write32(r0, key);
        ret32(0);
        break;
    }
    case SVC_PTHREAD_KEY_DELETE:
        for (auto it = g_tls.begin(); it != g_tls.end();)
            it = (it->first.second == r0) ? g_tls.erase(it) : std::next(it);
        ret32(0);
        break;
    case SVC_PTHREAD_SETSPECIFIC:
        if (r1 == 0 && getenv("LUNARIA_TRACE_TLS"))
            fprintf(stderr, "[arm_exec] pthread_setspecific(key=%u, NULL) lr=0x%08x tid=%u\n",
                    r0, regs[14], g_current_tid);
        g_tls[{g_current_tid, r0}] = r1;
        ret32(0);
        break;
    case SVC_PTHREAD_GETSPECIFIC: {
        auto it = g_tls.find({g_current_tid, r0});
        ret32(it != g_tls.end() ? it->second : 0u);
        break;
    }

    /* ---- zlib passthrough ----
 * ARM32 z_stream layout (offsets): next_in=0 avail_in=4 total_in=8
 * next_out=12 avail_out=16 total_out=20 msg=24 state=28 zalloc=32
 * zfree=36 opaque=40 data_type=44 adler=48 reserved=52 */

    case SVC_Z_INFLATEINIT2: {
        if (!g_zstreams) g_zstreams = new std::map<uint32_t, z_stream*>();
        z_stream *zs = new z_stream();
        memset(zs, 0, sizeof(*zs));
        int rc = inflateInit2(zs, (int)r1);
        if (rc == Z_OK) (*g_zstreams)[r0] = zs;
        else delete zs;
        ret32((uint32_t)rc);
        break;
    }
    case SVC_Z_INFLATE: {
        if (!g_zstreams) { ret32((uint32_t)Z_STREAM_ERROR); break; }
        auto it = g_zstreams->find(r0);
        if (it == g_zstreams->end()) { ret32((uint32_t)Z_STREAM_ERROR); break; }
        z_stream *zs = it->second;
        uint32_t next_in   = ctx.mem.read32(r0 + 0);
        uint32_t avail_in  = ctx.mem.read32(r0 + 4);
        uint32_t next_out  = ctx.mem.read32(r0 + 12);
        uint32_t avail_out = ctx.mem.read32(r0 + 16);
        zs->next_in   = next_in  ? ctx.mem.ptr(next_in)  : nullptr;
        zs->avail_in  = avail_in;
        zs->next_out  = next_out ? ctx.mem.ptr(next_out) : nullptr;
        zs->avail_out = avail_out;
        int rc = inflate(zs, (int)r1);
        uint32_t consumed = avail_in  - zs->avail_in;
        uint32_t produced = avail_out - zs->avail_out;
        ctx.mem.write32(r0 + 0,  next_in + consumed);
        ctx.mem.write32(r0 + 4,  zs->avail_in);
        ctx.mem.write32(r0 + 8,  (uint32_t)zs->total_in);
        ctx.mem.write32(r0 + 12, next_out + produced);
        ctx.mem.write32(r0 + 16, zs->avail_out);
        ctx.mem.write32(r0 + 20, (uint32_t)zs->total_out);
        ctx.mem.write32(r0 + 48, (uint32_t)zs->adler);
        ret32((uint32_t)rc);
        break;
    }
    case SVC_Z_INFLATEEND: {
        int rc = Z_STREAM_ERROR;
        if (g_zstreams) {
            auto it = g_zstreams->find(r0);
            if (it != g_zstreams->end()) {
                rc = inflateEnd(it->second);
                delete it->second;
                g_zstreams->erase(it);
            }
        }
        ret32((uint32_t)rc);
        break;
    }
    case SVC_Z_INFLATERESET: {
        int rc = Z_STREAM_ERROR;
        if (g_zstreams) {
            auto it = g_zstreams->find(r0);
            if (it != g_zstreams->end()) rc = inflateReset(it->second);
        }
        ret32((uint32_t)rc);
        break;
    }
    case SVC_Z_CRC32:
        ret32((uint32_t)crc32(r0, r1 ? ctx.mem.ptr(r1) : nullptr, r2)); break;
    case SVC_Z_ADLER32:
        ret32((uint32_t)adler32(r0, r1 ? ctx.mem.ptr(r1) : nullptr, r2)); break;

    
    case SVC_Z_INFLATEINIT: {
        if (!g_zstreams) g_zstreams = new std::map<uint32_t, z_stream*>();
        z_stream *zs = new z_stream();
        memset(zs, 0, sizeof(*zs));
        
        int rc = inflateInit(zs);
        if (rc == Z_OK) (*g_zstreams)[r0] = zs;
        else { delete zs; }
        ret32((uint32_t)rc);
        break;
    }

    /* deflateInit2_(z,level,method,windowBits,memLevel,strategy,version,stream_size) */
    case SVC_Z_DEFLATEINIT2: {
        if (!g_dstreams) g_dstreams = new std::map<uint32_t, z_stream*>();
        
        int level    = (int)(int32_t)r1;
        int method   = (int)r2;
        int wbits    = (int)(int32_t)r3;
        int memlevel = (int)ctx.mem.read32(regs[13]);
        int strategy = (int)ctx.mem.read32(regs[13] + 4);
        z_stream *zs = new z_stream();
        memset(zs, 0, sizeof(*zs));
        int rc = deflateInit2(zs, level, method, wbits, memlevel, strategy);
        if (rc == Z_OK) (*g_dstreams)[r0] = zs;
        else { delete zs; }
        ret32((uint32_t)rc);
        break;
    }
    case SVC_Z_DEFLATE: {
        if (!g_dstreams) { ret32((uint32_t)Z_STREAM_ERROR); break; }
        auto it = g_dstreams->find(r0);
        if (it == g_dstreams->end()) { ret32((uint32_t)Z_STREAM_ERROR); break; }
        z_stream *zs = it->second;
        uint32_t next_in   = ctx.mem.read32(r0 + 0);
        uint32_t avail_in  = ctx.mem.read32(r0 + 4);
        uint32_t next_out  = ctx.mem.read32(r0 + 12);
        uint32_t avail_out = ctx.mem.read32(r0 + 16);
        zs->next_in   = next_in  ? ctx.mem.ptr(next_in)  : nullptr;
        zs->avail_in  = avail_in;
        zs->next_out  = next_out ? ctx.mem.ptr(next_out) : nullptr;
        zs->avail_out = avail_out;
        int rc = deflate(zs, (int)r1);
        uint32_t consumed = avail_in  - zs->avail_in;
        uint32_t produced = avail_out - zs->avail_out;
        ctx.mem.write32(r0 + 0,  next_in + consumed);
        ctx.mem.write32(r0 + 4,  zs->avail_in);
        ctx.mem.write32(r0 + 8,  (uint32_t)zs->total_in);
        ctx.mem.write32(r0 + 12, next_out + produced);
        ctx.mem.write32(r0 + 16, zs->avail_out);
        ctx.mem.write32(r0 + 20, (uint32_t)zs->total_out);
        ctx.mem.write32(r0 + 48, (uint32_t)zs->adler);
        ret32((uint32_t)rc);
        break;
    }
    case SVC_Z_DEFLATEEND: {
        int rc = Z_STREAM_ERROR;
        if (g_dstreams) {
            auto it = g_dstreams->find(r0);
            if (it != g_dstreams->end()) {
                rc = deflateEnd(it->second);
                delete it->second;
                g_dstreams->erase(it);
            }
        }
        ret32((uint32_t)rc);
        break;
    }
    case SVC_Z_DEFLATERESET: {
        int rc = Z_STREAM_ERROR;
        if (g_dstreams) {
            auto it = g_dstreams->find(r0);
            if (it != g_dstreams->end()) rc = deflateReset(it->second);
        }
        ret32((uint32_t)rc);
        break;
    }

    /* ---- string→number conversions ---- */

    case SVC_ATOI: {
        const char *s = ctx.mem.cstr(r0);
        ret32(s ? (uint32_t)atoi(s) : 0u); break;
    }
    case SVC_ATOL: {
        const char *s = ctx.mem.cstr(r0);
        ret32(s ? (uint32_t)atol(s) : 0u); break;
    }
    case SVC_STRTOL: case SVC_STRTOUL: {
        const char *s = ctx.mem.cstr(r0);
        char *end = nullptr;
        unsigned long v = 0;
        if (s) v = (svc_no == SVC_STRTOL)
            ? (unsigned long)strtol(s, &end, (int)r2)
            : strtoul(s, &end, (int)r2);
        if (r1 && s) ctx.mem.write32(r1, r0 + (uint32_t)(end - s));
        ret32((uint32_t)v);
        break;
    }
    case SVC_STRTOD: {
        const char *s = ctx.mem.cstr(r0);
        char *end = nullptr;
        double v = s ? strtod(s, &end) : 0.0;
        if (r1 && s) ctx.mem.write32(r1, r0 + (uint32_t)(end - s));
        uint64_t bits; memcpy(&bits, &v, 8);
        ret64(bits);
        break;
    }
    case SVC_STRTOF: {
        const char *s = ctx.mem.cstr(r0);
        char *end = nullptr;
        float v = s ? strtof(s, &end) : 0.0f;
        if (r1 && s) ctx.mem.write32(r1, r0 + (uint32_t)(end - s));
        uint32_t bits; memcpy(&bits, &v, 4);
        ret32(bits);
        break;
    }
    case SVC_STRTOLL: case SVC_STRTOULL: {
        const char *s = ctx.mem.cstr(r0);
        char *end = nullptr;
        uint64_t v = 0;
        if (s) v = (svc_no == SVC_STRTOLL)
            ? (uint64_t)strtoll(s, &end, (int)r2)
            : strtoull(s, &end, (int)r2);
        if (r1 && s) ctx.mem.write32(r1, r0 + (uint32_t)(end - s));
        ret64(v);
        break;
    }

    

    case SVC_MEMALIGN: {
        /* memalign: allocate align-boundary block */
        uint32_t align = r0 ? r0 : 8u;
        if (align & (align - 1)) align = 8u; 
        ret32(arm_memalign(ctx, align, r1));
        break;
    }
    case SVC_POSIX_MEMALIGN: {
        /* posix_memalign(void **memptr, align, size) */
        uint32_t align = r1 ? r1 : 8u;
        if (align & (align - 1)) align = 8u;
        uint32_t p = arm_memalign(ctx, align, r2);
        if (r0) ctx.mem.write32(r0, p);
        ret32(p ? 0u : 12u /* ENOMEM */);
        break;
    }

    

    case SVC_MEMCMP:
        ret32((r0 && r1) ? (uint32_t)memcmp(ctx.mem.ptr(r0), ctx.mem.ptr(r1), r2)
                         : 0u);
        break;
    case SVC_MEMCHR: {
        const void *p = r0 ? memchr(ctx.mem.ptr(r0), (int)r1, r2) : nullptr;
        ret32(p ? r0 + (uint32_t)((const uint8_t*)p - ctx.mem.ptr(r0)) : 0u);
        break;
    }
    case SVC_MEMRCHR: {
        const void *p = r0 ? memrchr(ctx.mem.ptr(r0), (int)r1, r2) : nullptr;
        ret32(p ? r0 + (uint32_t)((const uint8_t*)p - ctx.mem.ptr(r0)) : 0u);
        break;
    }
    case SVC_MEMMEM: {
        const void *p = (r0 && r2) ? memmem(ctx.mem.ptr(r0), r1, ctx.mem.ptr(r2), r3)
                                   : nullptr;
        ret32(p ? r0 + (uint32_t)((const uint8_t*)p - ctx.mem.ptr(r0)) : 0u);
        break;
    }
    case SVC_STRCHR: {
        const char *s = ctx.mem.cstr(r0);
        const char *p = s ? strchr(s, (int)r1) : nullptr;
        ret32(p ? r0 + (uint32_t)(p - s) : 0u);
        break;
    }
    case SVC_STRRCHR: {
        const char *s = ctx.mem.cstr(r0);
        const char *p = s ? strrchr(s, (int)r1) : nullptr;
        ret32(p ? r0 + (uint32_t)(p - s) : 0u);
        break;
    }
    case SVC_STRSTR: {
        const char *s = ctx.mem.cstr(r0), *n = ctx.mem.cstr(r1);
        const char *p = (s && n) ? strstr(s, n) : nullptr;
        ret32(p ? r0 + (uint32_t)(p - s) : 0u);
        break;
    }
    case SVC_STRNLEN: {
        const char *s = ctx.mem.cstr(r0);
        ret32(s ? (uint32_t)strnlen(s, r1) : 0u);
        break;
    }
    case SVC_STRCASECMP: {
        const char *a = ctx.mem.cstr(r0), *b = ctx.mem.cstr(r1);
        ret32((a && b) ? (uint32_t)strcasecmp(a, b) : 0u);
        break;
    }
    case SVC_STRCSPN: {
        const char *s = ctx.mem.cstr(r0), *rej = ctx.mem.cstr(r1);
        ret32((s && rej) ? (uint32_t)strcspn(s, rej) : 0u);
        break;
    }
    case SVC_STRSPN: {
        const char *s = ctx.mem.cstr(r0), *acc = ctx.mem.cstr(r1);
        ret32((s && acc) ? (uint32_t)strspn(s, acc) : 0u);
        break;
    }
    case SVC_STRTOK_R: {
        
        uint32_t s_va = r0 ? r0 : (r2 ? ctx.mem.read32(r2) : 0u);
        const char *delim = ctx.mem.cstr(r1);
        if (!s_va || !delim) { ret32(0); break; }
        char *base = (char*)ctx.mem.ptr(0);
        char *s = base + s_va;
        s += strspn(s, delim);
        if (!*s) {
            if (r2) ctx.mem.write32(r2, (uint32_t)(s - base));
            ret32(0); break;
        }
        char *tok = s;
        s += strcspn(s, delim);
        if (*s) { *s = '\0'; ++s; }
        if (r2) ctx.mem.write32(r2, (uint32_t)(s - base));
        ret32((uint32_t)(tok - base));
        break;
    }
    case SVC_BASENAME: {
        const char *s = ctx.mem.cstr(r0);
        if (!s || !*s) { ret32(r0); break; }
        const char *p = strrchr(s, '/');
        ret32(p ? r0 + (uint32_t)(p + 1 - s) : r0);
        break;
    }
    case SVC_ISSPACE:
        ret32(isspace((int)(r0 & 0xff)) ? 1u : 0u); break;
    case SVC_WCSLEN: {
        /* ARM32 wchar_t = u32 */
        uint32_t n = 0;
        if (r0) while (ctx.mem.read32(r0 + n*4)) ++n;
        ret32(n);
        break;
    }
    case SVC_WMEMCPY: case SVC_WMEMMOVE:
        if (r0 && r1 && r2) {
            /* Diagnose wild args before the host memmove faults: a huge count or
 * a pointer outside the mapped guest range means the caller passed
 * garbage (upstream emulation bug) — log it instead of crashing. */
            uint64_t bytes = (uint64_t)r2 * 4;
            if (bytes > 0x10000000ull || r0 + bytes > 0xfffff000ull ||
                r1 + bytes > 0xfffff000ull) {
                fprintf(stderr, "[wmemcpy] WILD args: dst=0x%08x src=0x%08x n=%u "
                        "lr=0x%08x tid=%u — skipping\n",
                        r0, r1, r2, regs[14], g_current_tid);
                ret32(r0); break;
            }
            memmove(ctx.mem.ptr(r0), ctx.mem.ptr(r1), (size_t)bytes);
        }
        ret32(r0); break;
    case SVC_WMEMSET: {
        for (uint32_t i = 0; i < r2; ++i) ctx.mem.write32(r0 + i*4, r1);
        ret32(r0); break;
    }

    

    case SVC_SNPRINTF: {
        ArmVarArgs ap{ctx, regs, 3, regs[13], false};
        ret32(arm_format_to(ctx, r0, r1, arm_vformat(ctx, ctx.mem.cstr(r2), ap)));
        break;
    }
    case SVC_SPRINTF: {
        ArmVarArgs ap{ctx, regs, 2, regs[13], false};
        ret32(arm_format_to(ctx, r0, UINT32_MAX, arm_vformat(ctx, ctx.mem.cstr(r1), ap)));
        break;
    }
    case SVC_VSNPRINTF: {
        ArmVarArgs ap{ctx, regs, 0, r3, true};
        ret32(arm_format_to(ctx, r0, r1, arm_vformat(ctx, ctx.mem.cstr(r2), ap)));
        break;
    }
    case SVC_VASPRINTF: {
        /* vasprintf(char **strp, fmt, va_list) */
        ArmVarArgs ap{ctx, regs, 0, r2, true};
        std::string s = arm_vformat(ctx, ctx.mem.cstr(r1), ap);
        uint32_t p = arm_malloc(ctx, (uint32_t)s.size() + 1);
        if (p) { memcpy(ctx.mem.ptr(p), s.c_str(), s.size() + 1); }
        if (r0) ctx.mem.write32(r0, p);
        ret32(p ? (uint32_t)s.size() : ~0u);
        break;
    }
    case SVC_PRINTF: {
        ArmVarArgs ap{ctx, regs, 1, regs[13], false};
        std::string s = arm_vformat(ctx, ctx.mem.cstr(r0), ap);
        fwrite(s.data(), 1, s.size(), stdout);
        ret32((uint32_t)s.size());
        break;
    }
    case SVC_FPRINTF: {
        
        ArmVarArgs ap{ctx, regs, 2, regs[13], false};
        std::string s = arm_vformat(ctx, ctx.mem.cstr(r1), ap);
        fwrite(s.data(), 1, s.size(), stderr);
        ret32((uint32_t)s.size());
        break;
    }
    case SVC_VPRINTF: {
        ArmVarArgs ap{ctx, regs, 0, r1, true};
        std::string s = arm_vformat(ctx, ctx.mem.cstr(r0), ap);
        fwrite(s.data(), 1, s.size(), stdout);
        ret32((uint32_t)s.size());
        break;
    }
    case SVC_VFPRINTF: {
        ArmVarArgs ap{ctx, regs, 0, r2, true};
        std::string s = arm_vformat(ctx, ctx.mem.cstr(r1), ap);
        fwrite(s.data(), 1, s.size(), stderr);
        ret32((uint32_t)s.size());
        break;
    }
    case SVC_PUTS: {
        const char *s = ctx.mem.cstr(r0);
        if (s) { fputs(s, stdout); fputc('\n', stdout); }
        ret32(0); break;
    }
    case SVC_FPUTS: {
        const char *s = ctx.mem.cstr(r0);
        if (s) fputs(s, stderr);
        ret32(0); break;
    }
    case SVC_FPUTC:
        fputc((int)r0, stderr); ret32(r0); break;
    case SVC_ASSERT2: {
        const char *file = ctx.mem.cstr(r0), *fn = ctx.mem.cstr(r2),
                   *msg = ctx.mem.cstr(r3);
        fprintf(stderr, "[arm_exec] __assert2: %s:%u %s: %s\n",
                file ? file : "?", r1, fn ? fn : "?", msg ? msg : "?");
        ret32(0); break;
    }
    case SVC_LOG_VPRINT: {
        const char *tag = ctx.mem.cstr(r1);
        ArmVarArgs ap{ctx, regs, 0, r3, true};
        std::string s = arm_vformat(ctx, ctx.mem.cstr(r2), ap);
        fprintf(stderr, "[android_log p=%u t=%s lr=0x%08x] %s\n",
                r0, tag?tag:"?", regs[14], s.c_str());
        ret32(0); break;
    }

    

    case SVC_SINCOS: {
        uint64_t in = (uint64_t)r0 | ((uint64_t)r1 << 32);
        double a; memcpy(&a, &in, 8);
        double sv, cv; sincos(a, &sv, &cv);
        if (r2) { uint64_t b; memcpy(&b, &sv, 8);
                  ctx.mem.write32(r2, (uint32_t)b); ctx.mem.write32(r2+4, (uint32_t)(b>>32)); }
        if (r3) { uint64_t b; memcpy(&b, &cv, 8);
                  ctx.mem.write32(r3, (uint32_t)b); ctx.mem.write32(r3+4, (uint32_t)(b>>32)); }
        break;
    }
    case SVC_SINCOSF: {
        float a = rf(r0);
        float sv, cv; sincosf(a, &sv, &cv);
        uint32_t b;
        if (r1) { memcpy(&b, &sv, 4); ctx.mem.write32(r1, b); }
        if (r2) { memcpy(&b, &cv, 4); ctx.mem.write32(r2, b); }
        break;
    }
    case SVC_LDEXP: {
        uint64_t in = (uint64_t)r0 | ((uint64_t)r1 << 32);
        double a; memcpy(&a, &in, 8);
        double v = ldexp(a, (int)r2);
        uint64_t bits; memcpy(&bits, &v, 8); ret64(bits);
        break;
    }
    case SVC_LDEXPF: {
        float v = ldexpf(rf(r0), (int)r1);
        uint32_t bits; memcpy(&bits, &v, 4); ret32(bits);
        break;
    }
    case SVC_MODF: {
        uint64_t in = (uint64_t)r0 | ((uint64_t)r1 << 32);
        double a; memcpy(&a, &in, 8);
        double ip; double v = modf(a, &ip);
        if (r2) { uint64_t b; memcpy(&b, &ip, 8);
                  ctx.mem.write32(r2, (uint32_t)b); ctx.mem.write32(r2+4, (uint32_t)(b>>32)); }
        uint64_t bits; memcpy(&bits, &v, 8); ret64(bits);
        break;
    }
    case SVC_MODFF: {
        float ip; float v = modff(rf(r0), &ip);
        if (r1) { uint32_t b; memcpy(&b, &ip, 4); ctx.mem.write32(r1, b); }
        uint32_t bits; memcpy(&bits, &v, 4); ret32(bits);
        break;
    }

    

    case SVC_ACCESS: {
        const char *path = ctx.mem.cstr(r0);
        
        if (path && strncmp(path, "file://", 7) == 0) {
            const char *p = path + 7;
            if (*p == '/') {
                path = p;               /* "file:///abs/path" → "/abs/path" */
            } else {
                const char *sl = strchr(p, '/');
                if (sl) path = sl;      /* "file://host/path" → "/path" */
            }
        }
        if (getenv("LUNARIA_TRACE_OPEN"))
            fprintf(stderr, "[access] %s mode=%d -> %d\n", path ? path : "(null)", (int)r1,
                    path ? access(path, (int)r1) : -1);
        if (path && strstr(path, "Managed") && strstr(path, ".dll"))
            fprintf(stderr, "[access_dll] %s mode=%d -> %d lr=0x%08x\n", path, (int)r1,
                    access(path,(int)r1), regs[14]);
        if (path && access(path, (int)r1) == 0) { ret32(0); break; }
        char cache[PATH_MAX];
        if (path && apk_extract_to_cache(path, cache, sizeof cache) &&
            access(cache, (int)r1) == 0) { ret32(0); break; }
        ret32(~0u);
        break;
    }
    case SVC_REALPATH: {
        const char *path = ctx.mem.cstr(r0);
        char buf[PATH_MAX];
        if (path && realpath(path, buf)) {
            uint32_t dst = r1;
            if (!dst) dst = arm_malloc(ctx, (uint32_t)strlen(buf) + 1);
            if (dst) { strcpy((char*)ctx.mem.ptr(dst), buf); ret32(dst); break; }
        }
        ret32(0);
        break;
    }
    case SVC_PREAD: {
        /* bionic LP32: pread(fd, buf, count, off_t(32bit) r3) */
        ssize_t n = (r1) ? pread((int)r0, ctx.mem.ptr(r1), (size_t)r2, (off_t)(int32_t)r3)
                         : -1;
        ret32((uint32_t)n);
        break;
    }
    case SVC_PWRITE: {
        ssize_t n = (r1) ? pwrite((int)r0, ctx.mem.ptr(r1), (size_t)r2, (off_t)(int32_t)r3)
                         : -1;
        ret32((uint32_t)n);
        break;
    }
    case SVC_FGETS: {
        FILE *f = resolve_file(ctx, r2);
        if (!f || !r0 || r1 == 0) { ret32(0); break; }
        char *p = fgets((char*)ctx.mem.ptr(r0), (int)r1, f);
        ret32(p ? r0 : 0u);
        break;
    }
    case SVC_FILENO: {
        FILE *f = resolve_file(ctx, r0);
        ret32(f ? (uint32_t)fileno(f) : ~0u);
        break;
    }
    case SVC_FEOF: {
        FILE *f = resolve_file(ctx, r0);
        ret32(f ? (uint32_t)feof(f) : 1u);
        break;
    }
    case SVC_OPENDIR: {
        const char *path = ctx.mem.cstr(r0);
        DIR *d = path ? opendir(path) : nullptr;
        if (d) {
            uint32_t idx = g_dir_next++;
            g_dir_tab[idx] = {d, arm_malloc(ctx, 19 + 256), {}, 0};
            ret32(idx);
            break;
        }
        /* Fall back to enumerating an APK directory (App Bundle base/ prefix). */
        std::vector<std::pair<std::string,bool>> entries;
        if (path && apk_path_list_dir(path, entries)) {
            uint32_t idx = g_dir_next++;
            g_dir_tab[idx] = {nullptr, arm_malloc(ctx, 19 + 256), std::move(entries), 0};
            ret32(idx);
            break;
        }
        ret32(0);
        break;
    }
    case SVC_READDIR: {
        auto it = g_dir_tab.find(r0);
        if (it == g_dir_tab.end()) { ret32(0); break; }
        uint32_t va = it->second.dirent_va;
        /* bionic LP32 dirent: ino u64@0, off s64@8, reclen u16@16, type u8@18, name@19 */
        if (it->second.d) {
            struct dirent *e = readdir(it->second.d);
            if (!e) { ret32(0); break; }
            memset(ctx.mem.ptr(va), 0, 19 + 256);
            ctx.mem.write32(va, (uint32_t)e->d_ino);
            *ctx.mem.ptr(va + 18) = e->d_type;
            strncpy((char*)ctx.mem.ptr(va + 19), e->d_name, 255);
            ret32(va);
            break;
        }
        /* synthetic APK directory */
        if (it->second.apk_idx >= it->second.apk.size()) { ret32(0); break; }
        auto &ent = it->second.apk[it->second.apk_idx++];
        memset(ctx.mem.ptr(va), 0, 19 + 256);
        ctx.mem.write32(va, (uint32_t)(it->second.apk_idx)); /* synthetic inode */
        *ctx.mem.ptr(va + 18) = ent.second ? (uint8_t)DT_DIR : (uint8_t)DT_REG;
        strncpy((char*)ctx.mem.ptr(va + 19), ent.first.c_str(), 255);
        ret32(va);
        break;
    }
    case SVC_CLOSEDIR: {
        auto it = g_dir_tab.find(r0);
        if (it != g_dir_tab.end()) {
            if (it->second.d) closedir(it->second.d);
            g_dir_tab.erase(it);
        }
        ret32(0);
        break;
    }
    case SVC_PTHREAD_EQUAL:
        ret32(r0 == r1 ? 1u : 0u); break;
    case SVC_SETJMP: {
        
        if (r0) {
            ctx.mem.write32(r0, 0x4A4D5031u); /* magic "JMP1" */
            for (int i = 0; i < 8; ++i)
                ctx.mem.write32(r0 + 4 + (uint32_t)i*4, regs[4 + i]);
            ctx.mem.write32(r0 + 36, regs[13]);
            ctx.mem.write32(r0 + 40, regs[14]);
        }
        ret32(0);
        break;
    }
    case SVC_LONGJMP: {
        
        if (r0 && ctx.mem.read32(r0) == 0x4A4D5031u) {
            uint32_t caller_lr = regs[14];
            for (int i = 0; i < 8; ++i)
                regs[4 + i] = ctx.mem.read32(r0 + 4 + (uint32_t)i*4);
            regs[13] = ctx.mem.read32(r0 + 36);
            regs[14] = ctx.mem.read32(r0 + 40);
            if (getenv("LUNARIA_TRACE_LONGJMP"))
                fprintf(stderr, "[longjmp] env=0x%08x val=%u → restored lr=0x%08x sp=0x%08x "
                        "(called from lr=0x%08x)\n",
                        r0, r1?r1:1u, regs[14], regs[13], caller_lr);
            ret32(r1 ? r1 : 1u); 
        } else {
            fprintf(stderr, "[arm_exec] longjmp with invalid jmp_buf 0x%08x — "
                    "halting run\n", r0);
            regs[14] = SENTINEL_ADDR;
            regs[15] = SENTINEL_ADDR;
            ret32(0);
        }
        break;
    }
    case SVC_EXIT: {
        
        static int exit_count = 0;
        fprintf(stderr, "[arm_exec] guest exit(%d) — halting run (lr=0x%08x pc=0x%08x)\n",
                (int)r0, regs[14], regs[15]);
        if (exit_count++ < 1)  
            svc_ring_dump();
        regs[14] = SENTINEL_ADDR;
        regs[15] = SENTINEL_ADDR;
        ret32(0);
        break;
    }

    /* pthread mutex (lightweight counter impl) -------------------------------- */
    case SVC_PTHREAD_MUTEX_INIT:
        /* pthread_mutex_init: zero the guest mutex word */
        if (r0) ctx.mem.write32(r0, 0u);
        ret32(0); break;
    case SVC_PTHREAD_MUTEX_LOCK:
        
        if (r0) {
            uint32_t w = ctx.mem.read32(r0);
            ctx.mem.write32(r0, ((w & ~0xffu) + 0x100u) | (g_current_tid + 1u));
        }
        ret32(0); break;
    case SVC_PTHREAD_MUTEX_TRYLOCK:
        
        if (r0) {
            uint32_t w = ctx.mem.read32(r0);
            if (w >= 0x100u && (w & 0xffu) != g_current_tid + 1u) {
                ret32(16u /* EBUSY */); break;
            }
            ctx.mem.write32(r0, ((w & ~0xffu) + 0x100u) | (g_current_tid + 1u));
        }
        ret32(0); break;
    case SVC_PTHREAD_MUTEX_UNLOCK:
        if (r0) {
            uint32_t w = ctx.mem.read32(r0);
            w = (w >= 0x100u) ? w - 0x100u : 0u;
            if (w < 0x100u) w = 0u;   
            ctx.mem.write32(r0, w);
        }
        ret32(0); break;
    case SVC_PTHREAD_MUTEX_DESTROY:
        if (r0) ctx.mem.write32(r0, 0u);
        ret32(0); break;

    /* ---- pthread cond --------------------------------------------------- */
    case SVC_PTHREAD_COND_INIT:
        if (r0) { g_conds[r0] = 0; ctx.mem.write32(r0, 0u); }
        ret32(0); break;
    case SVC_PTHREAD_COND_SIGNAL: {
        static uint32_t cs_count = 0;
        /* always log 0x46ffed34 to track whether it ever gets signaled */
        if (cs_count++ < 40 || r0 == 0x46ffed34u || getenv("LUNARIA_TRACE_COND"))
            fprintf(stderr, "[cond] signal cond=0x%08x tid=%u lr=0x%08x\n",
                    r0, g_current_tid, regs[14]);
        if (r0) { auto &c = g_conds[r0]; if (c < 1) c = 1; }
        ret32(0); break;
    }
    case SVC_PTHREAD_COND_BROADCAST: {
        static uint32_t cb_count = 0;
        /* always log 0x46ffed34 to track whether it ever gets broadcast */
        if (cb_count++ < 40 || r0 == 0x46ffed34u || getenv("LUNARIA_TRACE_COND"))
            fprintf(stderr, "[cond] broadcast cond=0x%08x tid=%u lr=0x%08x\n",
                    r0, g_current_tid, regs[14]);
        /* Wake every *current* waiter — bounded by the thread count.  A
         * permanent token (former UINT32_MAX) breaks per-frame waits: after
         * UnityChoreographer's doFrame broadcast the vsync cond_wait then
         * returned instantly forever, so the main thread busy-looped without
         * ever re-entering the spin path that delivers the next doFrame.
         * pthreads semantics don't latch broadcasts either — correctly
         * written guests re-check their predicate before waiting. */
        if (r0) g_conds[r0] = (uint32_t)g_threads.size() + 1u;
        ret32(0); break;
    }
    case SVC_PTHREAD_COND_DESTROY:
        if (r0) g_conds.erase(r0);
        ret32(0); break;

    /* ---- pthread_exit ---------------------------------------------------- */
    case SVC_PTHREAD_EXIT:
        
        if (g_current_tid != 0) {
            
            for (auto &t : g_threads)
                if (t.id == g_current_tid) { t.finished = true; break; }
            g_yield_requested = true;
        } else {
            regs[14] = SENTINEL_ADDR;
            regs[15] = SENTINEL_ADDR;
        }
        ret32(0); break;

    /* ---- pthread attr / mutexattr / condattr no-op ----------------------- */
    case SVC_PTHREAD_ATTR_NOOP:
    case SVC_PTHREAD_MUTEXATTR_NOOP:
    case SVC_PTHREAD_CONDATTR_NOOP:
        ret32(0); break;

    /* ---- pthread_detach -------------------------------------------------- */
    case SVC_PTHREAD_DETACH:
        if (r0) g_detached_threads.insert(r0);
        ret32(0); break;

    /* ---- pthread_join ----------------------------------------------------- */
    case SVC_PTHREAD_JOIN: {
        
        uint32_t target_tid = r0;
        uint32_t retval_ptr = r1;
        bool found = false;
        int spin_limit = (int)env_ticks("LUNARIA_JOIN_SPINS", 2000);
        for (int spin = 0; spin < spin_limit; ++spin) {
            found = false;
            bool finished = false;
            for (auto &t : g_threads) {
                if (t.id == target_tid) { found = true; finished = t.finished; break; }
            }
            if (!found || finished) break; 
            schedule_threads(env_ticks("LUNARIA_THREAD_TICKS", 200'000'000ULL));
        }
        if (retval_ptr) ctx.mem.write32(retval_ptr, 0u);
        
        g_threads.erase(std::remove_if(g_threads.begin(), g_threads.end(),
            [&](const ArmThread &t){ return t.id == target_tid && t.finished; }),
            g_threads.end());
        ret32(0); break;
    }

    /* ---- pthread_cond_wait / timedwait ----------------------------------- */
    case SVC_PTHREAD_COND_WAIT: {

        uint32_t cond  = r0;
        uint32_t mutex = r1;
        static uint32_t cw_count = 0;
        /* Log the first few waits per (cond,tid): workers retry their wait
         * every slice, so unconditional logging floods the output. */
        static std::map<uint64_t, uint32_t> cw_per_site;
        uint32_t &site_n = cw_per_site[((uint64_t)g_current_tid << 32) | cond];
        if (cw_count++ < 20 || site_n++ < 3 ||
                cond == 0x46ffed34u || cond == 0x02ec7184u ||
                getenv("LUNARIA_TRACE_COND"))
            fprintf(stderr, "[cond] wait cond=0x%08x mutex=0x%08x tid=%u lr=0x%08x g_conds[cond]=%d\n",
                    cond, mutex, g_current_tid, regs[14],
                    (cond && g_conds.count(cond)) ? (int)g_conds[cond] : -1);
        if (cond) {
            auto it = g_conds.find(cond);
            if (it != g_conds.end() && it->second > 0) {

                if (it->second != UINT32_MAX) --it->second;
                ret32(0); break;
            }
        }

        uint32_t saved_w = mutex ? ctx.mem.read32(mutex) : 0u;
        if (mutex) ctx.mem.write32(mutex, 0u);
        /* Main thread: spin schedule_threads (like futex_wait) until cond is signaled.
         * A single pass is not enough when workers need multiple iterations to complete
         * their initialization task and call cond_signal. */
        {
            static uint32_t cwpre = 0;
            if (cwpre++ < 10)
                fprintf(stderr, "[cond_wait_pre] cond=0x%08x tid=%u sched=%d threads=%zu cond_val=%d\n",
                        cond, g_current_tid, (int)g_scheduling, g_threads.size(),
                        (cond && g_conds.count(cond)) ? (int)g_conds[cond] : -1);
        }
        if (g_current_tid == 0 && !g_scheduling && !g_threads.empty()) {
            /* Spin schedule_threads until cond is signaled by a worker.
             * Use LUNARIA_THREAD_TICKS (default 200M) per pass so workers get
             * enough time to complete their startup/task and call cond_signal. */
            int spin_limit = (int)env_ticks("LUNARIA_FUTEX_SPINS", 500);
            for (int spin = 0; spin < spin_limit; ++spin) {
                /* NOTE: do NOT force-wake parked futex waiters here.  An earlier
                 * version forged the futex word to 1 and injected wake tokens for
                 * every parked waiter, which made Baselib semaphore acquires
                 * succeed with no job posted — workers then dispatched a NULL
                 * job fn and died (blx 0).  Parked workers are woken by the
                 * guest's own FUTEX_WAKE (the poster writes fn/arg first), so
                 * plain scheduling is both sufficient and correct. */
                schedule_threads(env_ticks("LUNARIA_THREAD_TICKS", 200'000'000ULL));
                /* FMOD mixer/async work is callback-driven; pump it so workers
                 * we're waiting on (nonblocking opens etc.) can finish. */
                drive_aaudio_callbacks(ctx);
                drive_java_choreographer(ctx);
                if (cond && g_conds.count(cond) && g_conds[cond] > 0) break;
                /* Diagnose why workers aren't signaling: dump futex waiter/token state */
                if (spin < 3 || (spin % 50 == 0)) {
                    size_t active_t = 0;
                    for (auto &t2 : g_threads) if (!t2.finished) ++active_t;
                    fprintf(stderr, "[cond_spin] spin=%d cond=0x%08x cond_val=%d "
                            "futex_waiters=%zu futex_tokens=%zu threads=%zu(active=%zu) lr=0x%08x\n",
                            spin, cond,
                            (cond && g_conds.count(cond)) ? (int)g_conds[cond] : -1,
                            g_futex_wait_addrs.size(),
                            g_futex_wake_tokens.size(),
                            g_threads.size(), active_t, regs[14]);
                    if (spin < 2) {
                        /* main's call chain: scan the stack for Thumb return addrs */
                        static int bt_once = 0;
                        if (bt_once++ < 2) {
                            for (uint32_t o = 0; o < 0x300 && ctx.mem.ptr(regs[13] + o); o += 4) {
                                uint32_t w = ctx.mem.read32(regs[13] + o);
                                if ((w & 1) && w >= 0x000a0000u && w < 0x03400000u)
                                    fprintf(stderr, "  [main_bt] [sp+0x%03x]=0x%08x\n", o, w);
                            }
                            /* The 0x26f10b0 waiter loop polls *[this+0x70] with
                             * this = cond - 0xc; dump the object so we can see
                             * the flag pointer and its target. */
                            uint32_t self = cond - 0xcu;
                            if (ctx.mem.ptr(self) && ctx.mem.ptr(self + 0x7c)) {
                                for (uint32_t o = 0; o <= 0x78u; o += 4)
                                    fprintf(stderr, "  [waitobj] [this+0x%02x]=0x%08x\n",
                                            o, ctx.mem.read32(self + o));
                                uint32_t fp = ctx.mem.read32(self + 0x70u);
                                if (fp && ctx.mem.ptr(fp))
                                    fprintf(stderr, "  [waitobj] *flagptr(0x%08x)=0x%08x\n",
                                            fp, ctx.mem.read32(fp));
                                /* second ctor arg (bss global) and the queue-ish
                                 * subobjects it hangs off — what was the job
                                 * submitted to? */
                                uint32_t g = ctx.mem.read32(0x02e0f7b0u);
                                fprintf(stderr, "  [waitobj] g(0x02e0f7b0)=0x%08x\n", g);
                                /* jobject caches the ctor resolves via 0x2d1d104 */
                                fprintf(stderr, "  [waitobj] jcache1(0x02e0ae24)=0x%08x "
                                        "jcache2(0x02e0ae56)=0x%08x\n",
                                        ctx.mem.read32(0x02e0ae24u),
                                        ctx.mem.read32(0x02e0ae56u));
                                uint32_t jc = ctx.mem.read32(0x02e0ae24u);
                                if (jc && ctx.mem.ptr(jc))
                                    fprintf(stderr, "  [waitobj] jcache1→ 0x%08x 0x%08x 0x%08x 0x%08x\n",
                                            ctx.mem.read32(jc), ctx.mem.read32(jc+4),
                                            ctx.mem.read32(jc+8), ctx.mem.read32(jc+12));
                                fprintf(stderr, "  [waitobj] jnifn(0x02e85d78)=0x%08x\n",
                                        ctx.mem.read32(0x02e85d78u));
                                if (g && ctx.mem.ptr(g))
                                    for (uint32_t o = 0; o <= 0x10u; o += 4)
                                        fprintf(stderr, "  [waitobj]   g[+0x%02x]=0x%08x\n",
                                                o, ctx.mem.read32(g + o));
                                for (uint32_t fld : {0x1cu, 0x6cu, 0x74u}) {
                                    uint32_t p = ctx.mem.read32(self + fld);
                                    if (p && ctx.mem.ptr(p))
                                        fprintf(stderr, "  [waitobj] [this+0x%02x]→ 0x%08x 0x%08x 0x%08x 0x%08x\n",
                                                fld, ctx.mem.read32(p), ctx.mem.read32(p+4),
                                                ctx.mem.read32(p+8), ctx.mem.read32(p+12));
                                }
                            }
                        }
                        for (auto& kv : g_futex_wait_addrs)
                            fprintf(stderr, "  [futex_waiter] uaddr=0x%08x count=%u word=0x%08x\n",
                                    kv.first, kv.second, ctx.mem.read32(kv.first));
                        for (auto& kv : g_futex_wake_tokens)
                            fprintf(stderr, "  [futex_token]  uaddr=0x%08x tokens=%u\n",
                                    kv.first, kv.second);
                        /* dump last worker SVCs so we can see what they called after waking */
                        fprintf(stderr, "  [svc_ring] last worker SVCs (newest first):\n");
                        uint32_t wdump = 0;
                        for (uint32_t bi = 0; bi < 1024u && wdump < 30u; ++bi) {
                            const SvcTraceEnt &e = g_svc_ring[(g_svc_ring_pos - 1u - bi) & 1023u];
                            if (e.tid != 0u && (e.svc || e.lr)) {
                                fprintf(stderr, "    svc=%-5u lr=0x%08x r0=0x%08x tid=%u\n",
                                        e.svc, e.lr, e.r0, e.tid);
                                ++wdump;
                            }
                        }
                        /* dump g_conds */
                        fprintf(stderr, "  [g_conds] %zu entries:\n", g_conds.size());
                        uint32_t cdump = 0;
                        for (auto& kv2 : g_conds) {
                            if (cdump++ < 16u)
                                fprintf(stderr, "    cond=0x%08x val=%u\n", kv2.first, kv2.second);
                        }
                        /* dump thread PC/finished state incl. what each is blocked on */
                        fprintf(stderr, "  [threads] %zu total:\n", g_threads.size());
                        uint32_t tdump = 0;
                        for (auto& t3 : g_threads) {
                            if (tdump++ < 80u)
                                fprintf(stderr, "    tid=%-3u finished=%d pc=0x%08x entry=0x%08x "
                                        "wfutex=0x%08x wsem=0x%08x lr=0x%08x\n",
                                        t3.id, (int)t3.finished, t3.regs[15] & ~1u, t3.entry_pc & ~1u,
                                        t3.waiting_futex, t3.waiting_sem, t3.regs[14] & ~1u);
                        }
                    }
                }
            }
        } else if (!g_threads.empty()) {
            schedule_threads(env_ticks("LUNARIA_THREAD_TICKS", 200'000'000ULL));
        }

        if (mutex) {
            uint32_t w = saved_w < 0x100u ? 0x100u : (saved_w & ~0xffu);
            ctx.mem.write32(mutex, w | (g_current_tid + 1u));
        }

        if (g_current_tid != 0) g_yield_requested = true;
        ret32(0); break;
    }
    case SVC_PTHREAD_COND_TIMEDWAIT: {
        /* ignore timeout here; yield one slice */
        uint32_t cond  = r0;
        uint32_t mutex = r1;
        if (cond) {
            auto it = g_conds.find(cond);
            if (it != g_conds.end() && it->second > 0) {
                if (it->second != UINT32_MAX) --it->second;
                ret32(0); break;
            }
        }
        uint32_t saved_w = mutex ? ctx.mem.read32(mutex) : 0u;
        if (mutex) ctx.mem.write32(mutex, 0u);
        if (!g_threads.empty()) schedule_threads(env_ticks("LUNARIA_THREAD_TICKS", 200'000'000ULL));
        if (mutex) {
            uint32_t w = saved_w < 0x100u ? 0x100u : (saved_w & ~0xffu);
            ctx.mem.write32(mutex, w | (g_current_tid + 1u));
        }
        
        if (g_current_tid != 0) g_yield_requested = true;
        ret32(0); break;
    }

    /* ---- pthread_rwlock -------------------------------------------------- */
    case SVC_PTHREAD_RWLOCK_INIT:
        if (r0) { g_rwlocks[r0] = 0; g_rwlock_writer.erase(r0); }
        ret32(0); break;
    case SVC_PTHREAD_RWLOCK_RDLOCK:
        
        if (r0) g_rwlocks[r0] = std::max(0, g_rwlocks[r0]) + 1;
        ret32(0); break;
    case SVC_PTHREAD_RWLOCK_TRYRDLOCK:
        
        if (r0 && g_rwlocks.count(r0) && g_rwlocks[r0] < 0) {
            ret32(16u /* EBUSY */);
        } else {
            if (r0) g_rwlocks[r0] = std::max(0, g_rwlocks[r0]) + 1;
            ret32(0);
        }
        break;
    case SVC_PTHREAD_RWLOCK_WRLOCK:
        
        if (r0) { g_rwlocks[r0] = -1; g_rwlock_writer[r0] = g_current_tid; }
        ret32(0); break;
    case SVC_PTHREAD_RWLOCK_TRYWRLOCK:
        if (r0 && g_rwlocks.count(r0) && g_rwlocks[r0] != 0) {
            ret32(16u /* EBUSY */);
        } else {
            if (r0) { g_rwlocks[r0] = -1; g_rwlock_writer[r0] = g_current_tid; }
            ret32(0);
        }
        break;
    case SVC_PTHREAD_RWLOCK_UNLOCK:
        if (r0) {
            auto it = g_rwlocks.find(r0);
            if (it != g_rwlocks.end()) {
                if (it->second < 0) {
                    it->second = 0; g_rwlock_writer.erase(r0);
                } else if (it->second > 0) {
                    --it->second;
                }
            }
        }
        ret32(0); break;
    case SVC_PTHREAD_RWLOCK_DESTROY:
        if (r0) { g_rwlocks.erase(r0); g_rwlock_writer.erase(r0); }
        ret32(0); break;

    /* ---- ARM Linux kuser helpers ---- */
    case SVC_KUSER_CMPXCHG: {
        
        uint32_t cur = ctx.mem.read32(r2);
        if (cur == r0) { ctx.mem.write32(r2, r1); ret32(0); }
        else           { ret32(1u); }
        break;
    }
    case SVC_KUSER_GET_TLS: {
        
        if (g_ctx) {
            static std::map<uint32_t, uint32_t> tls_blocks;
            auto it = tls_blocks.find(g_current_tid);
            if (it != tls_blocks.end()) { ret32(it->second); break; }
            uint32_t va = arm_malloc(*g_ctx, 4096);
            tls_blocks[g_current_tid] = va;
            ret32(va);
        } else { ret32(0); }
        break;
    }

    /* ---- exception-identification logging (LUNARIA_TRACE_EXC) ----
 * These fire from the build_exc_logger_stub tail-call stubs.  They MUST NOT
 * modify any guest register (no ret32) — the stub then tail-calls the real
 * libmono function with the original args/lr intact. */
    case SVC_EXC_FROM_NAME: {
        /* mono_exception_from_name_msg(image, name_space, name, msg)
 * r0=image r1=name_space r2=name r3=msg  → the class name directly. */
        auto gstr = [&](uint32_t va) -> const char * {
            if (!va) return "(null)";
            const uint8_t *p = ctx.mem.ptr(va);
            return p ? reinterpret_cast<const char *>(p) : "(unmapped)";
        };
        fprintf(stderr, "[exc] from_name_msg ns='%.64s' name='%.64s' msg='%.160s' "
                "lr=0x%08x tid=%u\n",
                gstr(r1), gstr(r2), gstr(r3), regs[14], g_current_tid);
        break;
    }
    case SVC_EXC_RAISE: {
        /* mono_raise_exception(MonoException *exc)  r0=object.
 * Walk object→vtable→class and read the class name_space/name.  The
 * MonoClass name/name_space offsets vary by mono version; we probe a
 * small set of plausible offsets and print any that look like C strings. */
        uint32_t obj = r0;
        uint32_t vt  = obj ? ctx.mem.read32(obj) : 0;
        uint32_t kls = vt  ? ctx.mem.read32(vt)  : 0;
        fprintf(stderr, "[exc] raise obj=0x%08x vtable=0x%08x class=0x%08x lr=0x%08x tid=%u\n",
                obj, vt, kls, regs[14], g_current_tid);
        if (kls) {
            /* dump a window of the MonoClass so the name pointers are visible */
            for (uint32_t off = 0x28; off <= 0x40; off += 4) {
                uint32_t p = ctx.mem.read32(kls + off);
                const uint8_t *hp = p ? ctx.mem.ptr(p) : nullptr;
                if (hp) {
                    char buf[48]; size_t i = 0;
                    for (; i < sizeof(buf)-1 && hp[i] >= 0x20 && hp[i] < 0x7f; ++i)
                        buf[i] = (char)hp[i];
                    buf[i] = 0;
                    if (i >= 3)
                        fprintf(stderr, "[exc]   class+0x%02x -> '%s'\n", off, buf);
                }
            }
        }
        break;
    }

    /* ---- __aeabi_* / fortify / libc batch 2 ---- */

    case SVC_AEABI_MEMSET: {
        
        if (r1) memset(ctx.mem.ptr(r0), (int)(r2 & 0xFF), r1);
        ret32(r0); break;
    }
    case SVC_AEABI_MEMCLR: {
        if (r1) memset(ctx.mem.ptr(r0), 0, r1);
        ret32(r0); break;
    }
    case SVC_STRLCPY: {
        
        const char *src = ctx.mem.cstr(r1);
        if (!src) { ret32(0); break; }
        size_t slen = strlen(src);
        if (r0 && r2) {
            size_t n = slen < r2 - 1 ? slen : r2 - 1;
            memcpy(ctx.mem.ptr(r0), src, n);
            *ctx.mem.ptr(r0 + n) = 0;
        }
        ret32((uint32_t)slen);
        break;
    }
    case SVC_STRNCASECMP: {
        const char *a = ctx.mem.cstr(r0), *b = ctx.mem.cstr(r1);
        ret32((a && b) ? (uint32_t)strncasecmp(a, b, r2) : 0u);
        break;
    }
    case SVC_TOLOWER:  ret32((uint32_t)tolower((int)r0)); break;
    
    case SVC_BTOWC:    ret32((uint32_t)btowc((int)r0)); break;
    case SVC_WCTOB:    ret32((uint32_t)wctob((wint_t)r0)); break;
    case SVC_TOWLOWER: ret32((uint32_t)towlower((wint_t)r0)); break;
    case SVC_TOWUPPER: ret32((uint32_t)towupper((wint_t)r0)); break;
    case SVC_ISWCTYPE: ret32((uint32_t)iswctype((wint_t)r0, (wctype_t)r1)); break;
    case SVC_WCTYPE: {
        const char *name = ctx.mem.cstr(r0);
        ret32(name ? (uint32_t)(uintptr_t)wctype(name) : 0u);
        break;
    }
    case SVC_MBRTOWC: {
        /* mbrtowc(wchar_t* pwc, const char* s, size_t n, mbstate_t* ps).
 * The guest mbstate_t is opaque to us; use a host-local one (C locale is
 * stateless for UTF-8 lead bytes we care about). */
        wchar_t wc = 0;
        size_t n = (r1 == 0) ? 0
                 : mbrtowc(&wc, (const char*)ctx.mem.ptr(r1), (size_t)r2, nullptr);
        if (r0 && r1 && n != (size_t)-1 && n != (size_t)-2)
            ctx.mem.write32(r0, (uint32_t)wc);
        ret32((uint32_t)n);
        break;
    }
    case SVC_WCRTOMB: {
        /* wcrtomb(char* s, wchar_t wc, mbstate_t* ps) */
        char buf[MB_LEN_MAX];
        size_t n = wcrtomb(buf, (wchar_t)r1, nullptr);
        if (r0 && n != (size_t)-1) memcpy(ctx.mem.ptr(r0), buf, n);
        ret32((uint32_t)n);
        break;
    }
    case SVC_WMEMCHR: {
        /* wmemchr(const wchar_t* s, wchar_t c, size_t n) — guest wchar_t is 4B. */
        uint32_t found = 0;
        for (uint32_t i = 0; i < r2; ++i) {
            if (ctx.mem.read32(r0 + i * 4u) == r1) { found = r0 + i * 4u; break; }
        }
        ret32(found);
        break;
    }
    case SVC_STRCOLL: {
        const char *a = ctx.mem.cstr(r0), *b = ctx.mem.cstr(r1);
        ret32((a && b) ? (uint32_t)strcoll(a, b) : 0u);
        break;
    }
    case SVC_STRXFRM: {
        const char *src = ctx.mem.cstr(r1);
        size_t need = src ? strxfrm(nullptr, src, 0) : 0;
        if (r0 && src && r2 > need)
            strxfrm((char*)ctx.mem.ptr(r0), src, (size_t)r2);
        ret32((uint32_t)need);
        break;
    }
    case SVC_STRCASESTR: {
        const char *h = ctx.mem.cstr(r0), *n = ctx.mem.cstr(r1);
        const char *p = (h && n) ? strcasestr(h, n) : nullptr;
        ret32(p ? (uint32_t)(r0 + (uint32_t)(p - h)) : 0u);
        break;
    }
    case SVC_STRSEP: {
        /* strsep(char** stringp, const char* delim): *stringp is a guest ptr. */
        if (!r0) { ret32(0); break; }
        uint32_t cur = ctx.mem.read32(r0);
        if (!cur) { ret32(0); break; }
        const char *delim = ctx.mem.cstr(r1);
        char *base = (char*)ctx.mem.ptr(cur);
        size_t span = delim ? strcspn(base, delim) : strlen(base);
        uint32_t tok = cur;
        if (base[span]) {
            base[span] = '\0';
            ctx.mem.write32(r0, cur + (uint32_t)span + 1u);
        } else {
            ctx.mem.write32(r0, 0u); /* last token */
        }
        ret32(tok);
        break;
    }
    case SVC_ISALPHA:  ret32(isalpha((int)(r0 & 0xff)) ? 1u : 0u); break;
    case SVC_ISDIGIT:  ret32(isdigit((int)(r0 & 0xff)) ? 1u : 0u); break;
    case SVC_ISALNUM:  ret32(isalnum((int)(r0 & 0xff)) ? 1u : 0u); break;
    case SVC_ISXDIGIT: ret32(isxdigit((int)(r0 & 0xff)) ? 1u : 0u); break;
    case SVC_ISASCII:  ret32(r0 < 0x80 ? 1u : 0u); break;
    case SVC_ABS:      ret32((uint32_t)abs((int)r0)); break;
    case SVC_MKDIR: {
        const char *path = ctx.mem.cstr(r0);
        ret32(path ? (uint32_t)mkdir(path, (mode_t)r1) : ~0u);
        break;
    }
    case SVC_GETCWD: {
        char buf[PATH_MAX];
        if (!getcwd(buf, sizeof buf)) { ret32(0); break; }
        uint32_t dst = r0 ? r0 : arm_malloc(ctx, (uint32_t)strlen(buf) + 1);
        if (dst && (!r1 || strlen(buf) < r1))
            strcpy((char*)ctx.mem.ptr(dst), buf);
        ret32(dst);
        break;
    }
    case SVC_UNLINK: {
        const char *path = ctx.mem.cstr(r0);
        ret32(path ? (uint32_t)unlink(path) : ~0u);
        break;
    }
    case SVC_RENAME: {
        const char *a = ctx.mem.cstr(r0), *b = ctx.mem.cstr(r1);
        ret32((a && b) ? (uint32_t)rename(a, b) : ~0u);
        break;
    }
    case SVC_FTRUNCATE:
        ret32((uint32_t)ftruncate((int)r0, (off_t)(int32_t)r1)); break;
    case SVC_READLINK: {
        const char *path = ctx.mem.cstr(r0);
        ssize_t n = (path && r1) ? readlink(path, (char*)ctx.mem.ptr(r1), r2) : -1;
        ret32((uint32_t)n);
        break;
    }
    case SVC_WRITEV: {
        /* writev(fd, iovec*, cnt) — LP32 iovec {u32 base; u32 len} */
        ssize_t total = 0;
        for (uint32_t i = 0; i < r2; ++i) {
            uint32_t base = ctx.mem.read32(r1 + i*8);
            uint32_t len  = ctx.mem.read32(r1 + i*8 + 4);
            if (base && len) total += write((int)r0, ctx.mem.ptr(base), len);
        }
        ret32((uint32_t)total);
        break;
    }
    case SVC_CLOCK: ret32((uint32_t)clock()); break;
    case SVC_LOCALTIME_R: case SVC_GMTIME_R: {
        time_t t = (time_t)ctx.mem.read32(r0);
        struct tm tmv;
        if (svc_no == SVC_LOCALTIME_R) localtime_r(&t, &tmv); else gmtime_r(&t, &tmv);
        if (r1) write_guest_tm(ctx, r1, tmv);
        ret32(r1);
        break;
    }
    case SVC_LOCALTIME: case SVC_GMTIME: {
        static uint32_t tm_static = 0;
        if (!tm_static) tm_static = arm_malloc(ctx, 44);
        time_t t = (time_t)ctx.mem.read32(r0);
        struct tm tmv;
        if (svc_no == SVC_LOCALTIME) localtime_r(&t, &tmv); else gmtime_r(&t, &tmv);
        if (tm_static) write_guest_tm(ctx, tm_static, tmv);
        ret32(tm_static);
        /* Spin detection: log LR, r0 (time_t addr), and the actual time value
 * the first time, then periodically, so we know why it's spinning. */
        {
            static uint32_t lt_ctr = 0; static uint32_t lt_lr0 = 0;
            uint32_t lr = regs[14];
            if (lt_ctr++ < 3 || (lt_ctr % 50000 == 0))
                fprintf(stderr, "[localtime-spin] #%u lr=0x%08x r0=0x%08x t=%ld "
                        "hh:mm:ss=%02d:%02d:%02d\n",
                        lt_ctr, lr, r0, (long)t, tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
            if (lr != lt_lr0 && lt_ctr > 1) { lt_ctr = 0; lt_lr0 = lr; }
        }
        break;
    }
    case SVC_MKTIME: {
        struct tm tmv;
        read_guest_tm(ctx, r0, tmv);
        tmv.tm_isdst = -1;
        time_t t = mktime(&tmv);
        /* mktime normalizes the struct in place (tm_wday/tm_yday/tm_isdst and,
 * on bionic/glibc, tm_gmtoff/tm_zone).  Mono's GetTimeZoneData relies on
 * this: it computes gmtoff from the mktime-normalized struct and then
 * do{ t-=3600; }while(localtime(&t)->tm_gmtoff != gmtoff) — without the
 * write-back gmtoff stays 0 (memset) and the loop never terminates. */
        if (r0) write_guest_tm(ctx, r0, tmv);
        ret32((uint32_t)t);
        break;
    }
    case SVC_DIFFTIME: {
        double d = difftime((time_t)r0, (time_t)r1);
        uint64_t bits; memcpy(&bits, &d, 8); ret64(bits);
        break;
    }
    case SVC_STRFTIME: {
        const char *fmt = ctx.mem.cstr(r2);
        struct tm tmv;
        read_guest_tm(ctx, r3, tmv);
        char buf[512];
        size_t n = fmt ? strftime(buf, sizeof buf, fmt, &tmv) : 0;
        if (r0 && r1) {
            size_t m = n < r1 - 1 ? n : r1 - 1;
            memcpy(ctx.mem.ptr(r0), buf, m);
            *ctx.mem.ptr(r0 + m) = 0;
        }
        ret32((uint32_t)n);
        break;
    }
    case SVC_UNAME: {
        /* bionic utsname: 6 fields × 65 byte */
        static const char *fields[6] =
            { "Linux", "lunaria", "4.14.186", "#1 SMP PREEMPT", "armv7l", "localdomain" };
        if (r0) for (int i = 0; i < 6; ++i)
            strncpy((char*)ctx.mem.ptr(r0 + (uint32_t)i*65u), fields[i], 64);
        ret32(0);
        break;
    }
    case SVC_GETDTABLESIZE:
        /* getdtablesize(): max number of open fds.  mono's io-layer
 * _wapi_handle_init rounds this up to a multiple of 256 to size
 * _wapi_fd_reserve and the handle-table segment count; a garbage value
 * (the prior unresolved-symbol behaviour) breaks the segment layout so
 * every handle reads back WAPI_HANDLE_UNUSED.  Report the host limit
 * (clamped to a sane 1024 like stock Android). */
        ret32(1024u);
        break;
    case SVC_BSEARCH: {
        /* bsearch(key, base, nmemb, size, compar): 5th arg (compar) is on the
 * guest stack at [sp].  The comparator is a *guest* function, so it must
 * run on the guest CPU — we drive it through the dedicated callback JIT.
 * Mono uses this for mono_metadata_typedef_from_method and friends; an
 * unimplemented bsearch made every method-token resolution fail. */
        uint32_t key   = r0;
        uint32_t base  = r1;
        uint32_t nmemb = r2;
        uint32_t size  = r3;
        uint32_t compar = ctx.mem.read32(regs[13]);
        uint32_t found = 0;
        if (compar && size) {
            uint32_t lo = 0, hi = nmemb;
            while (lo < hi) {
                uint32_t mid  = lo + (hi - lo) / 2u;
                uint32_t elem = base + mid * size;
                int32_t  c    = call_guest_cb(ctx, compar, key, elem);
                if      (c < 0) hi = mid;
                else if (c > 0) lo = mid + 1u;
                else { found = elem; break; }
            }
        }
        ret32(found);
        break;
    }
    case SVC_QSORT: {
        
        uint32_t base   = r0;
        uint32_t nmemb  = r1;
        uint32_t size   = r2;
        uint32_t compar = r3;
        if (!compar || !size || nmemb < 2 || !base) { ret32(0); break; }
        
        std::vector<uint32_t> order(nmemb);
        std::iota(order.begin(), order.end(), 0u);
        
        std::vector<uint8_t> buf(nmemb * size);
        for (uint32_t i = 0; i < nmemb; ++i) {
            const uint8_t *src = ctx.mem.ptr(base + i * size);
            if (src) memcpy(buf.data() + i * size, src, size);
        }
        
        for (uint32_t i = 1; i < nmemb; ++i) {
            uint32_t key = order[i];
            int32_t j = (int32_t)i - 1;
            while (j >= 0) {
                
                uint32_t va_a = base + (uint32_t)j * size;
                uint32_t va_b = base + key * size;
                uint8_t *pa = ctx.mem.ptr(va_a);
                uint8_t *pb = ctx.mem.ptr(va_b);
                
                if (pa && pb) memcpy(pb, buf.data() + key * size, size);
                if (pa)       memcpy(pa, buf.data() + order[j] * size, size);
                int32_t c = call_guest_cb(ctx, compar, va_a, va_b);
                if (c <= 0) break;
                order[j + 1] = order[j];
                --j;
            }
            order[j + 1] = key;
        }
        
        for (uint32_t i = 0; i < nmemb; ++i) {
            uint8_t *dst = ctx.mem.ptr(base + i * size);
            if (dst) memcpy(dst, buf.data() + order[i] * size, size);
        }
        ret32(0); break;
    }
    case SVC_CXA_GUARD_ACQUIRE: {
        /* int __cxa_guard_acquire(guard*): r0 = guard.  Single-threaded model —
 * if the guard's first byte is already set the initialiser has run, so
 * return 0 (skip).  Otherwise return 1 so the caller runs it and then
 * calls __cxa_guard_release.  We deliberately do NOT set a "pending"
 * marker here: the compiler's inline ldrb fast-path tests this same byte
 * for non-zero, and a pending marker would make a recursive re-check
 * skip the not-yet-finished initialiser. */
        uint32_t acq = ctx.mem.ptr(r0)[0] ? 0u : 1u;
        if (getenv("LUNARIA_TRACE_CXA"))
            fprintf(stderr, "[cxa] acquire guard=0x%08x lr=0x%08x -> %u\n",
                    r0, regs[14], acq);
        ret32(acq);
        break;
    }
    case SVC_CXA_GUARD_RELEASE:
        /* void __cxa_guard_release(guard*): mark init complete.  Set bit 0 (=1)
 * so both the "test non-zero" and "test bit 0" inline check variants see
 * it as initialised. */
        ctx.mem.ptr(r0)[0] = 1u;
        if (getenv("LUNARIA_TRACE_CXA"))
            fprintf(stderr, "[cxa] release guard=0x%08x lr=0x%08x\n", r0, regs[14]);
        ret32(0);
        break;
    case SVC_CXA_GUARD_ABORT:
        /* void __cxa_guard_abort(guard*): initialiser threw — leave the guard
 * clear so a later attempt re-runs it. */
        ctx.mem.ptr(r0)[0] = 0u;
        if (getenv("LUNARIA_TRACE_CXA"))
            fprintf(stderr, "[cxa] ABORT guard=0x%08x lr=0x%08x\n", r0, regs[14]);
        ret32(0);
        break;
    case SVC_CXA_PURE_VIRTUAL:
        /* __cxa_pure_virtual(): a pure virtual was called — normally fatal.
 * Log once (rate-limited) and return instead of aborting so one bad
 * vtable slot doesn't kill the whole frame; the caller gets r0 unchanged. */
        {
            static int warned = 0;
            if (warned < 8) {
                fprintf(stderr, "[arm_exec] __cxa_pure_virtual called lr=0x%08x "
                        "(pure virtual call — returning)\n", regs[14]);
                ++warned;
            }
        }
        ret32(0);
        break;
    case SVC_AEABI_ATEXIT:
        /* int __aeabi_atexit(void* obj, void(*dtor)(void*), void* dso): register
 * a static-object destructor.  We never run global dtors (the process is
 * torn down wholesale), so just report success.  Returning the trampoline
 * garbage previously could make callers believe registration failed. */
        ret32(0);
        break;

    /* fp classification, fenv, wide-char classification/conversion */
    case SVC_ISNAN: {
        /* isnan(double): r0=low word, r1=high word */
        uint64_t bits = (uint64_t)r1 << 32 | r0;
        double v; memcpy(&v, &bits, 8);
        ret32(std::isnan(v) ? 1u : 0u);
        break;
    }
    case SVC_ISINF: {
        uint64_t bits = (uint64_t)r1 << 32 | r0;
        double v; memcpy(&v, &bits, 8);
        ret32(std::isinf(v) ? (v > 0 ? 1u : (uint32_t)-1) : 0u);
        break;
    }
    case SVC_ISFINITE: {
        uint64_t bits = (uint64_t)r1 << 32 | r0;
        double v; memcpy(&v, &bits, 8);
        ret32(std::isfinite(v) ? 1u : 0u);
        break;
    }
    case SVC_SIGNBIT: {
        uint64_t bits = (uint64_t)r1 << 32 | r0;
        double v; memcpy(&v, &bits, 8);
        ret32(std::signbit(v) ? 1u : 0u);
        break;
    }

    
    case SVC_FEGETROUND:
        ret32((uint32_t)fegetround());
        break;
    case SVC_FESETROUND:
        ret32((uint32_t)fesetround((int)r0));
        break;
    case SVC_FECLEAREXCEPT:
        ret32((uint32_t)feclearexcept((int)r0));
        break;
    case SVC_FERAISEEXCEPT:
        ret32((uint32_t)feraiseexcept((int)r0));
        break;
    case SVC_FETESTEXCEPT:
        ret32((uint32_t)fetestexcept((int)r0));
        break;

    
    case SVC_ISWSPACE:  ret32(iswspace((wint_t)r0)  ? 1u : 0u); break;
    case SVC_ISWDIGIT:  ret32(iswdigit((wint_t)r0)  ? 1u : 0u); break;
    case SVC_ISWALPHA:  ret32(iswalpha((wint_t)r0)  ? 1u : 0u); break;
    case SVC_ISWUPPER:  ret32(iswupper((wint_t)r0)  ? 1u : 0u); break;
    case SVC_ISWLOWER:  ret32(iswlower((wint_t)r0)  ? 1u : 0u); break;
    case SVC_ISWPRINT:  ret32(iswprint((wint_t)r0)  ? 1u : 0u); break;
    case SVC_ISWPUNCT:  ret32(iswpunct((wint_t)r0)  ? 1u : 0u); break;
    case SVC_ISWGRAPH:  ret32(iswgraph((wint_t)r0)  ? 1u : 0u); break;
    case SVC_ISWALNUM:  ret32(iswalnum((wint_t)r0)  ? 1u : 0u); break;
    case SVC_ISWBLANK:  ret32(iswblank((wint_t)r0)  ? 1u : 0u); break;
    case SVC_ISWCNTRL:  ret32(iswcntrl((wint_t)r0)  ? 1u : 0u); break;
    case SVC_WCTRANS: {
        /* wctrans(const char* property) → wctrans_t (opaque handle, fits in uint32) */
        const char *prop = r0 ? ctx.mem.cstr(r0) : nullptr;
        wctrans_t h = prop ? wctrans(prop) : (wctrans_t)0;
        ret32((uint32_t)(uintptr_t)h);
        break;
    }
    case SVC_TOWCTRANS: {
        /* towctrans(wint_t wc, wctrans_t t) → wint_t */
        wctrans_t t = (wctrans_t)(uintptr_t)r1;
        ret32((uint32_t)towctrans((wint_t)r0, t));
        break;
    }

    /* ---- wide-char stringconversion ---- */
    case SVC_STRTOLD: {
        /* strtold(const char* nptr, char** endptr) */
        const char *s = r0 ? ctx.mem.cstr(r0) : nullptr;
        char *end = nullptr;
        double v = s ? strtod(s, &end) : 0.0;
        if (r1 && s && end) {
            uint32_t off = (uint32_t)(end - (const char*)ctx.mem.ptr(r0));
            ctx.mem.write32(r1, r0 + off);
        }
        uint64_t bits; memcpy(&bits, &v, 8);
        ret64(bits);
        break;
    }
    case SVC_WCSTOD: {
        /* wcstod(const wchar_t* nptr, wchar_t** endptr) */
        const wchar_t *ws = r0 ? (const wchar_t*)ctx.mem.ptr(r0) : nullptr;
        wchar_t *wend = nullptr;
        double v = ws ? wcstod(ws, &wend) : 0.0;
        if (r1 && ws && wend) {
            uint32_t off = (uint32_t)((const uint8_t*)wend - (const uint8_t*)ws);
            ctx.mem.write32(r1, r0 + off);
        }
        uint64_t bits; memcpy(&bits, &v, 8);
        ret64(bits);
        break;
    }
    case SVC_WCSTOL: {
        const wchar_t *ws = r0 ? (const wchar_t*)ctx.mem.ptr(r0) : nullptr;
        wchar_t *wend = nullptr;
        long v = ws ? wcstol(ws, &wend, (int)r2) : 0L;
        if (r1 && ws && wend) {
            uint32_t off = (uint32_t)((const uint8_t*)wend - (const uint8_t*)ws);
            ctx.mem.write32(r1, r0 + off);
        }
        ret32((uint32_t)v);
        break;
    }
    case SVC_WCSTOUL: {
        const wchar_t *ws = r0 ? (const wchar_t*)ctx.mem.ptr(r0) : nullptr;
        wchar_t *wend = nullptr;
        unsigned long v = ws ? wcstoul(ws, &wend, (int)r2) : 0UL;
        if (r1 && ws && wend) {
            uint32_t off = (uint32_t)((const uint8_t*)wend - (const uint8_t*)ws);
            ctx.mem.write32(r1, r0 + off);
        }
        ret32((uint32_t)v);
        break;
    }
    case SVC_WCSTOLL: {
        const wchar_t *ws = r0 ? (const wchar_t*)ctx.mem.ptr(r0) : nullptr;
        wchar_t *wend = nullptr;
        long long v = ws ? wcstoll(ws, &wend, (int)r2) : 0LL;
        if (r1 && ws && wend) {
            uint32_t off = (uint32_t)((const uint8_t*)wend - (const uint8_t*)ws);
            ctx.mem.write32(r1, r0 + off);
        }
        ret64((uint64_t)v);
        break;
    }
    case SVC_WCSTOULL: {
        const wchar_t *ws = r0 ? (const wchar_t*)ctx.mem.ptr(r0) : nullptr;
        wchar_t *wend = nullptr;
        unsigned long long v = ws ? wcstoull(ws, &wend, (int)r2) : 0ULL;
        if (r1 && ws && wend) {
            uint32_t off = (uint32_t)((const uint8_t*)wend - (const uint8_t*)ws);
            ctx.mem.write32(r1, r0 + off);
        }
        ret64(v);
        break;
    }
    case SVC_STRTOIMAX: {
        /* strtoimax(const char* s, char** end, int base) → intmax_t (64-bit) */
        const char *s = r0 ? ctx.mem.cstr(r0) : nullptr;
        char *end = nullptr;
        long long v = s ? strtoll(s, &end, (int)r2) : 0LL;
        if (r1 && s && end) {
            uint32_t off = (uint32_t)(end - (const char*)ctx.mem.ptr(r0));
            ctx.mem.write32(r1, r0 + off);
        }
        ret64((uint64_t)v);
        break;
    }
    case SVC_STRTOUMAX: {
        const char *s = r0 ? ctx.mem.cstr(r0) : nullptr;
        char *end = nullptr;
        unsigned long long v = s ? strtoull(s, &end, (int)r2) : 0ULL;
        if (r1 && s && end) {
            uint32_t off = (uint32_t)(end - (const char*)ctx.mem.ptr(r0));
            ctx.mem.write32(r1, r0 + off);
        }
        ret64(v);
        break;
    }
    case SVC_WCSCMP: {
        const wchar_t *a = r0 ? (const wchar_t*)ctx.mem.ptr(r0) : nullptr;
        const wchar_t *b = r1 ? (const wchar_t*)ctx.mem.ptr(r1) : nullptr;
        ret32(a && b ? (uint32_t)wcscmp(a, b) : (r0 ? 1u : (uint32_t)-1));
        break;
    }
    case SVC_WCSNCMP: {
        const wchar_t *a = r0 ? (const wchar_t*)ctx.mem.ptr(r0) : nullptr;
        const wchar_t *b = r1 ? (const wchar_t*)ctx.mem.ptr(r1) : nullptr;
        ret32(a && b ? (uint32_t)wcsncmp(a, b, (size_t)r2) : 0u);
        break;
    }
    case SVC_WCSCPY: {
        
        wchar_t *dst = r0 ? (wchar_t*)ctx.mem.ptr(r0) : nullptr;
        const wchar_t *src = r1 ? (const wchar_t*)ctx.mem.ptr(r1) : nullptr;
        if (dst && src) wcscpy(dst, src);
        ret32(r0);
        break;
    }
    case SVC_WCSCAT: {
        wchar_t *dst = r0 ? (wchar_t*)ctx.mem.ptr(r0) : nullptr;
        const wchar_t *src = r1 ? (const wchar_t*)ctx.mem.ptr(r1) : nullptr;
        if (dst && src) {
            if (r2) wcsncat(dst, src, (size_t)r2);
            else wcscat(dst, src);
        }
        ret32(r0);
        break;
    }

    
    case SVC_DIV: {
        if ((int32_t)r1 == 0) { ret32(0); break; }
        div_t d = div((int)r0, (int)r1);
        regs[0] = (uint32_t)d.quot;
        regs[1] = (uint32_t)d.rem;
        break;
    }
    case SVC_LDIV: {
        if ((int32_t)r1 == 0) { ret32(0); break; }
        ldiv_t d = ldiv((long)r0, (long)r1);
        regs[0] = (uint32_t)d.quot;
        regs[1] = (uint32_t)d.rem;
        break;
    }

    /* for symbols not in kSymbolSvcMap */
    
    case SVC_FMA: {
        /* fma(double x[r0:r1], double y[r2:r3], double z[sp+0:sp+4]) → double[r0:r1] */
        uint64_t ix = (uint64_t)r0 | ((uint64_t)r1 << 32);
        uint64_t iy = (uint64_t)r2 | ((uint64_t)r3 << 32);
        uint32_t sp = regs[13];
        uint32_t zlo = ctx.mem.read32(sp);
        uint32_t zhi = ctx.mem.read32(sp + 4);
        uint64_t iz = (uint64_t)zlo | ((uint64_t)zhi << 32);
        double x, y, z; memcpy(&x, &ix, 8); memcpy(&y, &iy, 8); memcpy(&z, &iz, 8);
        double res = fma(x, y, z);
        uint64_t rb; memcpy(&rb, &res, 8);
        regs[0] = (uint32_t)rb; regs[1] = (uint32_t)(rb >> 32);
        break;
    }
    case SVC_FMAF: {
        /* fmaf(float x[r0], float y[r1], float z[r2]) → float[r0] */
        float fx, fy, fz;
        memcpy(&fx, &r0, 4); memcpy(&fy, &r1, 4); memcpy(&fz, &r2, 4);
        float res = fmaf(fx, fy, fz);
        uint32_t bits; memcpy(&bits, &res, 4);
        ret32(bits);
        break;
    }
    case SVC_SCALBN: {
        /* scalbn(double x[r0:r1], int n[r2]) → double[r0:r1] */
        uint64_t ix = (uint64_t)r0 | ((uint64_t)r1 << 32);
        double x; memcpy(&x, &ix, 8);
        double res = scalbn(x, (int)(int32_t)r2);
        uint64_t rb; memcpy(&rb, &res, 8);
        regs[0] = (uint32_t)rb; regs[1] = (uint32_t)(rb >> 32);
        break;
    }
    case SVC_SCALBNF: {
        /* scalbnf(float x[r0], int n[r1]) → float[r0] */
        float fx; memcpy(&fx, &r0, 4);
        float res = scalbnf(fx, (int)(int32_t)r1);
        uint32_t bits; memcpy(&bits, &res, 4);
        ret32(bits);
        break;
    }
    case SVC_ILOGB: {
        /* ilogb(double x[r0:r1]) → int[r0] */
        uint64_t ix = (uint64_t)r0 | ((uint64_t)r1 << 32);
        double x; memcpy(&x, &ix, 8);
        ret32((uint32_t)(int32_t)ilogb(x));
        break;
    }
    case SVC_ILOGBF: {
        /* ilogbf(float x[r0]) → int[r0] */
        float fx; memcpy(&fx, &r0, 4);
        ret32((uint32_t)(int32_t)ilogbf(fx));
        break;
    }
    case SVC_DUP: {
        /* dup(oldfd) → newfd or -1 */
        ret32((uint32_t)(int32_t)dup((int)r0));
        break;
    }
    case SVC_FERROR: {
        /* ferror(FILE* guest_stream) → int */
        FILE *f = resolve_file(ctx, r0);
        ret32(f ? (uint32_t)ferror(f) : 0u);
        break;
    }
    case SVC_REWINDDIR: {
        
        auto it = g_dir_tab.find(r0);
        if (it != g_dir_tab.end()) {
            if (it->second.d)
                rewinddir(it->second.d);
            else
                it->second.apk_idx = 0; 
        }
        ret32(0);
        break;
    }
    case SVC_MBTOWC: {
        /* mbtowc(wchar_t* pwc[r0], const char* s[r1], size_t n[r2]) → int */
        if (r1 == 0) { ret32(0); break; }  /* MB_CUR_MAX query */
        const char *s = (const char*)ctx.mem.ptr(r1);
        if (!s) { ret32(-1); break; }
        wchar_t wc = 0;
        int res = mbtowc(&wc, s, (size_t)r2);
        if (res > 0 && r0) ctx.mem.write32(r0, (uint32_t)wc);
        ret32((uint32_t)(int32_t)res);
        break;
    }
    case SVC_MBRLEN: {
        /* mbrlen(const char* s[r0], size_t n[r1], mbstate_t* ps[r2]) → size_t */
        const char *s = r0 ? (const char*)ctx.mem.ptr(r0) : nullptr;
        if (!s) { ret32(0); break; }
        
        if ((unsigned char)*s < 0x80) { ret32(*s ? 1u : 0u); break; }
        ret32((uint32_t)(ssize_t)mbrlen(s, (size_t)r1, nullptr));
        break;
    }
    case SVC_MBSRTOWCS: {
        /* mbsrtowcs(wchar_t* dst[r0], const char** src_ptr[r1], size_t n[r2], mbstate_t* ps[r3]) */
        if (!r1) { ret32(0); break; }
        
        uint32_t src_va = ctx.mem.read32(r1);
        if (!src_va) { ret32(0); break; }
        const char *src     = (const char*)ctx.mem.ptr(src_va);
        const char *src_org = src;
        size_t n = (size_t)r2;
        if (n == 0 || n > 0x4000000u) n = 0x4000000u; 
        if (r0 == 0) {
            
            size_t count = 0;
            const char *p = src;
            while (*p && count < n) {
                int mb = mbtowc(nullptr, p, MB_CUR_MAX);
                if (mb <= 0) { ret32((uint32_t)(size_t)-1); break; }
                p += mb; ++count;
            }
            ret32((uint32_t)count);
            break;
        }
        wchar_t *dst   = (wchar_t*)ctx.mem.ptr(r0);
        size_t   count = 0;
        while (count < n && *src) {
            wchar_t wc = 0;
            int mb = mbtowc(&wc, src, MB_CUR_MAX);
            if (mb <= 0) { ret32((uint32_t)(size_t)-1); goto mbsrtowcs_done; }
            dst[count] = wc;
            src += mb;
            ++count;
        }
        if (count < n) dst[count] = L'\0';
        
        if (!*src) {
            ctx.mem.write32(r1, 0u);
        } else {
            ctx.mem.write32(r1, src_va + (uint32_t)(src - src_org));
        }
        ret32((uint32_t)count);
        mbsrtowcs_done:
        break;
    }

    
    case SVC_LOGB: {
        /* logb(double x[r0:r1]) → double[r0:r1] */
        uint64_t bits = (uint64_t)r1 << 32 | r0;
        double v; memcpy(&v, &bits, 8);
        double res = logb(v);
        uint64_t rb; memcpy(&rb, &res, 8);
        regs[0] = (uint32_t)rb; regs[1] = (uint32_t)(rb >> 32);
        break;
    }
    case SVC_LRINTF: {
        /* lrintf(float x[r0]) → long/int[r0] */
        float f; uint32_t tmp = r0; memcpy(&f, &tmp, 4);
        ret32((uint32_t)(int32_t)lrintf(f));
        break;
    }
    case SVC_EXPM1F: {
        /* expm1f(float x[r0]) → float[r0] */
        float f; uint32_t tmp = r0; memcpy(&f, &tmp, 4);
        float res = expm1f(f);
        uint32_t rb; memcpy(&rb, &res, 4);
        ret32(rb);
        break;
    }
    case SVC_NANF: {
        /* nanf(const char* tagp[r0]) → float[r0] */
        const char *tag = r0 ? (const char*)ctx.mem.ptr(r0) : "";
        float res = std::nanf(tag ? tag : "");
        uint32_t rb; memcpy(&rb, &res, 4);
        ret32(rb);
        break;
    }
    case SVC_WMEMCMP: {
        /* wmemcmp(const wchar_t* s1[r0], const wchar_t* s2[r1], size_t n[r2]) → int
 * ARM32 wchar_t = 4 byte */
        if (!r0 || !r1 || !r2) { ret32(0); break; }
        int result = 0;
        for (uint32_t i = 0; i < r2 && result == 0; i++) {
            uint32_t c1 = ctx.mem.read32(r0 + i * 4);
            uint32_t c2 = ctx.mem.read32(r1 + i * 4);
            if (c1 < c2) result = -1;
            else if (c1 > c2) result = 1;
        }
        ret32((uint32_t)(int32_t)result);
        break;
    }
    case SVC_SWPRINTF: {
        /* swprintf(wchar_t* buf[r0], size_t maxlen[r1], const wchar_t* fmt[r2], ...) */
        if (!r0 || !r2) { ret32(-1); break; }
        
        std::string narrow_fmt;
        for (uint32_t va = r2; ; va += 4) {
            uint32_t wc = ctx.mem.read32(va);
            if (!wc) break;
            narrow_fmt += (wc < 128u) ? (char)wc : '?';
        }
        ArmVarArgs ap{ctx, regs, 3, regs[13], false};
        std::string s = arm_vformat(ctx, narrow_fmt.c_str(), ap);
        
        uint32_t maxn = r1 ? r1 : 1u;
        uint32_t written = 0;
        for (char c : s) {
            if (written >= maxn - 1) break;
            ctx.mem.write32(r0 + written * 4, (uint32_t)(unsigned char)c);
            written++;
        }
        ctx.mem.write32(r0 + written * 4, 0u);
        ret32(written);
        break;
    }
    case SVC_LOCALECONV: {
        /* localeconv() → struct lconv* (guest VA) */
        if (!g_lconv_va) {
            
            uint32_t base = arm_malloc(ctx, 64u);
            if (!base) { ret32(0); break; }
            
            uint32_t dot_va = base + 40u;  
            uint32_t emp_va = base + 44u;  
            ctx.mem.write32(dot_va, (uint32_t)'.');   /* '.' \0 \0 \0 (LE) */
            ctx.mem.write32(emp_va, 0u);               /* '\0' */
            
            /* [0] decimal_point */ ctx.mem.write32(base +  0, dot_va);
            /* [1] thousands_sep */ ctx.mem.write32(base +  4, emp_va);
            /* [2] grouping */ ctx.mem.write32(base +  8, emp_va);
            /* [3] int_curr_symbol */ ctx.mem.write32(base + 12, emp_va);
            /* [4] currency_symbol */ ctx.mem.write32(base + 16, emp_va);
            /* [5] mon_decimal_point */ ctx.mem.write32(base + 20, emp_va);
            /* [6] mon_thousands_sep */ ctx.mem.write32(base + 24, emp_va);
            /* [7] mon_grouping */ ctx.mem.write32(base + 28, emp_va);
            /* [8] positive_sign */ ctx.mem.write32(base + 32, emp_va);
            /* [9] negative_sign */ ctx.mem.write32(base + 36, emp_va);
            
            ctx.mem.write32(base + 48, 0x7F7F7F7Fu);
            ctx.mem.write32(base + 52, 0x7F7F7F7Fu);
            g_lconv_va = base;
        }
        ret32(g_lconv_va);
        break;
    }
    case SVC_SOCKETPAIR: {
        /* socketpair(int domain[r0], int type[r1], int protocol[r2], int sv[2][r3]) */
        if (!r3) { errno = EFAULT; ret32((uint32_t)-1); break; }
        int sv[2] = {-1, -1};
        int rc = socketpair((int)r0, (int)r1, (int)r2, sv);
        if (rc == 0) {
            ctx.mem.write32(r3,     (uint32_t)sv[0]);
            ctx.mem.write32(r3 + 4, (uint32_t)sv[1]);
        }
        ret32((uint32_t)(int32_t)rc);
        break;
    }
    case SVC_ACFG_SDKVER: {
        /* AConfiguration_getSdkVersion(AConfiguration* [r0]) → int32_t
 * Keep in sync with Build.VERSION.SDK_INT (31, jni_stubs.c) and
 * ro.build.version.sdk.  API 29+ makes Swappy use NDK AChoreographer. */
        (void)r0;
        ret32(31u);
        break;
    }
    case SVC_ACHOREOGRAPHER_GET: {
        /* AChoreographer_getInstance() → AChoreographer* */
        static int n = 0;
        if (n++ < 8)
            fprintf(stderr, "[achoreo] getInstance lr=0x%08x tid=%d\n",
                    regs[14], g_current_tid);
        ret32(ARM_ACHOREOGRAPHER);
        break;
    }
    case SVC_ACHOREOGRAPHER_POST: {
        /* void AChoreographer_postFrameCallback(ch[r0], cb[r1], data[r2]) */
        static int n = 0;
        if (n++ < 8)
            fprintf(stderr, "[achoreo] postFrameCallback cb=0x%08x data=0x%08x lr=0x%08x\n",
                    r1, r2, regs[14]);
        fire_choreographer_callback(ctx, r1, r2);
        break;
    }
    case SVC_ACHOREOGRAPHER_POST64: {
        /* void AChoreographer_postFrameCallback64(ch[r0], cb[r1], data[r2]) */
        static int n = 0;
        if (n++ < 8)
            fprintf(stderr, "[achoreo] postFrameCallback64 cb=0x%08x data=0x%08x lr=0x%08x\n",
                    r1, r2, regs[14]);
        fire_choreographer_callback(ctx, r1, r2);
        break;
    }
    case SVC_ACHOREOGRAPHER_POSTDELAY: {
        /* void AChoreographer_postFrameCallbackDelayed(ch[r0], cb[r1], data[r2], delay[r3]) */
        (void)r3;
        static int n = 0;
        if (n++ < 8)
            fprintf(stderr, "[achoreo] postFrameCallbackDelayed cb=0x%08x data=0x%08x lr=0x%08x\n",
                    r1, r2, regs[14]);
        fire_choreographer_callback(ctx, r1, r2);
        break;
    }
    case SVC_AASSETMGR_FROMJAVA: {
        /* AAssetManager_fromJava(JNIEnv* env[r0], jobject assetMgr[r1]) → AAssetManager* */
        (void)r0; (void)r1;
        ret32(0xA55E7FFFu); 
        break;
    }
    case SVC_AASSETMGR_OPEN: {
        /* AAssetManager_open(AAssetManager* mgr[r0], const char* filename[r1], int mode[r2]) */
        const char *fname = r1 ? (const char*)ctx.mem.ptr(r1) : nullptr;
        if (!fname) { ret32(0); break; }
        static const char *prefixes[] = {
            "assets/", "base/assets/", "UnityDataAssetPack/assets/", nullptr
        };
        std::vector<uint8_t> data;
        bool found = false;
        for (int pi = 0; prefixes[pi]; pi++) {
            std::string full = std::string(prefixes[pi]) + fname;
            if (asset_bridge::apk_read(full, data)) { found = true; break; }
        }
        if (!found) {
            fprintf(stderr, "[AAsset] open MISS %s\n", fname);
            ret32(0); break;
        }
        uint32_t buf_va = arm_malloc(ctx, (uint32_t)data.size() + 1u);
        if (!buf_va) { ret32(0); break; }
        uint8_t *dst = ctx.mem.ptr(buf_va);
        memcpy(dst, data.data(), data.size());
        dst[data.size()] = 0;
        uint32_t handle = g_next_asset_handle;
        g_next_asset_handle += 4u;
        g_assets[handle] = GuestAsset{ std::move(data), buf_va };
        fprintf(stderr, "[AAsset] open %s -> handle 0x%x (%zu bytes)\n",
                fname, handle, g_assets[handle].data.size());
        ret32(handle);
        break;
    }
    case SVC_AASSET_GETBUFFER: {
        /* AAsset_getBuffer(AAsset* asset[r0]) → const void* */
        auto it = g_assets.find(r0);
        ret32(it != g_assets.end() ? it->second.buf_va : 0u);
        break;
    }
    case SVC_AASSET_GETLENGTH: {
        /* AAsset_getLength(AAsset* asset[r0]) → off_t (32-bit) */
        auto it = g_assets.find(r0);
        ret32(it != g_assets.end() ? (uint32_t)it->second.data.size() : 0u);
        break;
    }

    case SVC_UNKNOWN_CALL: {
        static uint64_t s_unk_count = 0;
        uint64_t n = s_unk_count++;
        if (n < 30 || (n & 0xfff) == 0)
            fprintf(stderr, "[unresolved] stub called #%llu lr=0x%08x r0=%08x r1=%08x r2=%08x r3=%08x\n",
                    (unsigned long long)n, regs[14], r0, r1, r2, r3);
        ret32(0);
        break;
    }

    
    case SVC_FREXP: {
        /* frexp(double x [r0:r1], int* exp [r2]) → double in r0:r1 */
        uint64_t bits = (uint64_t)r1 << 32 | r0;
        double v; memcpy(&v, &bits, 8);
        int exp_out = 0;
        double frac = frexp(v, &exp_out);
        if (r2) ctx.mem.write32(r2, (uint32_t)(int32_t)exp_out);
        uint64_t rb; memcpy(&rb, &frac, 8);
        regs[0] = (uint32_t)rb; regs[1] = (uint32_t)(rb >> 32);
        break;
    }
    case SVC_RINT: {
        /* rint(double x [r0:r1]) → double in r0:r1 */
        uint64_t bits = (uint64_t)r1 << 32 | r0;
        double v; memcpy(&v, &bits, 8);
        double res = rint(v);
        uint64_t rb; memcpy(&rb, &res, 8);
        regs[0] = (uint32_t)rb; regs[1] = (uint32_t)(rb >> 32);
        break;
    }
    case SVC_LRAND48:
        ret32((uint32_t)lrand48());
        break;
    case SVC_SRAND48:
        srand48((long)(int32_t)r0);
        ret32(0);
        break;
    case SVC_STRPBRK: {
        const char* s1 = r0 ? (const char*)ctx.mem.ptr(r0) : nullptr;
        const char* s2 = r1 ? (const char*)ctx.mem.ptr(r1) : nullptr;
        if (!s1 || !s2) { ret32(0); break; }
        const char* p = strpbrk(s1, s2);
        ret32(p ? (uint32_t)(r0 + (p - s1)) : 0u);
        break;
    }
    case SVC_STRTOK: {
        
        char* s   = r0 ? (char*)ctx.mem.ptr(r0) : nullptr;
        const char* sep = r1 ? (const char*)ctx.mem.ptr(r1) : nullptr;
        if (!sep) { ret32(0); break; }
        char* tok = strtok(s ? s : nullptr, sep);
        if (!tok) { ret32(0); break; }
        
        ret32((uint32_t)(tok - (char*)ctx.mem.ptr(0)));
        break;
    }
    case SVC_DUP2:
        ret32((uint32_t)(int32_t)dup2((int)r0, (int)r1));
        break;
    case SVC_CLOCK_GETRES: {
        struct timespec ts{};
        int rc = clock_getres((clockid_t)r0, &ts);
        if (!rc && r1) {
            ctx.mem.write32(r1,     (uint32_t)ts.tv_sec);
            ctx.mem.write32(r1 + 4, (uint32_t)ts.tv_nsec);
        }
        ret32((uint32_t)(int32_t)rc);
        break;
    }
    case SVC_GETHOSTNAME: {
        char* buf = r0 ? (char*)ctx.mem.ptr(r0) : nullptr;
        if (!buf) { ret32((uint32_t)-1); break; }
        ret32((uint32_t)(int32_t)gethostname(buf, (size_t)r1));
        break;
    }
    case SVC_GETRUSAGE: {
        /* ARM32 rusage: timeval={u32,u32}×2 + 14×u32 = 72 bytes */
        struct rusage ru{};
        int rc = getrusage((int)r0, &ru);
        if (!rc && r1) {
            /* utime: sec, usec */
            ctx.mem.write32(r1 + 0,  (uint32_t)ru.ru_utime.tv_sec);
            ctx.mem.write32(r1 + 4,  (uint32_t)ru.ru_utime.tv_usec);
            ctx.mem.write32(r1 + 8,  (uint32_t)ru.ru_stime.tv_sec);
            ctx.mem.write32(r1 + 12, (uint32_t)ru.ru_stime.tv_usec);
            ctx.mem.write32(r1 + 16, (uint32_t)ru.ru_maxrss);
            
            for (int i = 20; i < 72; i += 4) ctx.mem.write32(r1 + i, 0);
        }
        ret32((uint32_t)(int32_t)rc);
        break;
    }
    case SVC_VSPRINTF2: {
        /* vsprintf(char* buf, const char* fmt, va_list ap) */
        if (r0 && r1) {
            ArmVarArgs ap{ctx, regs, 0, r2, true};
            ret32(arm_format_to(ctx, r0, UINT32_MAX, arm_vformat(ctx, ctx.mem.cstr(r1), ap)));
        } else {
            ret32(0);
        }
        break;
    }
    case SVC_GETC: {
        
        FILE* f = resolve_file(ctx, r0);
        ret32((uint32_t)(int32_t)(f ? fgetc(f) : EOF));
        break;
    }
    case SVC_PUTCHAR:
        ret32((uint32_t)(int32_t)putchar((int)r0));
        break;
    case SVC_FPCLASSIFYF: {
        /* __fpclassifyf(float x [r0]) → int
 * Android bionic uses different constants than glibc:
 * bionic: FP_INFINITE=1, FP_NAN=2, FP_NORMAL=4, FP_SUBNORMAL=8, FP_ZERO=16
 * glibc:  FP_NAN=0,      FP_INFINITE=1, FP_ZERO=2, FP_SUBNORMAL=3, FP_NORMAL=4
 * PhysX in libunity checks (result & 3) == 0 to pass; with bionic values
 * ZERO(16), NORMAL(4), SUBNORMAL(8) all pass while NaN(2) and Inf(1) fail. */
        float f; uint32_t tmp = r0; memcpy(&f, &tmp, 4);
        int cls = std::fpclassify(f);
        uint32_t bionic_cls;
        switch (cls) {
            case FP_NAN:       bionic_cls = 2;  break;
            case FP_INFINITE:  bionic_cls = 1;  break;
            case FP_ZERO:      bionic_cls = 16; break;
            case FP_SUBNORMAL: bionic_cls = 8;  break;
            default:           bionic_cls = 4;  break; /* FP_NORMAL */
        }
        ret32(bionic_cls);
        break;
    }
    case SVC_INET_ADDR: {
        const char* cp = r0 ? (const char*)ctx.mem.ptr(r0) : nullptr;
        ret32(cp ? (uint32_t)inet_addr(cp) : (uint32_t)INADDR_NONE);
        break;
    }
    case SVC_FCNTL2: {
        
        int rc = fcntl((int)r0, (int)r1, (int)r2);
        ret32((uint32_t)(int32_t)rc);
        break;
    }
    case SVC_G_FILENAME_URI: {
        /* mono_escape_uri_string: ensure paths are expressed as file:// URIs.
 * Decoding here breaks mono_image_open_full → g_filename_from_uri (+0x2b6cb4). */
        const char *inp = r0 ? ctx.mem.cstr(r0) : nullptr;
        if (!inp) { ret32(0); break; }
        std::string uri;
        if (strncmp(inp, "file:", 5) == 0) {
            const char *fpath = inp + 7; /* file:///abs/path */
            if (inp[5] != '/' || inp[6] != '/')
                fpath = inp + 5;
            std::string norm = fpath;
            mono_rewrite_corlib_path(norm);
            uri = "file://" + norm;
        } else {
            const char *p = inp;
            while (p[0] == '/' && p[1] == '/') ++p;
            std::string norm = (p[0] == '/') ? std::string(p) : std::string("/") + p;
            mono_rewrite_corlib_path(norm);
            uri = "file://" + norm;
        }
        uint32_t addr = arm_malloc(ctx, (uint32_t)(uri.size() + 1));
        if (addr) {
            memcpy(ctx.mem.ptr(addr), uri.c_str(), uri.size() + 1);
            if (getenv("LUNARIA_TRACE_OPEN"))
                fprintf(stderr, "[mono_escape_uri] %s -> %s\n", inp, uri.c_str());
            ret32(addr);
        } else {
            ret32(0);
        }
        break;
    }
    case SVC_G_FILENAME_FROM_URI: {
        /* g_filename_from_uri @ libmono+0x2b6cb4 (uri in r1): file:// → plain path. */
        const char *uri = r1 ? ctx.mem.cstr(r1) : nullptr;
        if (!uri) { ret32(0); break; }
        const char *fpath = uri;
        if (strncmp(uri, "file://", 7) == 0) {
            fpath = uri + 7;
            if (*fpath != '/') {
                fpath = strchr(fpath, '/');
                if (!fpath) { ret32(0); break; }
            }
        }
        uint32_t addr = arm_malloc(ctx, (uint32_t)(strlen(fpath) + 1));
        if (addr) {
            memcpy(ctx.mem.ptr(addr), fpath, strlen(fpath) + 1);
            if (getenv("LUNARIA_TRACE_OPEN"))
                fprintf(stderr, "[g_filename_from_uri] %s -> %s\n", uri, fpath);
            ret32(addr);
        } else {
            ret32(0);
        }
        break;
    }
    case SVC_MONO_PATH_NORM: {
        /* libmono path helpers: return "file:///" + normalized path so
 * try_open_assembly enters its fopen branch.  Do NOT realpath() here —
 * resolving mono/2.0/mscorlib.dll → ../../mscorlib.dll makes mono reject
 * the load because the path no longer sits under mono/2.0/. */
        const char *inp = r0 ? ctx.mem.cstr(r0) : nullptr;
        std::string norm;
        if (inp && inp[0] != '\0' && !strstr(inp, "://")) {
            while (inp[0] == '/' && inp[1] == '/')
                ++inp;
            const char *p = inp;
            bool first = true;
            while (*p) {
                while (*p == '/') ++p;
                if (!*p) break;
                const char *s = p;
                while (*p && *p != '/') ++p;
                if (!first) norm += '/';
                norm.append(s, (size_t)(p - s));
                first = false;
            }
            if (!norm.empty() && norm[0] != '/')
                norm.insert(norm.begin(), '/');
        } else if (inp) {
            norm = inp;
        }
        mono_rewrite_corlib_path(norm);
        std::string uri;
        if (!norm.empty() && norm.find("://") == std::string::npos) {
            uri = (norm[0] == '/') ? ("file://" + norm) : ("file:///" + norm);
        } else {
            uri = norm;
        }
        if (!uri.empty()) {
            uint32_t addr = arm_malloc(ctx, (uint32_t)(uri.size() + 1));
            if (addr) {
                memcpy(ctx.mem.ptr(addr), uri.c_str(), uri.size() + 1);
                if (getenv("LUNARIA_TRACE_OPEN"))
                    fprintf(stderr, "[mono_path_norm] %s -> %s\n", inp ? inp : "(null)", uri.c_str());
                ret32(addr);
                break;
            }
        }
        ret32(r0);
        break;
    }
    case SVC_MONO_FILE_MAP_OPEN: {
        /* Unity registers mono_file_map_override() to read assemblies from the
 * Android asset manager; that path is inert here, so fopen never runs and
 * mscorlib never loads.  This Unity libmono build's mono_file_map_open()
 * returns a bionic FILE* (fd at +0x0e), not a MonoFileMap wrapper. */
        const char *name = r0 ? ctx.mem.cstr(r0) : nullptr;
        if (!name) { ret32(0); break; }
        if (strncmp(name, "file://", 7) == 0) {
            name += 7;
            if (*name != '/') {
                const char *sl = strchr(name, '/');
                if (sl) name = sl;
            }
        }
        /* corlib must appear under mono/2.0/ or mono rejects the load */
        const char *open_name = name;
        char corlib_path[PATH_MAX];
        if (name && strstr(name, "/Managed/mscorlib.dll") && !strstr(name, "/mono/")) {
            const char *base = strstr(name, "/Managed/mscorlib.dll");
            size_t prefix = (size_t)(base - name);
            if (prefix < sizeof corlib_path - 32) {
                memcpy(corlib_path, name, prefix);
                corlib_path[prefix] = '\0';
                strncat(corlib_path, "/Managed/mono/2.0/mscorlib.dll",
                        sizeof corlib_path - prefix - 1);
                open_name = corlib_path;
            }
        }
        FILE *file = fopen(open_name, "rb");
        if (!file) {
            char cache[PATH_MAX];
            if (apk_extract_to_cache(open_name, cache, sizeof cache))
                file = fopen(cache, "rb");
        }
        if (!file && open_name != name) {
            file = fopen(name, "rb");
            if (!file) {
                char cache[PATH_MAX];
                if (apk_extract_to_cache(name, cache, sizeof cache))
                    file = fopen(cache, "rb");
            }
        }
        if (!file) {
            if (name && strstr(name, ".dll"))
                fprintf(stderr, "[mono_file_map_open] FAILED: %s\n", name);
            ret32(0);
            break;
        }
        uint32_t fshim = register_file(ctx, file);
        if (!fshim) { ret32(0); break; }
        if (name && strstr(name, ".dll")) {
            static int open_log = 0;
            if (open_log++ < 4)
                fprintf(stderr, "[mono_file_map_open] OK: %s shim=%#x fd=%d\n",
                        open_name, fshim, fileno(file));
        }
        ret32(fshim);
        break;
    }
    case SVC_MONO_FILE_MAP_FD: {
        uint32_t fshim = r0;
        if (!fshim) { ret32(~0u); break; }
        uint16_t fd16 = 0;
        if (uint8_t *p = (uint8_t*)ctx.mem.ptr(fshim))
            memcpy(&fd16, p + 14, 2);
        FILE *f = resolve_file(ctx, fshim);
        int fd = f ? fileno(f) : (int)fd16;
        static int fd_log = 0;
        if (fd_log++ < 6)
            fprintf(stderr, "[mono_file_map_fd] shim=%#x fd16=%u host=%d lr=0x%08x\n",
                    fshim, (unsigned)fd16, fd, regs[14]);
        ret32((uint32_t)fd);
        break;
    }
    case SVC_MONO_FILE_MAP_SIZE: {
        uint32_t fshim = r0;
        if (!fshim) { ret32(0); break; }
        FILE *f = resolve_file(ctx, fshim);
        if (!f) { ret32(0); break; }
        struct stat st;
        if (fstat(fileno(f), &st) != 0) { ret32(0); break; }
        static int sz_log = 0;
        if (sz_log++ < 4)
            fprintf(stderr, "[mono_file_map_size] -> %lld lr=0x%08x\n",
                    (long long)st.st_size, regs[14]);
        ret64((uint64_t)st.st_size);
        break;
    }
    case SVC_MONO_FILE_MAP: {
        /* mono_file_map(size_t length, int flags, int fd, offset, void **ret_handle) */
        uint32_t length = r0;
        int fd = (int)(int32_t)r2;
        uint32_t sp = regs[13];
        uint32_t off_lo = sp ? ctx.mem.read32(sp) : 0u;
        uint32_t off_hi = sp ? ctx.mem.read32(sp + 4u) : 0u;
        uint32_t ret_handle = sp ? ctx.mem.read32(sp + 8u) : 0u;
        uint64_t offset = ((uint64_t)off_hi << 32) | off_lo;
        static int map_log = 0;
        if (map_log++ < 8)
            fprintf(stderr, "[mono_file_map] len=%u fd=%d off=%llu ret=%#x lr=0x%08x\n",
                    length, fd, (unsigned long long)offset, ret_handle, regs[14]);
        if (!length) { ret32(0); break; }
        uint32_t addr = mmap_bump(length);
        if (addr == ~0u) { ret32(0); break; }
        if (fd >= 0) {
            
            uint8_t *dst = ctx.mem.ptr(addr);
            size_t done = 0;
            while (done < length) {
                ssize_t n = pread(fd, dst + done, length - done,
                                  (off_t)(offset + done));
                if (n <= 0) break;
                done += (size_t)n;
            }
            if (done != length)
                fprintf(stderr, "[mono_file_map] SHORT READ %zu/%u fd=%d off=%llu\n",
                        done, length, fd, (unsigned long long)offset);
        }
        if (map_log <= 8) {
            uint32_t magic = ctx.mem.read32(addr);
            fprintf(stderr, "[mono_file_map] -> %#x MZ=%c%c\n", addr,
                    (char)(magic & 0xff), (char)((magic >> 8) & 0xff));
        }
        if (ret_handle) {
            ctx.mem.write32(ret_handle, addr);
            if (getenv("LUNARIA_TRACE_OPEN") && map_log <= 8)
                fprintf(stderr, "[mono_file_map] *handle@%#x = %#x ret=%#x\n",
                        ret_handle, addr, addr);
        }
        ret32(addr);
        break;
    }
    case SVC_MONO_FILE_MAP_CLOSE: {
        /* mono_file_map_close(FILE*): close host fd behind guest shim. */
        FILE *f = resolve_file(ctx, r0);
        if (f) {
            for (uint32_t k = 1; k < 256; ++k)
                if (g_file_tab[k] == f) { g_file_tab[k] = nullptr; break; }
            fclose(f);
        }
        if (r0 >= 256u) arm_free(ctx, r0);
        ret32(0);
        break;
    }
    case SVC_MONO_PREP: {
        /* Side-effect only: register libunity machine.config before mjiv (regs preserved). */
        arm_exec_prepare_mono_config();
        break;
    }

    case SVC_GETRLIMIT:
        
        if (r1) {
            uint32_t v = (r0 == 3 /* RLIMIT_STACK */) ? STACK_SIZE : 0x40000000u;
            ctx.mem.write32(r1, v);
            ctx.mem.write32(r1 + 4, v);
        }
        ret32(0);
        break;
    case SVC_MREMAP: {
        
        if (r2 <= r1) { ret32(r0); break; }
        uint32_t p = arm_malloc(ctx, r2);
        if (!p) { ret32(~0u); break; }
        if (r0 && r1) memcpy(ctx.mem.ptr(p), ctx.mem.ptr(r0), r1);
        ret32(p);
        break;
    }
    case SVC_STRERROR: {
        static uint32_t err_static = 0;
        if (!err_static) err_static = arm_malloc(ctx, 64);
        if (err_static)
            snprintf((char*)ctx.mem.ptr(err_static), 64, "error %d", (int)r0);
        ret32(err_static);
        break;
    }
    case SVC_SETLOCALE: {
        static uint32_t loc_static = 0;
        if (!loc_static) {
            loc_static = arm_malloc(ctx, 2);
            if (loc_static) strcpy((char*)ctx.mem.ptr(loc_static), "C");
        }
        ret32(loc_static);
        break;
    }
    case SVC_RETM1: ret32(~0u); break;
    /* pipe(fds[2]) / pipe2(fds[2], flags) — host-backed; guest fds are host fds */
    case SVC_PIPE:
    case SVC_PIPE2: {
        if (!r0) { ret32(~0u); break; }
        int hostflags = 0;
        if (svc_no == SVC_PIPE2) {
            if (r1 & 0x800u)   hostflags |= O_NONBLOCK; /* bionic O_NONBLOCK */
            if (r1 & 0x80000u) hostflags |= O_CLOEXEC;  /* bionic O_CLOEXEC */
        }
        int hfds[2];
        int rc = pipe2(hfds, hostflags);
        if (rc == 0) {
            ctx.mem.write32(r0,     (uint32_t)hfds[0]);
            ctx.mem.write32(r0 + 4, (uint32_t)hfds[1]);
            fprintf(stderr, "[arm_exec] pipe%s → fds=[%d,%d] flags=0x%x tid=%u\n",
                    svc_no == SVC_PIPE2 ? "2" : "", hfds[0], hfds[1],
                    (unsigned)(svc_no == SVC_PIPE2 ? r1 : 0), g_current_tid);
        }
        ret32((uint32_t)rc);
        break;
    }
    case SVC_PTHREAD_GETATTR_NP: {
        /* attr (bionic): {u32 flags; void* stack_base; size_t stack_size; ...} */
        uint32_t base = STACK_BASE, size = STACK_SIZE;
        for (auto &t : g_threads)
            if (t.id == r0) { base = t.stack_base; size = THREAD_STACK_SIZE; break; }
        if (r1) {
            ctx.mem.write32(r1, 0);
            ctx.mem.write32(r1 + 4, base);
            ctx.mem.write32(r1 + 8, size);
            ctx.mem.write32(r1 + 12, 0);
        }
        ret32(0);
        break;
    }
    case SVC_PTHREAD_ATTR_GETSTACK:
        if (r1) ctx.mem.write32(r1, r0 ? ctx.mem.read32(r0 + 4) : STACK_BASE);
        if (r2) ctx.mem.write32(r2, r0 ? ctx.mem.read32(r0 + 8) : STACK_SIZE);
        ret32(0);
        break;
    case SVC_PTHREAD_ATTR_GETSTACKSZ:
        if (r1) {
            uint32_t sz = r0 ? ctx.mem.read32(r0 + 8) : STACK_SIZE;
            ctx.mem.write32(r1, sz ? sz : STACK_SIZE);
        }
        ret32(0);
        break;
    case SVC_VSNPRINTF_CHK: {
        /* __vsnprintf_chk(buf, size, flags, slen, fmt@sp0, va@sp4) */
        uint32_t fmt_va = ctx.mem.read32(regs[13]);
        uint32_t ap_va  = ctx.mem.read32(regs[13] + 4);
        ArmVarArgs ap{ctx, regs, 0, ap_va, true};
        ret32(arm_format_to(ctx, r0, r1, arm_vformat(ctx, ctx.mem.cstr(fmt_va), ap)));
        break;
    }
    case SVC_VSPRINTF_CHK: {
        /* __vsprintf_chk(buf, flags, slen, fmt, va@sp0) */
        uint32_t ap_va = ctx.mem.read32(regs[13]);
        ArmVarArgs ap{ctx, regs, 0, ap_va, true};
        ret32(arm_format_to(ctx, r0, UINT32_MAX, arm_vformat(ctx, ctx.mem.cstr(r3), ap)));
        break;
    }
    case SVC_SSCANF: {
        ArmVarArgs ap{ctx, regs, 2, regs[13], false};
        ret32(arm_vsscanf(ctx, ctx.mem.cstr(r0), ctx.mem.cstr(r1), ap));
        break;
    }
    case SVC_VSSCANF: {
        ArmVarArgs ap{ctx, regs, 0, r2, true};
        ret32(arm_vsscanf(ctx, ctx.mem.cstr(r0), ctx.mem.cstr(r1), ap));
        break;
    }

    default:
        /* ---- math passthrough blocks (softfp ABI) ---- */
        if (svc_no >= SVC_MATH_F1_BASE && svc_no < SVC_MATH_F1_BASE + SVC_MATH_F1_COUNT) {
            float v = kMathF1[svc_no - SVC_MATH_F1_BASE].second(rf(r0));
            uint32_t bits; memcpy(&bits, &v, 4); ret32(bits); break;
        }
        if (svc_no >= SVC_MATH_F2_BASE && svc_no < SVC_MATH_F2_BASE + SVC_MATH_F2_COUNT) {
            float v = kMathF2[svc_no - SVC_MATH_F2_BASE].second(rf(r0), rf(r1));
            uint32_t bits; memcpy(&bits, &v, 4); ret32(bits); break;
        }
        if (svc_no >= SVC_MATH_D1_BASE && svc_no < SVC_MATH_D1_BASE + SVC_MATH_D1_COUNT) {
            uint64_t in = (uint64_t)r0 | ((uint64_t)r1 << 32);
            double a; memcpy(&a, &in, 8);
            double v = kMathD1[svc_no - SVC_MATH_D1_BASE].second(a);
            uint64_t bits; memcpy(&bits, &v, 8); ret64(bits); break;
        }
        if (svc_no >= SVC_MATH_D2_BASE && svc_no < SVC_MATH_D2_BASE + SVC_MATH_D2_COUNT) {
            uint64_t ia = (uint64_t)r0 | ((uint64_t)r1 << 32);
            uint64_t ib = (uint64_t)r2 | ((uint64_t)r3 << 32);
            double a, b; memcpy(&a, &ia, 8); memcpy(&b, &ib, 8);
            double v = kMathD2[svc_no - SVC_MATH_D2_BASE].second(a, b);
            uint64_t bits; memcpy(&bits, &v, 8); ret64(bits); break;
        }
        fprintf(stderr, "[arm_exec] unhandled SVC #%u (pc=0x%08x)\n",
                svc_no, regs[15]);
        ret32(0);
        break;
    }

#undef AS_CLASS
#undef AS_OBJ
#undef AS_MID
#undef AS_FID
#undef AS_STR
#undef AS_ARR
#undef RET_OBJ
#undef ARM_STR
}

/* -------------------------------------------------------------------------
 * ---------------------------------------------------------------------- */
class StubCoprocessor final : public Dynarmic::A32::Coprocessor {
    static std::uint64_t NopCb(void*, std::uint32_t, std::uint32_t) { return 0; }
    static std::uint64_t TlsCb(void*, std::uint32_t, std::uint32_t) {
        if (!g_ctx) return 0;
        static std::map<uint32_t, uint32_t> tls_blocks;
        auto it = tls_blocks.find(g_current_tid);
        if (it != tls_blocks.end()) return it->second;
        uint32_t va = arm_malloc(*g_ctx, 4096);
        tls_blocks[g_current_tid] = va;
        return va;
    }
    static Callback nop() { return Callback{NopCb, std::nullopt}; }

public:
    std::optional<Callback> CompileInternalOperation(bool, unsigned, Dynarmic::A32::CoprocReg,
            Dynarmic::A32::CoprocReg, Dynarmic::A32::CoprocReg, unsigned) override
        { return nop(); }
    CallbackOrAccessOneWord CompileSendOneWord(bool, unsigned, Dynarmic::A32::CoprocReg,
            Dynarmic::A32::CoprocReg, unsigned) override
        { return nop(); }
    CallbackOrAccessTwoWords CompileSendTwoWords(bool, unsigned, Dynarmic::A32::CoprocReg) override
        { return nop(); }
    CallbackOrAccessOneWord CompileGetOneWord(bool, unsigned opc1, Dynarmic::A32::CoprocReg CRn,
            Dynarmic::A32::CoprocReg CRm, unsigned opc2) override {
        using CR = Dynarmic::A32::CoprocReg;
        if (opc1 == 0 && CRn == CR::C13 && CRm == CR::C0 && opc2 == 3)
            return Callback{TlsCb, std::nullopt};
        return nop();
    }
    CallbackOrAccessTwoWords CompileGetTwoWords(bool, unsigned, Dynarmic::A32::CoprocReg) override
        { return nop(); }
    std::optional<Callback> CompileLoadWords(bool, bool, Dynarmic::A32::CoprocReg,
            std::optional<std::uint8_t>) override
        { return nop(); }
    std::optional<Callback> CompileStoreWords(bool, bool, Dynarmic::A32::CoprocReg,
            std::optional<std::uint8_t>) override
        { return nop(); }
};


static void install_stub_coprocessors(Dynarmic::A32::UserConfig &cfg) {
    static auto stub = std::make_shared<StubCoprocessor>();
    for (size_t i = 0; i < cfg.coprocessors.size(); ++i)
        if (i != 10 && i != 11)
            cfg.coprocessors[i] = stub;
}

/* -------------------------------------------------------------------------
 * dynarmic callbacks
 * ---------------------------------------------------------------------- */
class ArmCallbacks : public Dynarmic::A32::UserCallbacks {
public:
    ArmExecCtx          *ctx;
    Dynarmic::A32::Jit  *jit = nullptr;
    uint64_t             ticks = 0;
    uint64_t             ticks_limit = UINT64_MAX;
    uint32_t             last_svc = UINT32_MAX;
    bool                 periodic_yield = false; /* force Run() return every 256M ticks */

    std::optional<uint32_t> MemoryReadCode(uint32_t va) override {
        /* Sentinel return address: halt the JIT (function returned) */
        if (pc_in_sentinel(va)) return {};
        /* LUNARIA_TRACE_PC=hexva[,hexva...]: report when a watched address is
         * first compiled (== first reached; JIT blocks are compiled lazily). */
        {
            static const std::set<uint32_t> trace_pcs = [] {
                std::set<uint32_t> s;
                if (const char *e = getenv("LUNARIA_TRACE_PC")) {
                    char *dup = strdup(e);
                    for (char *tok = strtok(dup, ","); tok; tok = strtok(nullptr, ","))
                        s.insert((uint32_t)strtoul(tok, nullptr, 16) & ~1u);
                    free(dup);
                }
                return s;
            }();
            if (!trace_pcs.empty() && trace_pcs.count(va & ~1u))
                fprintf(stderr, "[trace_pc] reached 0x%08x tid=%u\n", va & ~1u, g_current_tid);
        }
        /* LUNARIA_TRACE_CCOV=lo:hi — first-compile coverage: log each block
         * start compiled in [lo,hi).  Detects block starts as a gap in the
         * sequential code-fetch stream (compilation is lazy per block, so
         * coverage ≈ execution path). */
        {
            static uint32_t ccov_lo = 0, ccov_hi = 0, ccov_left = 0;
            static uint32_t ccov_prev = 0;
            static bool ccov_init = false;
            if (!ccov_init) {
                ccov_init = true;
                if (const char *e = getenv("LUNARIA_TRACE_CCOV")) {
                    sscanf(e, "%x:%x", &ccov_lo, &ccov_hi);
                    ccov_left = 2000;
                }
            }
            if (ccov_hi && ccov_left) {
                uint32_t a = va & ~1u;
                if (a >= ccov_lo && a < ccov_hi) {
                    if (a != ccov_prev + 2 && a != ccov_prev + 4 && a != ccov_prev) {
                        fprintf(stderr, "[ccov] blk=0x%08x tid=%u\n", a, g_current_tid);
                        --ccov_left;
                    }
                    ccov_prev = a;
                }
            }
        }
        /* Null-pointer call guard: nothing legitimate executes below 0x1000
 * (the flat arena would otherwise happily "execute" the ELF header) */
        if (va < 0x1000u) {
            static int nc = 0;
            if (nc++ < 10 && jit) {
                auto &r = jit->Regs();
                uint32_t sp = r[13];
                /* The thunk saves {R4,LR} before the indirect call, so [SP+4]
 * holds the caller's return address (the instruction after BL thunk). */
                uint32_t stk0 = (sp >= 4 && ctx->mem.ptr(sp))   ? ctx->mem.read32(sp)   : 0;
                uint32_t stk4 = (sp+4 < 0x50000000u && ctx->mem.ptr(sp+4)) ? ctx->mem.read32(sp+4) : 0;
                uint32_t stk8 = (sp+8 < 0x50000000u && ctx->mem.ptr(sp+8)) ? ctx->mem.read32(sp+8) : 0;
                fprintf(stderr, "[arm_exec] NULL call: pc=0x%08x lr=0x%08x "
                        "r0=0x%08x r1=0x%08x r2=0x%08x sp=0x%08x — halting\n"
                        "[arm_exec]   stack: [sp+0]=0x%08x [sp+4]=0x%08x [sp+8]=0x%08x "
                        "(caller_ret likely=0x%08x)\n",
                        va, r[14], r[0], r[1], r[2], r[13],
                        stk0, stk4, stk8, stk4);
                fprintf(stderr, "[arm_exec]   r3=0x%08x r4=0x%08x r5=0x%08x r6=0x%08x "
                        "r7=0x%08x r8=0x%08x r9=0x%08x r10=0x%08x r11=0x%08x r12=0x%08x\n",
                        r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12]);
                
                if (r[11] >= 0x40u && r[11] < 0xfff00000u && ctx->mem.ptr(r[11]))
                    for (int k = -8; k <= 0; ++k)
                        fprintf(stderr, "[arm_exec]   [r11%+d]=0x%08x\n", k*4,
                                ctx->mem.read32(r[11] + k*4));
                
                {
                    uint32_t fp = r[11];
                    for (int depth = 0; depth < 12 &&
                         fp >= 0x1000u && fp < 0xfff00000u && ctx->mem.ptr(fp); ++depth) {
                        uint32_t ret = ctx->mem.read32(fp);
                        uint32_t next = ctx->mem.read32(fp - 4);
                        fprintf(stderr, "[arm_exec]   bt#%d ret=0x%08x fp=0x%08x\n",
                                depth, ret, fp);
                        if (next <= fp) break;
                        fp = next;
                    }
                }
                /* Stack-scan backtrace: any odd word pointing into loaded code
                 * is a plausible Thumb return address; prints the call chain
                 * even without reliable frame pointers. */
                for (uint32_t o = 0; o < 0x200 && ctx->mem.ptr(sp + o); o += 4) {
                    uint32_t w = ctx->mem.read32(sp + o);
                    if ((w & 1) && w >= 0x000a0000u && w < 0x03400000u)
                        fprintf(stderr, "[arm_exec]   scan [sp+0x%03x]=0x%08x\n", o, w);
                }
            }
            return {};
        }
        /* Trampoline region: compute SVC #n / BX LR from VA instead of reading
 * host memory.  Guards against any guest code accidentally zeroing the
 * trampoline page (e.g. a large memset that crosses TRAMP_BASE). */
        if (va >= TRAMP_BASE && va < TRAMP_BASE + SVC_TRAMP_TOTAL * TRAMP_STRIDE) {
            uint32_t off = va - TRAMP_BASE;
            uint32_t n   = off / TRAMP_STRIDE;
            uint32_t pos = off % TRAMP_STRIDE;
            if (pos == 0) return 0xEF000000u | n;   /* svc #n */
            if (pos == 4) return 0xE12FFF1Eu;        /* bx lr */
            return {};
        }
        uint32_t v; std::memcpy(&v, ctx->mem.ptr(va), 4);
        /* Zero-word fetch = PC fell into zeroed data (heap/BSS).  Letting the
 * JIT compile multi-MB "andeq sleds" exhausts host memory in seconds. */
        if (v == 0) {
            /* Lazy generic-trampoline fill: only mono_trampoline_code[0..11] exist. */
            if (g_mono_base && va >= MMAP_BASE && va < MMAP_END) {
                const uint32_t tc = mono_sym_va("mono_trampoline_code");
                if (tc) {
                    /* JIT-compiled methods `bl` straight into a generic trampoline
 * (mono_trampoline_code[t]) with a PC-relative branch computed
 * one word short of the recorded base — every such call lands
 * exactly 4 bytes before the trampoline's real first
 * instruction, in the empty gap the code manager left before
 * it, so the fetch here reads zero and halts.  Bridge that
 * single missing word with a synthetic forward branch so
 * control reaches the trampoline's true entry (fetches from
 * there proceed normally, off the correct PC-relative base). */
                    for (uint32_t t = 0; t < 12u; ++t) {
                        uint32_t a = ctx->mem.read32(tc + t * 4u);
                        if (a && va == a - 4u)
                            return 0xEAFFFFFFu; /* b va+4 (i.e. b <trampoline entry>) */
                    }
                    uint32_t hit = ~0u;
                    for (uint32_t t = 0; t < 12u; ++t) {
                        uint32_t a = ctx->mem.read32(tc + t * 4u);
                        if (a && va >= a && va < a + 0xc8u) { hit = t; break; }
                    }
                    if (hit != ~0u && hit <= 11u && !ctx->mem.read32(tc + hit * 4u)) {
                        const uint32_t create = mono_fn_va(nullptr, 0x12d4bcu);
                        if (create) {
                            uint32_t addr = (uint32_t)run_arm(*ctx, create, hit, 0, 0, 0,
                                                              50'000'000ULL);
                            if (addr)
                                ctx->mem.write32(tc + hit * 4u, addr);
                        }
                        if (jit) jit->InvalidateCacheRange(va & ~3u, 16);
                        std::memcpy(&v, ctx->mem.ptr(va), 4);
                        if (v) return v;
                    }
                }
            }
            static int zf = 0;
            if (zf++ < 10 && jit) {
                auto &r = jit->Regs();
                fprintf(stderr, "[arm_exec] zero-instruction fetch at 0x%08x — "
                        "halting (wild jump?) lr=0x%08x r0=0x%08x r1=0x%08x "
                        "r2=0x%08x r3=0x%08x r4=0x%08x r12=0x%08x sp=0x%08x\n",
                        va, r[14], r[0], r[1], r[2], r[3], r[4], r[12], r[13]);
                fprintf(stderr, "[arm_exec]   r5=0x%08x r6=0x%08x r7=0x%08x "
                        "r8=0x%08x r9=0x%08x r10=0x%08x r11=0x%08x\n",
                        r[5], r[6], r[7], r[8], r[9], r[10], r[11]);
                /* Dump return-address candidates from the guest stack so the
 * faulting call chain can be reconstructed offline. */
                for (int i = 0; i < 64; ++i) {
                    uint32_t w; std::memcpy(&w, ctx->mem.ptr(r[13] + i*4), 4);
                    if (w >= 0x1000u && w < 0x02000000u && (w & 1u))
                        fprintf(stderr, "[arm_exec]   stack[%02d]=0x%08x (ret?)\n", i, w);
                }
                
                if (r[5] > 0x1000u && r[5] < 0x02000000u) {
                    uint32_t t = r[5];
                    uint8_t flag = *ctx->mem.ptr(t + 0xc1f);
                    uint32_t main_alloc; std::memcpy(&main_alloc, ctx->mem.ptr(t + 0xc38), 4);
                    int nz_a74 = 0, nz_1d14 = 0;
                    for (int i = 0; i <= 0xa4; ++i) {
                        uint32_t w1; std::memcpy(&w1, ctx->mem.ptr(t + 0xa74 + i*4), 4);
                        uint32_t w2; std::memcpy(&w2, ctx->mem.ptr(t + 0x1d14 + i*16), 4);
                        if (w1 > 0x200u) ++nz_a74;
                        if (w2 > 0x200u) ++nz_1d14;
                    }
                    uint32_t lbl; std::memcpy(&lbl, ctx->mem.ptr(t + 0x1d14 + r[9]*16), 4);
                    fprintf(stderr, "[arm_exec]   mm@0x%08x flag(c1f)=%u main(c38)=0x%08x "
                            "tab_a74_nz=%d tab_1d14_nz=%d entry[label=0x%x]=0x%08x\n",
                            t, flag, main_alloc, nz_a74, nz_1d14, lbl, lbl);
                }
            }
            return {};
        }
        /* LUNARIA_WATCH_EXEC: comma-separated guest VAs — log when the JIT first
 * compiles the instruction at that address (i.e. execution reached it).
 * Blocks are cached, so each address logs on first reach (and after any
 * cache invalidation) — enough to map which branches of a guest control
 * flow are ever taken. */
        static const std::vector<uint32_t> exec_watch = [] {
            std::vector<uint32_t> w;
            if (const char *e = getenv("LUNARIA_WATCH_EXEC")) {
                char *dup = strdup(e);
                for (char *tok = strtok(dup, ","); tok; tok = strtok(nullptr, ","))
                    w.push_back((uint32_t)strtoul(tok, nullptr, 0));
                free(dup);
            }
            return w;
        }();
        for (uint32_t wa : exec_watch) {
            if (wa != va) continue;
            if (jit) {
                auto &r = jit->Regs();
                fprintf(stderr, "[watch-exec] reached va=0x%08x (lr~0x%08x r0=0x%08x "
                        "r1=0x%08x r2=0x%08x r3=0x%08x r4=0x%08x r11=0x%08x sp=0x%08x)\n",
                        va, (uint32_t)r[14], (uint32_t)r[0], (uint32_t)r[1],
                        (uint32_t)r[2], (uint32_t)r[3], (uint32_t)r[4],
                        (uint32_t)r[11], (uint32_t)r[13]);
            } else {
                fprintf(stderr, "[watch-exec] reached va=0x%08x\n", va);
            }
            break;
        }
        return v;
    }
    uint8_t  MemoryRead8 (uint32_t va) override { return *ctx->mem.ptr(va); }
    uint16_t MemoryRead16(uint32_t va) override {
        uint16_t v; memcpy(&v, ctx->mem.ptr(va), 2); return v; }
    uint32_t MemoryRead32(uint32_t va) override {
        uint32_t v; memcpy(&v, ctx->mem.ptr(va), 4); return v; }
    uint64_t MemoryRead64(uint32_t va) override {
        uint64_t v; memcpy(&v, ctx->mem.ptr(va), 8); return v; }

    void watch_hit(uint32_t va, uint32_t v, uint32_t bytes) {
        if (g_wrange_hi)
            wrange_log(va, v, bytes, jit ? (uint32_t)jit->Regs()[15] : 0,
                       jit ? (uint32_t)jit->Regs()[14] : 0, "store");
        if (g_watch_hi == 0 || va >= g_watch_hi || va + bytes <= g_watch_lo)
            return;
        auto &R = jit->Regs();
        /* dynarmic only writes back R[15] at block boundaries, so the PC visible
 * inside a memory callback is the (stale) block-entry address — never the
 * storing instruction.  We therefore do NOT report PC here.  Instead, the
 * first hit records the exact word being clobbered, arms single-step, and
 * halts; run_arm's step loop then polls that word between single-steps,
 * where R[15] *is* valid, to pin down the precise faulting instruction. */
        if (!g_step_mode) {
            fprintf(stderr, "[watch] first hit va=0x%08x val=0x%08x (stale block-pc=0x%08x) "
                    "— arming single-step poll\n", va, v, (uint32_t)R[15]);
            g_step_mode       = true;
            g_watch_poll_addr = va & ~3u;   /* word-align the polled address */
            jit->HaltExecution();
            return;
        }
        /* Already stepping: keep the watched range live (re-hits are expected as
 * the fill loop keeps running) but let run_arm's poll do the reporting. */
    }
    void MemoryWrite8 (uint32_t va, uint8_t  v) override { watch_hit(va,v,1); *ctx->mem.ptr(va) = v; }
    void MemoryWrite16(uint32_t va, uint16_t v) override {
        watch_hit(va,v,2); memcpy(ctx->mem.ptr(va), &v, 2); }
    void MemoryWrite32(uint32_t va, uint32_t v) override {
        watch_hit(va,v,4); memcpy(ctx->mem.ptr(va), &v, 4); }
    void MemoryWrite64(uint32_t va, uint64_t v) override {
        watch_hit(va,(uint32_t)v,8); memcpy(ctx->mem.ptr(va), &v, 8); }

    /* strex/ldrex: dynarmic passes the value observed at ldrex time as
     * `expected`; the callback must only store when memory still holds that
     * value and report failure otherwise.  Guest threads are scheduled
     * cooperatively, so a thread can be descheduled between the ldrex block
     * and the strex block (they end up in different dynarmic basic blocks
     * when a branch sits between them, e.g. Baselib's CAS loops).  The old
     * unconditional-store implementation let such interleaved CASes succeed
     * on stale values, corrupting lock/semaphore words. */
    template<typename T>
    bool write_exclusive(uint32_t va, T v, T expected) {
        uint8_t *p = ctx->mem.ptr(va);
        bool ok;
        if (((uintptr_t)p & (sizeof(T) - 1)) == 0) {
            auto *a = reinterpret_cast<std::atomic<T>*>(p);
            ok = a->compare_exchange_strong(expected, v);
        } else {
            T cur; memcpy(&cur, p, sizeof(T));
            ok = (cur == expected);
            if (ok) memcpy(p, &v, sizeof(T));
        }
        if (g_xcl_hi && va + sizeof(T) > g_xcl_lo && va < g_xcl_hi) {
            static int xn = 0;
            if (xn++ < 200)
                fprintf(stderr, "[xcl] va=0x%08x exp=0x%08x new=0x%08x ok=%d "
                        "tid=%u lr~0x%08x\n", va, (uint32_t)expected, (uint32_t)v,
                        (int)ok, g_current_tid,
                        jit ? (uint32_t)jit->Regs()[14] : 0);
        }
        if (ok) watch_hit(va, (uint32_t)v, sizeof(T));
        return ok;
    }
    bool MemoryWriteExclusive8 (uint32_t va,uint8_t  d,uint8_t  e) override
        {return write_exclusive<uint8_t >(va,d,e);}
    bool MemoryWriteExclusive16(uint32_t va,uint16_t d,uint16_t e) override
        {return write_exclusive<uint16_t>(va,d,e);}
    bool MemoryWriteExclusive32(uint32_t va,uint32_t d,uint32_t e) override
        {return write_exclusive<uint32_t>(va,d,e);}
    bool MemoryWriteExclusive64(uint32_t va,uint64_t d,uint64_t e) override
        {return write_exclusive<uint64_t>(va,d,e);}

    void CallSVC(uint32_t svc_no) override {
        auto &regs = jit->Regs();
        auto arr = regs;
        last_svc = svc_no;
        if (trace_svcs) {
            fprintf(stderr, "[svc] %u r0=0x%x r1=0x%x r2=0x%x r3=0x%x lr=0x%x\n",
                    svc_no, arr[0], arr[1], arr[2], arr[3], arr[14]);
            if (++trace_svc_count >= 5000) trace_svcs = false;
        }
        svc_ring_record(svc_no, arr[14], arr[0]);
        /* Watch a guest word for changes at SVC granularity (LUNARIA_WATCH_FLAG=0x<addr>).
 * Used to localize who writes the engine "initialized" flag. */
        if (g_watch_flag_va) {
            uint32_t v = ctx->mem.read32(g_watch_flag_va);
            if (v != g_watch_flag_last) {
                fprintf(stderr,"[watchflag] 0x%08x: 0x%08x -> 0x%08x (around svc=%u lr=0x%08x)\n",
                        g_watch_flag_va, g_watch_flag_last, v, svc_no, arr[14]);
                g_watch_flag_last = v;
            }
        }
        dispatch_svc(*ctx, svc_no, arr);
        regs = arr;
        /* thread blocked (e.g. sem_wait): yield this thread's slice */
        if (g_yield_requested) {
            g_yield_requested = false;
            static uint64_t yield_n = 0;
            if (yield_n++ < 8)
                fprintf(stderr, "[halt] g_yield_requested svc=%u tid=%u lr=0x%08x #%llu\n",
                        svc_no, g_current_tid, arr[14], (unsigned long long)(yield_n-1));
            if (jit) jit->HaltExecution();
        }
    }
    bool trace_svcs = (getenv("LUNARIA_TRACE_SVC") != nullptr);
    int  trace_svc_count = 0;

    void InterpreterFallback(uint32_t pc, size_t n) override {
        uint32_t instr = 0;
        if (auto p = ctx->mem.ptr(pc)) std::memcpy(&instr, p, 4);
        fprintf(stderr,"[arm_exec] InterpreterFallback pc=0x%08x n=%zu instr=0x%08x\n",
                pc, n, instr);
        if(jit) jit->HaltExecution();
    }
    void ExceptionRaised(uint32_t pc, Dynarmic::A32::Exception ex) override {
        using E = Dynarmic::A32::Exception;
        if(ex==E::NoExecuteFault) {
            if(!pc_in_sentinel(pc)) {
                static int nef = 0;
                if (nef++ < 4 && jit) {
                    uint32_t sp = jit->Regs()[13];
                    fprintf(stderr, "[arm_exec] NoExecuteFault at pc=0x%08x "
                            "lr=0x%08x r0=0x%08x r1=0x%08x r2=0x%08x r3=0x%08x sp=0x%08x — halting\n",
                            pc, jit->Regs()[14], jit->Regs()[0], jit->Regs()[1],
                            jit->Regs()[2], jit->Regs()[3], sp);
                    /* dump stack for LR chain */
                    fprintf(stderr, "[arm_exec] stack[sp..sp+40]:");
                    for (int _i = 0; _i < 10; ++_i) {
                        uint32_t v = ctx->mem.read32(sp + (uint32_t)(_i * 4));
                        fprintf(stderr, " %08x", v);
                    }
                    fprintf(stderr, "\n");
                }
            }
            if(jit) jit->HaltExecution();
            return;
        }
        uint32_t instr = 0;
        if (auto p = ctx->mem.ptr(pc)) std::memcpy(&instr, p, 4);
        static int exc_count = 0;
        if (exc_count < 3) {
            auto &regs = jit->Regs();
            fprintf(stderr,"[arm_exec] ExceptionRaised ex=%d pc=0x%08x instr=0x%08x "
                    "r0=%08x r1=%08x r4=%08x lr=%08x sp=%08x cpsr=%08x\n",
                    (int)ex, pc, instr,
                    regs[0], regs[1], regs[4], regs[14], regs[13],
                    jit->Cpsr());
        }
        ++exc_count;
        if (ex == E::UndefinedInstruction) {
            /* Runaway guard: thousands of patches mean the PC is marching
 * through data (e.g. after a wild jump).  Patching forever grows
 * the JIT cache without bound — halt instead. */
            static int patch_count = 0;
            if (++patch_count > 20000) {
                if (patch_count == 20001 || (patch_count % 100000) == 0)
                    fprintf(stderr,"[arm_exec] undefined-instruction patch limit "
                            "(%d) — halting run (pc=0x%08x)\n", patch_count, pc);
                if (jit) jit->HaltExecution();
                return;
            }
            if ((patch_count % 1000) == 0)
                fprintf(stderr,"[arm_exec] NOP-patch count: %d (pc=0x%08x)\n",
                        patch_count, pc);
            if (jit) {
                bool thumb = (jit->Cpsr() >> 5) & 1;
                uint32_t nop_pc = pc & ~1u;
                if (pc < 0x200) {
                    /* Vtable-stub area: patch with BX LR so stubs return cleanly */
                    if (thumb)
                        ctx->mem.write32(nop_pc, 0x46C04770u); /* BX LR ; NOP */
                    else
                        ctx->mem.write32(nop_pc, 0xE12FFF1Eu); /* ARM BX LR */
                    fprintf(stderr,"[arm_exec] patched undefined at 0x%08x with BX LR\n", pc);
                } else {
                    /* Valid Unity code: NOP the bad instruction and continue */
                    if (thumb) {
                        /* Peek at first halfword to decide 16-bit vs 32-bit Thumb-2 */
                        uint16_t hw = 0;
                        if (auto *p = ctx->mem.ptr(nop_pc)) std::memcpy(&hw, p, 2);
                        if ((hw >> 11) >= 0x1Du) /* 0b11101/11110/11111 = 32-bit */
                            ctx->mem.write32(nop_pc, 0xF3AF8000u); /* NOP.W */
                        else
                            ctx->mem.write32(nop_pc, 0xBF00BF00u); /* NOP ; NOP */
                    } else {
                        ctx->mem.write32(nop_pc, 0xE320F000u); /* ARM NOP */
                    }
                    fprintf(stderr,"[arm_exec] NOP'd undefined at 0x%08x, continuing\n", pc);
                }
                jit->InvalidateCacheRange(nop_pc, 8);
                return;  /* don't halt — JIT retries from PC with the patched NOP */
            }
        }
        if (ex == E::UnpredictableInstruction) {
            fprintf(stderr, "[arm_exec] UnpredictableInstruction at pc=0x%08x "
                    "— halting\n", pc);
            if (jit) jit->HaltExecution();
        }
    }
    uint64_t trace_blocks = 0; /* LUNARIA_TRACE_BLOCKS: log pc per JIT block */
    /* recent block PC ring (dumped by LUNARIA_DUMP_LAST_SVC) */
    uint32_t pc_ring[64] = {};
    uint32_t pc_ring_pos = 0;
    /* recent guest code PCs (excludes trampolines at 0x41000000+) */
    uint32_t code_ring[64] = {};
    uint32_t code_ring_pos = 0;

    void AddTicks(uint64_t t) override {
        uint64_t prev = ticks;
        ticks += t;
        if (jit) {
            uint32_t p = jit->Regs()[15];
            pc_ring[pc_ring_pos++ & 63] = p;
            if ((p & ~1u) < 0x41000000u) code_ring[code_ring_pos++ & 63] = p;
        }
        
        static const bool prof = getenv("LUNARIA_PROF") != nullptr;
        if (prof && jit) {
            static std::map<uint32_t, uint64_t> buckets;
            buckets[(jit->Regs()[15] & ~1u) >> 12] += t;
            if ((prev >> 30) != (ticks >> 30)) {
                std::vector<std::pair<uint64_t, uint32_t>> top;
                top.reserve(buckets.size());
                uint64_t total = 0;
                for (auto &kv : buckets) { top.push_back(std::make_pair(kv.second, kv.first)); total += kv.second; }
                std::sort(top.begin(), top.end(), std::greater<>());
                fprintf(stderr, "[prof] %llu ticks in %zu buckets:\n",
                        (unsigned long long)total, buckets.size());
                for (size_t i = 0; i < top.size() && i < 12; ++i)
                    fprintf(stderr, "[prof]   0x%08x000 %5.1f%% (%llu)\n",
                            top[i].second, 100.0 * (double)top[i].first / (double)total,
                            (unsigned long long)top[i].first);
                buckets.clear();
            }
        }
        static const bool trace_jitinit = getenv("LUNARIA_TRACE_JITINIT") != nullptr;
        if (trace_jitinit && jit) {
            uint32_t pc15 = jit->Regs()[15] & ~1u;
            if (pc15 == 0x2013407cu) { /* mono_jit_init_version entry */
                static uint64_t n = 0;
                fprintf(stderr, "[jitinit] enter #%llu lr=0x%08x r0=0x%08x tid=%u\n",
                        (unsigned long long)n++, (uint32_t)jit->Regs()[14],
                        (uint32_t)jit->Regs()[0], g_current_tid);
            }
        }
        /* LUNARIA_TRACE_EXC self-test: mono_jit_init_version (a libmono
 * export) is watched here via AddTicks PC sampling.
 * RESULT: this hook NEVER fires, proving AddTicks PC observation does NOT
 * catch function entries — it samples block PCs that are ~99% trampolines  This is why Sessions 3–5's PC-sampling conclusions were
 * unreliable.  The actual exception capture is done reliably via the SVC
 * logging stubs installed at load time (see build_exc_logger_stub /
 * the LUNARIA_TRACE_EXC redirect in load_elf), which every PLT/GOT call must
 * pass through. */
        static const bool trace_exc = getenv("LUNARIA_TRACE_EXC") != nullptr;
        if (trace_exc && jit) {
            static uint32_t va_mjiv = ~0u;
            if (va_mjiv == ~0u) {
                auto it = g_exported_syms.find("mono_jit_init_version");
                va_mjiv = (it != g_exported_syms.end()) ? (it->second & ~1u) : 0u;
                fprintf(stderr, "[exc] AddTicks self-test target mjiv=0x%08x\n", va_mjiv);
            }
            if (va_mjiv && (jit->Regs()[15] & ~1u) == va_mjiv) {
                static uint64_t n = 0;
                fprintf(stderr, "[exc] selftest mjiv-entry #%llu (PC sampling DID fire) "
                        "lr=0x%08x tid=%u\n",
                        (unsigned long long)n++, (uint32_t)jit->Regs()[14], g_current_tid);
            }
        }
        /* LUNARIA_TRACE_IL2CPP: log each real entry to il2cpp_init (guest 0x008c2864)
 * to confirm whether the engine re-enters runtime init (the GC
 * "Exclusion ranges overlap" abort happens on the 2nd call). */
        static const bool trace_il2cpp = getenv("LUNARIA_TRACE_IL2CPP") != nullptr;
        if (trace_il2cpp && jit) {
            uint32_t pc15 = jit->Regs()[15] & ~1u;
            if (pc15 == 0x008c2864u) {
                static uint64_t n = 0;
                fprintf(stderr, "[il2cpp_init] enter #%llu lr=0x%08x tid=%u\n",
                        (unsigned long long)n++, (uint32_t)jit->Regs()[14],
                        g_current_tid);
            }
        }
        /* LUNARIA_TRACE_MUTEX: log entry to il2cpp os::FastMutex::Lock (guest
 * 0x8bcd7c) with caller lr and the mutex pointer (r1) so we can find
 * who passes the bogus (code-pointing) mutex object. */
        static const bool trace_mutex = getenv("LUNARIA_TRACE_MUTEX") != nullptr;
        if (trace_mutex && jit) {
            uint32_t pc15 = jit->Regs()[15] & ~1u;
            /* Catch the block right after the entry pushes regs + calls the
 * tid helper (0x8bcd90).  The caller's lr was saved on the stack by
 * `push {r4,r5,r6,r7,r11,lr}` at entry → saved lr is at sp+0x14. */
            /* Fire at the spin point (0x8bcdd4, reliably reported by AddTicks).
 * The frame pushed at entry is still intact, so the caller's lr is
 * at sp+0x14 (push {r4,r5,r6,r7,r11,lr}). */
            if (pc15 == 0x8bcdd4u) {
                static uint64_t n = 0;
                if (n < 5) {
                    uint32_t sp = (uint32_t)jit->Regs()[13];
                    uint32_t saved_lr = g_ctx ? g_ctx->mem.read32(sp + 0x14) : 0;
                    /* literal pool used by the caller to compute the mutex addr:
 * ldr r1,[pc,#0xec8] @ guest 0x914524. Expected 0x1a0c2fc. */
                    uint32_t lit = g_ctx ? g_ctx->mem.read32(0x914524u) : 0;
                    fprintf(stderr, "[mutexlock] #%llu caller_lr=0x%08x mutex(r5)=0x%08x "
                            "lit[0x914524]=0x%08x tid=%u\n",
                            (unsigned long long)n, saved_lr,
                            (uint32_t)jit->Regs()[5], lit, g_current_tid);
                }
                ++n;
            }
        }
        if (trace_blocks > 0 && jit) {
            fprintf(stderr, "[blk] pc=0x%08x lr=0x%08x r0=0x%08x\n",
                    (uint32_t)jit->Regs()[15], (uint32_t)jit->Regs()[14],
                    (uint32_t)jit->Regs()[0]);
            --trace_blocks;
        }
        /* LUNARIA_TRACE_PCRANGE=lo:hi — log block entries whose PC falls in [lo,hi).
 * Diagnostic for locating where a specific function bails (logs r0 at
 * each block boundary, i.e. right after every sub-call returns). */
        if (jit) {
            static uint32_t pcr_lo = 0, pcr_hi = 0, pcr_left = 0;
            static bool pcr_init = false;
            if (!pcr_init) {
                pcr_init = true;
                if (const char *e = getenv("LUNARIA_TRACE_PCRANGE")) {
                    sscanf(e, "%x:%x", &pcr_lo, &pcr_hi);
                    pcr_left = 1200; /* cap output */
                }
            }
            if (pcr_hi && pcr_left) {
                uint32_t pc15 = (uint32_t)jit->Regs()[15] & ~1u;
                if (pc15 >= pcr_lo && pc15 < pcr_hi) {
                    fprintf(stderr, "[pcr] pc=0x%08x r0=0x%08x r4=0x%08x lr=0x%08x\n",
                            pc15, (uint32_t)jit->Regs()[0], (uint32_t)jit->Regs()[4],
                            (uint32_t)jit->Regs()[14]);
                    --pcr_left;
                }
            }
        }
        if (ticks >= ticks_limit && jit)
            jit->HaltExecution();
        /* Unlimited runs: force Run() to return every 256M ticks so the run_arm
 * heartbeat loop can fire schedule_threads for worker thread progress. */
        if (ticks_limit == UINT64_MAX && (prev >> 28) != (ticks >> 28))
            periodic_yield = true;
        /* Progress log every ~1B ticks so long inits are visible */
        if ((prev >> 30) != (ticks >> 30) && jit) {
            long rss_pages = 0;
            if (FILE *f = fopen("/proc/self/statm", "r")) {
                long dummy; if (fscanf(f, "%ld %ld", &dummy, &rss_pages) != 2) rss_pages = 0;
                fclose(f);
            }
            fprintf(stderr, "[arm_exec] ...%llu ticks, pc=0x%08x rss=%ldMB\n",
                    (unsigned long long)ticks, (uint32_t)jit->Regs()[15],
                    rss_pages * 4096 / (1024*1024));
            
            fprintf(stderr, "[arm_exec]   recent code pcs:");
            for (int k = 1; k <= 8; ++k)
                fprintf(stderr, " %08x", code_ring[(code_ring_pos - k) & 63]);
            fprintf(stderr, "\n");
        }
        /* Poll GLFW events every ~16M ticks to keep window responsive */
        if (g_glfw && (prev >> 24) != (ticks >> 24))
            glfwPollEvents();
    }
    uint64_t GetTicksRemaining() override {
        if (ticks >= ticks_limit) return 0;
        if (periodic_yield) return 0;   /* periodic yield: stop before next block */
        uint64_t rem = ticks_limit - ticks;
        return rem < 1'000'000ull ? rem : 1'000'000ull;
    }
};

/* -------------------------------------------------------------------------
 * Cooperative thread scheduler (auxiliary JIT)
 * ---------------------------------------------------------------------- */
static void schedule_threads(uint64_t slice) {
    if (!g_ctx || g_scheduling || g_threads.empty()) return;
    ArmExecCtx &ctx = *g_ctx;

    /* Lazily create the auxiliary JIT used exclusively for guest threads */
    if (!ctx.jit_aux) {
        ctx.cb_aux = std::make_unique<ArmCallbacks>();
        ctx.cb_aux->ctx = &ctx;
        Dynarmic::A32::UserConfig cfg;
        cfg.callbacks = ctx.cb_aux.get();
        cfg.processor_id = 1;
        cfg.global_monitor = &ctx.excl_mon;
        if (!g_wrange_hi) {
            cfg.fastmem_pointer = (uintptr_t)ctx.mem.host;
            cfg.recompile_on_fastmem_failure = true;
        }
        install_stub_coprocessors(cfg);
        ctx.jit_aux = std::make_unique<Dynarmic::A32::Jit>(cfg);
        ctx.cb_aux->jit = ctx.jit_aux.get();
    }

    g_scheduling = true;
    ArmCallbacks &cb = *ctx.cb_aux;
    Dynarmic::A32::Jit &jit = *ctx.jit_aux;
    uint32_t prev_tid = g_current_tid;

    using HR = Dynarmic::HaltReason;
    for (size_t i = 0; i < g_threads.size(); ++i) {
        ArmThread &t = g_threads[i];
        if (t.finished) continue;
        /* sem_wait blocking: skip slices until g_sems[waiting_sem] > 0 (SVC retries) */
        if (t.waiting_sem) {
            auto it = g_sems.find(t.waiting_sem);
            if (it != g_sems.end() && it->second > 0) {
                --it->second;
                t.regs[0] = 0;               /* sem_wait success */
            } else if (t.sem_timed && ++t.sem_skip_passes >= 2000u) {
                t.regs[0] = ~0u;             /* ETIMEDOUT */
                if (uint32_t eva = errno_va(ctx, t.id))
                    ctx.mem.write32(eva, 110u);
            } else {
                continue;
            }
            t.waiting_sem = 0;
            t.sem_skip_passes = 0;
        }
        /* futex_wait parking: keep the thread de-scheduled until a matching
 * FUTEX_WAKE token is pending or the futex word changes.  Resuming it
 * only on a real wake keeps the Baselib semaphore acquire in lock-step
 * with the poster (fn/arg written before the wake), so the worker never
 * dispatches a NULL job. */
        if (t.waiting_futex) {
            uint32_t uaddr = t.waiting_futex;
            auto tok = g_futex_wake_tokens.find(uaddr);
            bool woken = (tok != g_futex_wake_tokens.end() && tok->second > 0u);
            bool changed = (ctx.mem.read32(uaddr) != t.futex_val);
            if (woken) {
                if (--tok->second == 0u) g_futex_wake_tokens.erase(tok);
            } else if (!changed) {
                continue;   /* still parked */
            }
            t.waiting_futex = 0;
            auto wa = g_futex_wait_addrs.find(uaddr);
            if (wa != g_futex_wait_addrs.end() && wa->second > 0u && --wa->second == 0u)
                g_futex_wait_addrs.erase(wa);
        }
        /* Swappy / libc++ thread may be registered before its std::function at
 * [arg+0x10] is populated; skip slices until the callback pointer appears. */
        if (t.entry_pc >= 0x02703800u && t.entry_pc < 0x02704000u) {
            uint32_t arg = t.regs[0];
            if (arg && ctx.mem.ptr(arg + 0x10u) && !ctx.mem.read32(arg + 0x10u)) {
                if (++t.defer_count >= 256u) {
                    t.finished = true;
                    fprintf(stderr, "[sched] tid=%u retired: callback at [arg+0x10] never set\n",
                            t.id);
                } else {
                    static int defer_log = 0;
                    if (defer_log++ < 4)
                        fprintf(stderr, "[sched] defer tid=%u: empty callback at [arg+0x10]\n",
                                t.id);
                }
                continue;
            }
            t.defer_count = 0;
        }
        cb.ticks = 0;
        cb.ticks_limit = slice;
        jit.ClearHalt(HR::MemoryAbort | HR::UserDefined1 | HR::UserDefined2 |
                      HR::UserDefined3 | HR::UserDefined4 | HR::Step);
        jit.Regs()    = t.regs;
        jit.ExtRegs() = t.ext;
        jit.SetCpsr(t.cpsr);
        jit.SetFpscr(t.fpscr);
        set_cur_tid(ctx, t.id);
        {
            static uint32_t sched_count = 0;
            if (sched_count < 10)
                fprintf(stderr, "[sched] running thread tid=%u fn=0x%08x pc=0x%08x #%u\n",
                        t.id, t.entry_pc, t.regs[15] & ~1u, sched_count);
            ++sched_count;
        }
        /* Same premature-return semantics as run_arm: Run() exits with hr == 0
 * when the cycle budget runs dry (refilled only at SVCs).  Keep resuming
 * so a CPU-bound thread actually consumes its whole slice instead of
 * being cut off after ~1M ticks. */
        Dynarmic::HaltReason thr;
        for (;;) {
            thr = jit.Run();
            if (thr != Dynarmic::HaltReason{}) break;
            if (pc_in_sentinel(jit.Regs()[15])) break;
            if (cb.ticks >= slice) break;
        }
        t.regs  = jit.Regs();
        t.ext   = jit.ExtRegs();
        t.cpsr  = jit.Cpsr();
        t.fpscr = jit.Fpscr();
        t.total_ticks += cb.ticks;
        
        uint32_t pc15 = t.regs[15] & ~1u;
        if (pc_in_sentinel(pc15)) {
            t.finished = true;
            fprintf(stderr, "[arm_exec] thread %u finished (ret=0x%08x pc=0x%08x)\n",
                    t.id, t.regs[0], pc15);
        }
        /* NULL indirect call (blx r0/r2==0): PC marches through the zero page
 * forever.  Retire the thread immediately — nothing below 0x1000 is
 * legitimate guest code in our flat arena layout. */
        if (!t.finished && pc15 < 0x1000u) {
            t.finished = true;
            static int null_retire = 0;
            if (null_retire++ < 6)
                fprintf(stderr, "[arm_exec] thread %u retired: NULL indirect call "
                        "(pc=0x%08x lr=0x%08x)\n", t.id, pc15, t.regs[14]);
            if (getenv("LUNARIA_TRACE_JOBS")) {
                uint32_t th = t.regs[4]; /* worker loop keeps `this` in r4 */
                fprintf(stderr, "  [retire dump] r4=0x%08x r5=0x%08x r6=0x%08x",
                        t.regs[4], t.regs[5], t.regs[6]);
                if (th >= 0x1000u)
                    fprintf(stderr, " | +0x48=0x%08x +0x88=0x%08x +0xc8=0x%08x "
                            "+0x108=0x%08x +0x148=0x%08x +0x14c=0x%08x +0x150=0x%08x",
                            ctx.mem.read32(th+0x48), ctx.mem.read32(th+0x88),
                            ctx.mem.read32(th+0xc8), ctx.mem.read32(th+0x108),
                            ctx.mem.read32(th+0x148), ctx.mem.read32(th+0x14c),
                            ctx.mem.read32(th+0x150));
                fprintf(stderr, "\n");
            }
        }
        /* Detect a thread permanently stuck on a zero-instruction fetch: if it
 * returns to the same PC 32 times in a row with MemoryAbort, the code
 * at that address will never appear (no other thread writes JIT code to
 * it) — retire the thread to prevent an infinite no-progress loop. */
        if ((thr & Dynarmic::HaltReason::MemoryAbort) != Dynarmic::HaltReason{}) {
            if (pc15 == t.stuck_pc) {
                if (++t.stuck_count >= 32) {
                    t.finished = true;
                    fprintf(stderr, "[arm_exec] thread %u retired: stuck at 0x%08x "
                            "(MemoryAbort ×%u)\n", t.id, pc15, t.stuck_count);
                }
            } else {
                t.stuck_pc    = pc15;
                t.stuck_count = 1;
            }
        } else {
            t.stuck_count = 0; /* made progress, reset counter */
        }
    }
    
    {
        static const uint32_t dump_every =
            (uint32_t)strtoul(getenv("LUNARIA_SCHED_DUMP") ? getenv("LUNARIA_SCHED_DUMP") : "0",
                              nullptr, 10);
        static uint32_t pass = 0;
        if (dump_every && (++pass % dump_every) == 0) {
            fprintf(stderr, "[sched] pass=%u threads:\n", pass);
            for (const auto &t : g_threads)
                if (!t.finished)
                    fprintf(stderr, "[sched]   tid=%-3u pc=0x%08x lr=0x%08x r0=0x%08x "
                            "ticks=%llu\n", t.id, t.regs[15] & ~1u, t.regs[14],
                            t.regs[0], (unsigned long long)t.total_ticks);
        }
    }
    set_cur_tid(ctx, prev_tid);
    /* A thread slice may set g_yield_requested to pause *that* thread's JIT
 * (jit_aux).  Clear it here so the flag does not propagate back to the
 * main JIT when schedule_threads() is called from inside a CallSVC that
 * is running on ctx.jit — otherwise the next g_yield_requested check in
 * CallSVC would fire HaltExecution() on the main JIT unintentionally. */
    g_yield_requested = false;
    g_scheduling = false;
}

/* Run a short guest function fn_va(a, b) on the dedicated callback JIT and
 * return its r0 as a signed int.  Lets an SVC handler invoke a guest callback
 * (the comparator bsearch is handed) without re-entering the main or aux JIT,
 * which dynarmic forbids while their Run() is on the stack. */
static int32_t call_guest_cb(ArmExecCtx &ctx, uint32_t fn_va, uint32_t a, uint32_t b,
                             uint32_t c, uint32_t d,
                             const uint32_t *stk, int nstk) {
    if (!g_ctx || !fn_va) return 0;
    /* The callback JIT is itself single-threaded.  mono's metadata comparators
 * are pure (they never call bsearch again), so guard against accidental
 * recursion rather than corrupt the JIT — bail safely instead. */
    if (g_in_cb) {
        static int warned = 0;
        if (warned++ < 4)
            fprintf(stderr, "[bsearch] nested guest callback fn=0x%08x — skipped\n", fn_va);
        return 0;
    }
    if (!ctx.jit_cb) {
        ctx.cb_cb = std::make_unique<ArmCallbacks>();
        ctx.cb_cb->ctx = &ctx;
        Dynarmic::A32::UserConfig cfg;
        cfg.callbacks = ctx.cb_cb.get();
        cfg.processor_id = 2;
        cfg.global_monitor = &ctx.excl_mon;
        if (!g_wrange_hi) {
            cfg.fastmem_pointer = (uintptr_t)ctx.mem.host;
            cfg.recompile_on_fastmem_failure = true;
        }
        install_stub_coprocessors(cfg);
        ctx.jit_cb = std::make_unique<Dynarmic::A32::Jit>(cfg);
        ctx.cb_cb->jit = ctx.jit_cb.get();
        ctx.mem.map(CB_STACK_BASE, CB_STACK_SIZE);
    }

    g_in_cb = true;
    ArmCallbacks &cb = *ctx.cb_cb;
    Dynarmic::A32::Jit &jit = *ctx.jit_cb;
    uint32_t prev_tid = g_current_tid;

    using HR = Dynarmic::HaltReason;
    cb.ticks = 0;
    cb.ticks_limit = 200'000'000ull;  /* comparators tiny; signal handlers need more */
    cb.last_svc = UINT32_MAX;
    cb.trace_svcs = false;
    jit.ClearHalt(HR::MemoryAbort | HR::UserDefined1 | HR::UserDefined2 |
                  HR::UserDefined3 | HR::UserDefined4 | HR::Step);

    bool is_thumb = (fn_va & 1u) != 0;
    auto &regs = jit.Regs();
    regs.fill(0);
    regs[0]  = a;
    regs[1]  = b;
    regs[2]  = c;
    regs[3]  = d;
    uint32_t cb_sp = CB_STACK_BASE + CB_STACK_SIZE - 16;
    if (stk && nstk > 0) {
        cb_sp -= (uint32_t)((nstk + 1) & ~1) * 4u;   /* keep 8-byte alignment */
        for (int i = 0; i < nstk; ++i)
            ctx.mem.write32(cb_sp + 4u * (uint32_t)i, stk[i]);
    }
    regs[13] = cb_sp;
    regs[14] = SENTINEL_ADDR;
    regs[15] = fn_va & ~1u;
    jit.SetCpsr(is_thumb ? 0x00000030u : 0x00000010u);
    /* Resume across cycle-budget exhaustion (see run_arm) so a comparator
 * longer than one refill window still runs to completion. */
    for (;;) {
        Dynarmic::HaltReason chr = jit.Run();
        if (chr != Dynarmic::HaltReason{}) break;
        if (pc_in_sentinel(jit.Regs()[15])) break;
        if (cb.ticks >= cb.ticks_limit) break;
    }

    int32_t rv = (int32_t)jit.Regs()[0];
    if (!pc_in_sentinel(jit.Regs()[15])) {
        static int warned = 0;
        if (warned++ < 4)
            fprintf(stderr, "[bsearch] comparator fn=0x%08x did not return cleanly "
                    "(pc=0x%08x ticks=%llu) — result may be unreliable\n",
                    fn_va, (uint32_t)jit.Regs()[15], (unsigned long long)cb.ticks);
    }
    g_current_tid = prev_tid;
    g_in_cb = false;
    return rv;
}

/* Pump every started AAudio stream's data callback once with a silent burst.
 * On device this callback runs on a realtime audio thread and drives FMOD's
 * mixer + async command processing; without it, nonblocking opens never
 * complete and Unity's main thread waits forever on their done-flags.
 * Callback ABI: result cb(AAudioStream*, void *user, void *audioData, i32 frames). */
static void drive_aaudio_callbacks(ArmExecCtx &ctx) {
    if (g_in_cb || g_aaudio_streams.empty()) return;
    if (!g_aaudio_buf_mapped) {
        ctx.mem.map(AAUDIO_BUF_BASE, AAUDIO_BUF_SIZE);
        g_aaudio_buf_mapped = true;
    }
    for (auto &kv : g_aaudio_streams) {
        AAudioStreamState &st = kv.second;
        if (!st.started || !st.data_cb) continue;
        uint32_t bytes_per_sample = (st.format == 1u) ? 2u : 4u; /* I16 : FLOAT */
        uint32_t bytes = AAUDIO_BURST_FRAMES * st.channels * bytes_per_sample;
        if (bytes > AAUDIO_BUF_SIZE) bytes = AAUDIO_BUF_SIZE;
        memset(ctx.mem.ptr(AAUDIO_BUF_BASE), 0, bytes);   /* silence in/out */
        int32_t rc = call_guest_cb(ctx, st.data_cb, kv.first, st.user_data,
                                   AAUDIO_BUF_BASE, AAUDIO_BURST_FRAMES);
        static uint64_t pump_n = 0;
        if (pump_n < 8 || (pump_n % 2000) == 0)
            fprintf(stderr, "[aaudio] pump #%llu stream=0x%08x cb=0x%08x rc=%d\n",
                    (unsigned long long)pump_n, kv.first, st.data_cb, rc);
        ++pump_n;
    }
}

/* Deliver deferred Java Choreographer.doFrame callbacks: re-enter the
 * JNIBridge proxy that implements Choreographer$FrameCallback with the
 * interface class + Method recorded at newInterfaceProxy time (the native
 * matcher compares those exact handles with IsSameObject).  On device this
 * runs once per vsync on the UnityChoreographer looper thread; the native
 * doFrame handler re-posts postFrameCallback for the next frame. */
static void drive_java_choreographer(ArmExecCtx &ctx) {
    if (g_in_cb || !g_java_choreo_pending || !g_ctx) return;
    uint64_t fc_ptr = 0;
    uint32_t fc_cls = 0;
    for (auto &it : g_jnibridge_iface)
        for (auto &rec : it.second)
            if (rec.name.find("Choreographer$FrameCallback") != std::string::npos)
                { fc_ptr = it.first; fc_cls = rec.cls; }
    if (!fc_ptr || !fc_cls) return;
    uint32_t fn = 0;
    for (auto &rn : g_ctx->natives)
        if (rn.name == "invoke" && strstr(rn.klass.c_str(), "JNIBridge"))
            { fn = rn.fn_va; break; }
    if (!fn) return;
    struct jvm *jvm = ctx.jvm;
    JNIEnv *env = &jvm->env;
    static uint32_t br_cls = 0, dof = 0;
    if (!br_cls) br_cls = (uint32_t)(uintptr_t)jvm->native.FindClass(
                              env, "bitter/jnibridge/JNIBridge");
    if (!dof) {
        dof = (uint32_t)(uintptr_t)jvm->native.GetMethodID(
            env, (jclass)(uintptr_t)fc_cls, "doFrame", "(J)V");
        if (dof && dof <= 65536u)
            g_method_stub_names[dof] = "doFrame";
    }
    if (!dof) return;
    /* args = { boxed frameTimeNanos }: the guest unboxes via longValue() */
    static jobject long_box = nullptr;
    if (!long_box)
        long_box = jvm->native.AllocObject(env,
            jvm->native.FindClass(env, "java/lang/Long"));
    jobjectArray args = jvm->native.NewObjectArray(
        env, 1, (jclass)(uintptr_t)fc_cls, nullptr);
    jvm->native.SetObjectArrayElement(env, args, 0, long_box);
    g_java_choreo_pending = 0;   /* consumed; doFrame re-posts if it wants more */
    g_choreo_frame_ns += 16'666'666ull;
    uint32_t stk[3] = { fc_cls, dof, (uint32_t)(uintptr_t)args };
    int32_t rc = call_guest_cb(ctx, fn, ENV_SLOT_BASE, br_cls,
                               (uint32_t)fc_ptr, (uint32_t)(fc_ptr >> 32),
                               stk, 3);
    static uint64_t dof_n = 0;
    if (dof_n < 8 || (dof_n % 600) == 0)
        fprintf(stderr, "[jnibridge] doFrame #%llu ptr=0x%llx rc=0x%x pending=%u\n",
                (unsigned long long)dof_n, (unsigned long long)fc_ptr,
                (uint32_t)rc, g_java_choreo_pending);
    ++dof_n;
}

static void build_fast_mutex_stubs(ArmExecCtx &ctx) {
    if (g_fastmutex_lock_va) return;
    auto movw = [](uint32_t rd, uint32_t imm){
        return 0xE3000000u | ((imm & 0xf000u) << 4) | (rd << 12) | (imm & 0x0fffu); };
    auto movt = [](uint32_t rd, uint32_t imm){
        return 0xE3400000u | ((imm & 0xf000u) << 4) | (rd << 12) | (imm & 0x0fffu); };
    ctx.mem.map(FAST_MUTEX_PAGE, 0x100);
    ctx.mem.write32(CUR_TID_VA, 1u);            /* main thread = tid 0 → 1 */
    uint32_t p = FAST_MUTEX_PAGE + 0x10;

    
    g_fastmutex_lock_va = p;
    const uint32_t lock_code[] = {
        movw(2, CUR_TID_VA & 0xffffu), movt(2, CUR_TID_VA >> 16),
        0xE5922000u,                    /* ldr r2, [r2]        (tid+1) */
        0xE5903000u,                    /* ldr r3, [r0] */
        0xE2833C01u,                    /* add r3, r3, #0x100  count++ */
        0xE3C330FFu,                    /* bic r3, r3, #0xff */
        0xE1833002u,                    /* orr r3, r3, r2      owner=cur */
        0xE5803000u,                    /* str r3, [r0] */
        0xE3A00000u,                    /* mov r0, #0 */
        0xE12FFF1Eu,                    /* bx lr */
    };
    for (uint32_t w : lock_code) { ctx.mem.write32(p, w); p += 4; }

    
    p = (p + 15) & ~15u;
    g_fastmutex_unlock_va = p;
    const uint32_t unlock_code[] = {
        0xE5903000u,                    /* ldr r3, [r0] */
        0xE2533C01u,                    /* subs r3, r3, #0x100 */
        0x43A03000u,                    /* movmi r3, #0        (underflow) */
        0xE3530C01u,                    /* cmp r3, #0x100 */
        0x33A03000u,                    /* movlo r3, #0        (count==0 → clear owner) */
        0xE5803000u,                    /* str r3, [r0] */
        0xE3A00000u,                    /* mov r0, #0 */
        0xE12FFF1Eu,                    /* bx lr */
    };
    for (uint32_t w : unlock_code) { ctx.mem.write32(p, w); p += 4; }

    
    p = (p + 15) & ~15u;
    g_fastmutex_trylock_va = p;
    const uint32_t trylock_code[] = {
        movw(2, CUR_TID_VA & 0xffffu), movt(2, CUR_TID_VA >> 16),
        0xE5922000u,                    /* ldr r2, [r2] */
        0xE5903000u,                    /* ldr r3, [r0] */
        0xE3530C01u,                    /* cmp r3, #0x100      held? */
        0x3A000002u,                    /* blo acquire         (not held) */
        0xE20310FFu,                    /* and r1, r3, #0xff */
        0xE1510002u,                    /* cmp r1, r2          own thread? */
        0x1A000005u,                    /* bne busy */
        0xE2833C01u,                    /* acquire: add r3, r3, #0x100 */
        0xE3C330FFu,                    /* bic r3, r3, #0xff */
        0xE1833002u,                    /* orr r3, r3, r2 */
        0xE5803000u,                    /* str r3, [r0] */
        0xE3A00000u,                    /* mov r0, #0 */
        0xE12FFF1Eu,                    /* bx lr */
        0xE3A00010u,                    /* busy: mov r0, #16 (EBUSY) */
        0xE12FFF1Eu,                    /* bx lr */
    };
    for (uint32_t w : trylock_code) { ctx.mem.write32(p, w); p += 4; }
    fprintf(stderr, "[arm_exec] fast mutex stubs: lock=%#x unlock=%#x trylock=%#x\n",
            g_fastmutex_lock_va, g_fastmutex_unlock_va, g_fastmutex_trylock_va);
}


static void set_cur_tid(ArmExecCtx &ctx, uint32_t tid) {
    g_current_tid = tid;
    if (g_fastmutex_lock_va) ctx.mem.write32(CUR_TID_VA, tid + 1u);
}

/* -------------------------------------------------------------------------
 * Build JNI/JVM tables in ARM address space
 * ---------------------------------------------------------------------- */
static void build_jni_tables(ArmExecCtx &ctx) {
    build_fast_mutex_stubs(ctx);
    auto tramp = [](uint32_t n){ return TRAMP_BASE + n * TRAMP_STRIDE; };

    /* Trampoline page: SVC #n + BX LR (ARM32).  Map the whole window up to
     * JNI_TBL_BASE — slots past SVC_TRAMP_TOTAL are written lazily by dlsym
     * for unknown-symbol stubs. */
    ctx.mem.map(TRAMP_BASE, JNI_TBL_BASE - TRAMP_BASE);
    for (uint32_t n = 0; n < SVC_TRAMP_TOTAL; ++n) {
        ctx.mem.write32(TRAMP_BASE + n*TRAMP_STRIDE,     0xEF000000u | n);
        ctx.mem.write32(TRAMP_BASE + n*TRAMP_STRIDE + 4, 0xE12FFF1Eu);
    }

    /* JNINativeInterface table (229 entries) */
    ctx.mem.map(JNI_TBL_BASE, JNI_VTABLE_COUNT * 4);
    for (uint32_t i = 0; i < JNI_VTABLE_COUNT; ++i)
        ctx.mem.write32(JNI_TBL_BASE + i*4, tramp(i));

    /* JNIInvokeInterface table (8 slots) */
    ctx.mem.map(JVM_TBL_BASE, JVM_SLOT_COUNT * 4);
    for (uint32_t i = 0; i < JVM_SLOT_COUNT; ++i)
        ctx.mem.write32(JVM_TBL_BASE + i*4, tramp(0)); /* default: no-op */
    /* Override critical slots */
    ctx.mem.write32(JVM_TBL_BASE + JVM_SLOT_DESTROY * 4, tramp(SVC_JVM_DESTROY));
    ctx.mem.write32(JVM_TBL_BASE + JVM_SLOT_ATTACH  * 4, tramp(SVC_JVM_ATTACH));
    ctx.mem.write32(JVM_TBL_BASE + JVM_SLOT_GETENV  * 4, tramp(SVC_JVM_GETENV));

    /* ENV_SLOT: points to JNI_TBL_BASE (this IS the JNIEnv* value seen by ARM) */
    ctx.mem.map(ENV_SLOT_BASE, 32);
    ctx.mem.write32(ENV_SLOT_BASE, JNI_TBL_BASE);

    /* VM_SLOT: points to JVM_TBL_BASE (this IS the JavaVM* value seen by ARM) */
    /* VM_SLOT_BASE = ENV_SLOT_BASE + 8; already in the same mapped region */
    ctx.mem.write32(VM_SLOT_BASE, JVM_TBL_BASE);

    /* String scratch buffer */
    ctx.mem.map(STR_SCRATCH, 4096);

    /* libc data globals (bionic exports these as variables) */
    ctx.mem.map(LIBC_DATA, 4096);
    ctx.mem.write32(LIBC_PAGE_SIZE,  4096u);
    ctx.mem.write32(LIBC_PAGE_SHIFT, 12u);
    ctx.mem.write32(LIBC_PAGE_MASK,  0xfffu);
    /* noop stub — mov r0,#0 (e3a00000); bx lr (e12fff1e).
 * Used as substitute target when blx r12 would call a heap/null address. */
    ctx.mem.write32(NOOP_RET0,     0xe3a00000u); /* mov r0, #0 */
    ctx.mem.write32(NOOP_RET0 + 4, 0xe12fff1eu); /* bx lr */

    /* pthread_once guest stub:
 * push {r4,lr}; ldr r4,[r0]; cmp r4,#0; bne 1f;
 * mov r4,#1; str r4,[r0]; blx r1;
 * 1: mov r0,#0; pop {r4,pc} */
    static const uint32_t once_stub[] = {
        0xE92D4010u, 0xE5904000u, 0xE3540000u, 0x1A000002u,
        0xE3A04001u, 0xE5804000u, 0xE12FFF31u, 0xE3A00000u,
        0xE8BD8010u,
    };
    ctx.mem.map(PTHREAD_ONCE_STUB, sizeof(once_stub));
    for (size_t i = 0; i < sizeof(once_stub)/4; ++i)
        ctx.mem.write32(PTHREAD_ONCE_STUB + (uint32_t)(i*4), once_stub[i]);

    /* 0xffff0fe0: __kuser_get_tls
 * 0xffff0fc0: __kuser_cmpxchg */
    ctx.mem.write32(0xffff0ffcu, 2u); /* version */
    ctx.mem.write32(0xffff0fe0u, 0xEF000000u | SVC_KUSER_GET_TLS);
    ctx.mem.write32(0xffff0fe4u, 0xE12FFF1Eu); /* BX LR */
    ctx.mem.write32(0xffff0fc0u, 0xEF000000u | SVC_KUSER_CMPXCHG);
    ctx.mem.write32(0xffff0fc4u, 0xE12FFF1Eu); /* BX LR */
}

/* -------------------------------------------------------------------------
 * ELF loader + import resolver
 * base_addr: added to all virtual addresses (for loading at non-zero base)
 * ---------------------------------------------------------------------- */

/* mono_add_internal_call (0x198154) inserts into the BSS global icall_hash
 * @ +0x3bfb8c without lazily creating it (mono_icall_init only runs inside
 * mono_init, but Unity registers internal calls from initJni first), so the
 * first registration hits ghashtable.c:184.  Pre-create icall_hash exactly the
 * way mono itself would: g_hash_table_new(g_str_hash, g_str_equal) — the keys
 * are strdup'd icall names and mono_lookup_internal_call looks them up with a
 * freshly-built string, so a direct-pointer hash would never match.
 * Do NOT touch jit_icall_hash_name/addr @ +0x3bfb90/94: mono_register_jit_icall
 * (0x199208) lazily creates them with the correct hash functions when the name
 * slot is still NULL.  Pre-filling them with direct-hash tables made every
 * mono_find_jit_icall_by_name lookup miss ("unknown MONO_PATCH_INFO_INTERNAL_
 * METHOD mono_get_lmf_addr", mini.c:4500 `callinfo' assert) and broke all JIT
 * compilation. */
static void ensure_mono_icall_hashes(ArmExecCtx &ctx, uint32_t mono_base) {
    if (getenv("LUNARIA_ICALL_HASH_INIT") && strcmp(getenv("LUNARIA_ICALL_HASH_INIT"), "0") == 0)
        return;
    const uint32_t slot = mono_base + 0x3bfb8cu;   /* icall_hash */
    if (ctx.mem.read32(slot) != 0) return;
    const uint32_t ghash_new = mono_fn_va(nullptr, 0x2b66d8u); /* g_hash_table_new */
    /* g_str_hash / g_str_equal: eglib statics, reachable via libmono's own
 * relocated GOT entries (same slots mono_register_jit_icall loads from). */
    const uint32_t str_hash = ctx.mem.read32(mono_base + 0x3b8a74u);
    const uint32_t str_eq   = ctx.mem.read32(mono_base + 0x3b8e3cu);
    if (!ghash_new || !str_hash || !str_eq) {
        fprintf(stderr, "[arm_exec] icall_hash init skipped (new=%#x hash=%#x eq=%#x)\n",
                ghash_new, str_hash, str_eq);
        return;
    }
    uint32_t tbl = (uint32_t)run_arm(ctx, ghash_new, str_hash, str_eq, 0, 0, 50'000'000ULL);
    if (!tbl) {
        fprintf(stderr, "[arm_exec] g_hash_table_new failed for icall_hash\n");
        return;
    }
    ctx.mem.write32(slot, tbl);
    fprintf(stderr, "[arm_exec] icall_hash 0x%08x = 0x%08x (g_str_hash/g_str_equal)\n",
            slot, tbl);
}

/* mono_assembly_open_full calls mono_assembly_load_from_full for corlib, but
 * mono_defaults.corlib (@+0x3c0e6c) is only consumed at mono_runtime_init tail
 * (0x135da8).  Unity loads corlib later in nativeRender, so we register the
 * image here and defer the type-init pass until the guest JIT is idle. */
static constexpr uint32_t MONO_ASM_WRAP_STUB = 0x4100e000u;
static uint32_t g_mono_wrap_next = MONO_ASM_WRAP_STUB;

/* Preserve the first 8 bytes of an export before redirect_export overwrites them. */
static uint32_t build_mono_export_trampoline(ArmExecCtx &ctx, uint32_t export_va) {
    export_va &= ~1u;
    const uint32_t o0 = ctx.mem.read32(export_va);
    const uint32_t o1 = ctx.mem.read32(export_va + 4);
    const uint32_t stub = g_mono_wrap_next;
    g_mono_wrap_next += 0x20u;
    const uint32_t code[] = { o0, o1, 0xE51FF004u, export_va + 8u };
    ctx.mem.map(stub, sizeof(code));
    for (size_t i = 0; i < 4; ++i)
        ctx.mem.write32(stub + (uint32_t)(i * 4), code[i]);
    return stub;
}

static void redirect_export(ArmExecCtx &ctx, const char *sym, uint32_t stub_va) {
    auto it = g_exported_syms.find(sym);
    if (it == g_exported_syms.end()) {
        fprintf(stderr, "[lunaria] redirect %s: symbol not exported, skip\n", sym);
        return;
    }
    uint32_t va = it->second & ~1u;
    ctx.mem.write32(va,     0xE51FF004u);
    ctx.mem.write32(va + 4, stub_va);
    if (ctx.jit) ctx.jit->InvalidateCacheRange(va, 8);
    fprintf(stderr, "[lunaria] redirect %s @0x%08x -> stub 0x%08x\n", sym, va, stub_va);
}

/* Config already parsed in arm_exec_prepare_mono_config; mjiv must not re-enter. */
static uint32_t build_mono_config_parse_stub(ArmExecCtx &ctx) {
    const uint32_t code[] = { 0xE12FFF1Eu }; /* bx lr */
    ctx.mem.map(MONO_CFG_PARSE_STUB, (uint32_t)sizeof(code));
    ctx.mem.write32(MONO_CFG_PARSE_STUB, code[0]);
    fprintf(stderr, "[arm_exec] mono_config_parse noop stub @0x%08x\n", MONO_CFG_PARSE_STUB);
    return MONO_CFG_PARSE_STUB;
}

/* glib g_build_filename(NULL, …) trips gmisc-unix.c:69 during path probing. */
static uint32_t build_g_build_filename_stub(ArmExecCtx &ctx, uint32_t real_va) {
    auto movw = [](uint32_t rd, uint32_t imm){
        return 0xE3000000u | (rd<<12) | ((imm & 0xf000u)<<4) | (imm & 0x0fffu); };
    auto movt = [](uint32_t rd, uint32_t imm){
        return 0xE3400000u | (rd<<12) | ((imm & 0xf000u)<<4) | (imm & 0x0fffu); };
    const uint32_t E = MONO_EMPTY_STR;
    const uint32_t code[] = {
        0xE3500000u, /* cmp r0, #0 */
        0x1A000001u, /* bne → word4 (ldr pc,[pc,#-4]) */
        movw(0, E & 0xffff), movt(0, (E >> 16) & 0xffff),
        0xE51FF004u, /* ldr pc, [pc, #-4] */
        real_va,
    };
    ctx.mem.map(G_BUILD_FN_STUB, (uint32_t)sizeof(code));
    for (size_t i = 0; i < sizeof(code) / 4; ++i)
        ctx.mem.write32(G_BUILD_FN_STUB + (uint32_t)(i * 4), code[i]);
    fprintf(stderr, "[arm_exec] g_build_filename NULL-guard stub @0x%08x → real 0x%08x\n",
            G_BUILD_FN_STUB, real_va);
    return G_BUILD_FN_STUB;
}

static void redirect_va(ArmExecCtx &ctx, uint32_t va, uint32_t stub_va, const char *label) {
    va &= ~1u;
    ctx.mem.write32(va,     0xE51FF004u);
    ctx.mem.write32(va + 4, stub_va);
    if (ctx.jit) ctx.jit->InvalidateCacheRange(va, 8);
    fprintf(stderr, "[lunaria] redirect %s @0x%08x -> stub 0x%08x\n", label, va, stub_va);
}

/* Unity APK has no *.dll.config; machine.config comes from mono_register_machine_config. */
static uint32_t build_mono_cfg_asm_stub(ArmExecCtx &ctx) {
    const uint32_t code[] = {
        0xE3A00000u, /* mov r0, #0 */
        0xE12FFF1Eu, /* bx lr */
    };
    ctx.mem.map(MONO_CFG_ASM_STUB, (uint32_t)sizeof(code));
    for (size_t i = 0; i < sizeof(code) / 4; ++i)
        ctx.mem.write32(MONO_CFG_ASM_STUB + (uint32_t)(i * 4), code[i]);
    fprintf(stderr, "[arm_exec] mono_config_for_assembly noop stub @0x%08x\n",
            MONO_CFG_ASM_STUB);
    return MONO_CFG_ASM_STUB;
}

/* Build the mono_jit_init_version idempotency wrapper (see MJIV_STUB comment).
 * real_va = guest VA of the real mono_jit_init_version (a libmono export).
 * domain_get_va = mono_domain_get (fallback when init unwinds without caching). */
static uint32_t build_mjiv_stub(ArmExecCtx &ctx, uint32_t real_va, uint32_t domain_get_va) {
    if (g_mjiv_stub_va) return g_mjiv_stub_va;
    auto movw = [](uint32_t rd, uint32_t imm){
        return 0xE3000000u | (rd<<12) | ((imm & 0xf000u)<<4) | (imm & 0x0fffu); };
    auto movt = [](uint32_t rd, uint32_t imm){
        return 0xE3400000u | (rd<<12) | ((imm & 0xf000u)<<4) | (imm & 0x0fffu); };
    const uint32_t C = MJIV_CACHE, R = real_va, D = domain_get_va;
    const uint32_t M = MJIV_ENTRY_MARK;
    std::vector<uint32_t> code;
    auto emit = [&](uint32_t w){ code.push_back(w); };
    emit(movw(3, C & 0xffff)); emit(movt(3, (C>>16) & 0xffff));
    emit(0xE5932000u); /* ldr r2,[r3] */
    emit(0xE3520000u); /* cmp r2,#0 */
    emit(0x11A00002u); /* movne r0,r2 */
    emit(0x112FFF1Eu); /* bxne lr */
    emit(movw(2, M & 0xffff)); emit(movt(2, (M>>16) & 0xffff));
    emit(0xE5921000u); /* ldr r1,[r2] */
    emit(0xE3510000u); /* cmp r1,#0 */
    size_t first_time_branch = code.size();
    emit(0x0A000000u);
    if (D) {
        emit(0xE92D4008u); /* push {r3,lr} */
        emit(movw(12, D & 0xffff)); emit(movt(12, (D>>16) & 0xffff));
        emit(0xE12FFF3Cu); /* blx ip  mono_domain_get */
        emit(0xE8BD4008u); /* pop {r3,lr} */
        emit(0xE3500000u); /* cmp r0,#0 */
        size_t strne_idx = code.size();
        emit(0x05830000u);
        code[strne_idx] = 0x05330000u; /* strne r0,[r3] */
        emit(0x112FFF1Eu); /* bxne lr */
    }
    size_t first_time_pc = code.size();
    code[first_time_branch] = 0x0A000000u | ((uint32_t)(first_time_pc - first_time_branch - 2) & 0xffffffu);
    emit(0xE5921000u);
    emit(0xE2811001u);
    emit(0xE5821000u);
    emit(0xEF000000u | SVC_MONO_PREP);
    {
        static const char kMonoRuntimeVer[] = "v1.1.4322";
        memcpy(ctx.mem.ptr(MJIV_RUNTIME_VER), kMonoRuntimeVer, sizeof kMonoRuntimeVer);
        emit(movw(1, MJIV_RUNTIME_VER & 0xffff));
        emit(movt(1, (MJIV_RUNTIME_VER >> 16) & 0xffff));
    }
    emit(0xE92D4008u);
    emit(movw(12, R & 0xffff)); emit(movt(12, (R>>16) & 0xffff));
    emit(0xE12FFF3Cu); /* blx ip  real mono_jit_init_version */
    emit(0xE8BD4008u);
    emit(0xE5830000u); /* str r0,[r3] */
    emit(0xE12FFF1Eu);
    ctx.mem.map(MJIV_STUB, (uint32_t)(code.size() * 4));
    for (size_t i = 0; i < code.size(); ++i)
        ctx.mem.write32(MJIV_STUB + (uint32_t)(i * 4), code[i]);
    ctx.mem.write32(MJIV_CACHE, 0u);
    g_mjiv_stub_va = MJIV_STUB;
    fprintf(stderr,"[arm_exec] mono_jit_init_version idempotency wrapper "
            "@0x%08x → real 0x%08x domain_get 0x%08x\n", MJIV_STUB, real_va, D);
    return g_mjiv_stub_va;
}

/* Build a logging tail-call stub for a libmono export (Session 9, LUNARIA_TRACE_EXC).
 * stub_va: where to place it; svc_num: the logging SVC; real_va: the real fn.
 * Layout (ARM):  svc #svc_num ; ldr pc,[pc,#-4] ; <real_va literal>
 * The SVC handler logs the (preserved) args and returns without touching regs;
 * the ldr pc then tail-calls real_va with r0-r3/lr intact, so real returns
 * directly to the original caller (the stub is transparent except for logging). */
static uint32_t build_exc_logger_stub(ArmExecCtx &ctx, uint32_t stub_va,
                                      uint32_t svc_num, uint32_t real_va) {
    const uint32_t code[] = {
        0xEF000000u | svc_num, /* svc #svc_num   (log; regs preserved) */
        0xE51FF004u,           /* ldr pc,[pc,#-4]  -> next word */
        real_va,               /* literal: real libmono fn (thumb bit honored) */
    };
    ctx.mem.map(stub_va, sizeof(code));
    for (size_t i = 0; i < sizeof(code)/4; ++i)
        ctx.mem.write32(stub_va + (uint32_t)(i*4), code[i]);
    fprintf(stderr, "[arm_exec] exc-logger stub @0x%08x (svc=%u) -> real 0x%08x\n",
            stub_va, svc_num, real_va);
    return stub_va;
}

/* --- inline detours (LUNARIA_TRACE_EXC) ---------------------------
 * Reliably trace a libmono-internal function that is called via direct bl /
 * a registered function pointer (so neither PC sampling nor GOT/PLT redirect
 * sees it).  We overwrite the function's first 2 instructions with
 * `ldr pc,[pc,#-4]; <detour>` and relocate the originals into a detour stub:
 * svc #(SVC_DETOUR_BASE+n)   ; log args (regs preserved)
 * <orig insn 0>              ; copy of patched-out instruction
 * <orig insn 1>
 * ldr pc,[pc,#-4]; <target+8>; resume after the patched region
 * REQUIRES the first 2 instructions to be position-independent (push/add/sub
 * prologues are fine).  The >32MB branch range is handled by ldr-pc literals.
 * State (g_detour_*) is declared near EXC_STUB_BASE so dispatch_svc can read it. */
/* Redirect a guest function entry to an SVC trampoline by exported symbol name.
 * Used for libmono helpers that are called via direct BL inside libmono.so (PLT
 * redirect alone is not enough).  Symbol lookup keeps this version-agnostic. */
static void hook_va_to_svc(ArmExecCtx &ctx, uint32_t va, uint32_t svc_no, const char *label) {
    va &= ~1u;
    uint32_t tramp = TRAMP_BASE + svc_no * TRAMP_STRIDE;
    ctx.mem.write32(va,     0xE51FF004u); /* ldr pc,[pc,#-4] */
    ctx.mem.write32(va + 4, tramp);
    if (ctx.jit) ctx.jit->InvalidateCacheRange(va, 8);
    fprintf(stderr, "[lunaria] hook %s @0x%08x -> svc %u (tramp 0x%08x)\n",
            label, va, svc_no, tramp);
}

static void hook_export_to_svc(ArmExecCtx &ctx, const char *sym, uint32_t svc_no) {
    auto it = g_exported_syms.find(sym);
    if (it == g_exported_syms.end()) {
        fprintf(stderr, "[lunaria] hook %s: symbol not exported, skip\n", sym);
        return;
    }
    uint32_t va = it->second & ~1u;
    uint32_t tramp = TRAMP_BASE + svc_no * TRAMP_STRIDE;
    ctx.mem.write32(va,     0xE51FF004u); /* ldr pc,[pc,#-4] */
    ctx.mem.write32(va + 4, tramp);
    if (ctx.jit) ctx.jit->InvalidateCacheRange(va, 8);
    fprintf(stderr, "[lunaria] hook %s @0x%08x -> svc %u (tramp 0x%08x)\n",
            sym, va, svc_no, tramp);
}

static void install_inline_detour(ArmExecCtx &ctx, uint32_t target_va, const char *name) {
    if (g_detour_count >= NUM_DETOURS) return;
    uint32_t n   = g_detour_count;
    uint32_t tgt = target_va & ~1u;             /* ARM target */
    uint32_t o0  = ctx.mem.read32(tgt);
    uint32_t o1  = ctx.mem.read32(tgt + 4);
    uint32_t stub = DETOUR_STUB_BASE + n * 0x20u;
    const uint32_t code[] = {
        0xEF000000u | (SVC_DETOUR_BASE + n), /* svc #(base+n)  (log, no reg change) */
        o0, o1,                              /* relocated original prologue */
        0xE51FF004u,                         /* ldr pc,[pc,#-4] */
        tgt + 8u,                            /* resume after the patched 2 instrs */
    };
    ctx.mem.map(stub, sizeof(code));
    for (size_t i = 0; i < sizeof(code)/4; ++i)
        ctx.mem.write32(stub + (uint32_t)(i*4), code[i]);
    ctx.mem.write32(tgt,     0xE51FF004u);   /* ldr pc,[pc,#-4] */
    ctx.mem.write32(tgt + 4, stub);
    if (ctx.jit) ctx.jit->InvalidateCacheRange(tgt, 8);
    g_detour_names[n]   = name;
    g_detour_targets[n] = tgt;
    ++g_detour_count;
    fprintf(stderr, "[detour] %s @0x%08x -> stub 0x%08x (svc=%u) o0=%08x o1=%08x\n",
            name, tgt, stub, SVC_DETOUR_BASE + n, o0, o1);
}

static bool load_elf(ArmExecCtx &ctx, const char *path,
                     uint32_t &jni_onload_va, uint32_t base_addr,
                     bool ctors_via_cb) {
    int fd = open(path, O_RDONLY);
    if(fd<0){perror("arm_exec: open"); return false;}
    struct stat st; fstat(fd,&st);
    std::vector<uint8_t> buf(st.st_size);
    if(read(fd,buf.data(),st.st_size)!=(ssize_t)st.st_size){close(fd);return false;}
    close(fd);

    const auto *ehdr = reinterpret_cast<const Elf32_Ehdr *>(buf.data());
    if(memcmp(ehdr->e_ident,ELFMAG,SELFMAG)||
       ehdr->e_ident[EI_CLASS]!=ELFCLASS32||ehdr->e_machine!=EM_ARM){
        fprintf(stderr,"arm_exec: not ARM32 ELF\n"); return false; }

    /* Map PT_LOAD segments at base_addr + p_vaddr */
    const auto *phdrs = reinterpret_cast<const Elf32_Phdr *>(buf.data()+ehdr->e_phoff);
    uint32_t lib_lo = UINT32_MAX, lib_hi = 0;   /* span used by diagnostic code below */
    uint32_t exidx_va = 0, exidx_count = 0;
    for(int i=0;i<ehdr->e_phnum;++i){
        const bool is_exidx = phdrs[i].p_type == (Elf32_Word)0x70000001u;
        if(!is_exidx && phdrs[i].p_type != PT_LOAD) continue;
        if(phdrs[i].p_memsz == 0) continue;
        uint32_t va=base_addr+phdrs[i].p_vaddr, msz=phdrs[i].p_memsz,
                 fsz=phdrs[i].p_filesz, off=phdrs[i].p_offset;
        ctx.mem.map(va,msz);
        if(fsz>0&&off+fsz<=(uint32_t)buf.size())
            memcpy(ctx.mem.ptr(va), buf.data()+off, fsz);
        /* Update high-water mark (page-aligned) */
        uint32_t seg_end = (va + msz + 0xffffu) & ~0xffffu;
        if(seg_end > g_lib_load_end) g_lib_load_end = seg_end;
        if(va < lib_lo) lib_lo = va;
        if(va + msz > lib_hi) lib_hi = va + msz;
        if(is_exidx) { exidx_va = va; exidx_count = msz / 8u; }
        if(!is_exidx) {
            /* Record each PT_LOAD with accurate ELF permissions so synth_guest_maps
 * can mark code r-xp and data rw-p — Boehm GC only scans writable
 * segments for roots, so this avoids scanning ~14 MB of code text. */
            g_loaded_regions.push_back({va & ~0xfffu, ((va+msz)+0xfffu) & ~0xfffu,
                                        phdrs[i].p_flags, path});
        }
    }
    if (exidx_va)
        g_module_exidx.push_back({lib_lo, lib_hi, exidx_va, exidx_count});
    /* Do not rewrite EXIDX entries in the loaded image — binary patches are
 * version-specific and can mask or introduce bugs. */

    /* Section headers */
    const auto *shdrs = reinterpret_cast<const Elf32_Shdr *>(buf.data()+ehdr->e_shoff);

    const Elf32_Sym *dynsym=nullptr; size_t dynsym_n=0;
    const char      *dynstr=nullptr;

    struct RelSec { const Elf32_Rel *rels; size_t n; uint32_t link; };
    std::vector<RelSec> rel_secs;

    for(int i=0;i<ehdr->e_shnum;++i){
        const auto &sh=shdrs[i];
        if(sh.sh_type==SHT_DYNSYM){
            dynsym=(const Elf32_Sym*)(buf.data()+sh.sh_offset);
            dynsym_n=sh.sh_size/sizeof(Elf32_Sym);
            dynstr=(const char*)(buf.data()+shdrs[sh.sh_link].sh_offset);
        } else if(sh.sh_type==SHT_REL&&sh.sh_size>0){
            rel_secs.push_back({(const Elf32_Rel*)(buf.data()+sh.sh_offset),
                                sh.sh_size/sizeof(Elf32_Rel), sh.sh_link});
        }
    }

    /* Trampoline address for a given SVC number */
    auto tramp = [](uint32_t n){ return TRAMP_BASE + n * TRAMP_STRIDE; };

    /* Symbol name → SVC number */
    auto sym_svc = [](const char *name) -> uint32_t {
        return lookup_symbol_svc(name);
    };

    jni_onload_va = 0;
    if(dynsym&&dynstr){
        /* Collect exported symbols for cross-library resolution */
        for(size_t i=0;i<dynsym_n;++i){
            const char *n=dynstr+dynsym[i].st_name;
            if(!n||!*n) continue;
            if(strcmp(n,"JNI_OnLoad")==0 && dynsym[i].st_value)
                jni_onload_va = base_addr + dynsym[i].st_value;
            /* Export defined symbols (non-UND, non-zero value) */
            if(dynsym[i].st_shndx!=SHN_UNDEF && dynsym[i].st_value){
                uint32_t sym_va = base_addr + dynsym[i].st_value;
                if(!g_exported_syms.count(n))   /* first library wins */
                    g_exported_syms[n] = sym_va;
            }
        }
        fprintf(stderr,"[arm_exec] %s: exported %zu symbols (base=0x%08x)\n",
                path, g_exported_syms.size(), base_addr);

        /* Apply relocations */
        for(auto &rs:rel_secs){
            const auto *stab=(const Elf32_Sym*)(buf.data()+shdrs[rs.link].sh_offset);
            const char *str =(const char*)(buf.data()+shdrs[shdrs[rs.link].sh_link].sh_offset);
            for(size_t j=0;j<rs.n;++j){
                uint32_t si=ELF32_R_SYM(rs.rels[j].r_info);
                uint32_t rt=ELF32_R_TYPE(rs.rels[j].r_info);
                uint32_t ro=base_addr + rs.rels[j].r_offset;
                /* R_ARM_RELATIVE: *ro += base (REL: addend stored in place) */
                if(rt==R_ARM_RELATIVE){
                    if(base_addr)
                        ctx.mem.write32(ro, ctx.mem.read32(ro) + base_addr);
                    continue;
                }
                if(!si) continue;
                /* Defined symbol: eager-bind to base + st_value */
                if(stab[si].st_shndx!=SHN_UNDEF){
                    uint32_t sv = base_addr + stab[si].st_value;
                    switch(rt){
                    case R_ARM_JUMP_SLOT:
                    case R_ARM_GLOB_DAT:
                        ctx.mem.write32(ro, sv);
                        break;
                    case R_ARM_ABS32:
                        ctx.mem.write32(ro, ctx.mem.read32(ro) + sv);
                        break;
                    default: break;
                    }
                    continue;
                }
                const char *sym=str+stab[si].st_name;

                /* Priority: cross-library export → direct stub → SVC trampoline */
                uint32_t tv;
                auto exp_it = g_exported_syms.find(sym);
                /* mono_jit_init_version: redirect to the idempotency wrapper that
 * caches the first non-NULL domain so per-frame re-init returns it
 * instead of re-running mini_init.  Default ON; set LUNARIA_MJIV_WRAP=0
 * to disable.  No-op for IL2CPP titles. */
                const char *mjiv_env = getenv("LUNARIA_MJIV_WRAP");
                bool mjiv_wrap = !mjiv_env || strcmp(mjiv_env, "0") != 0;
                /* LUNARIA_TRACE_EXC — redirect the two libmono exception
 * entry points to logging tail-call stubs.  Applies to every
 * importing library (libunity AND libmono itself), so mono's own
 * internal calls through its PLT are caught too — the reliable way
 * to learn which exception aborts mono_jit_init_version. */
                static const bool trace_exc = getenv("LUNARIA_TRACE_EXC") != nullptr;
                if(trace_exc && exp_it != g_exported_syms.end() &&
                   strcmp(sym,"mono_exception_from_name_msg")==0){
                    if(!g_exc_stub_from_name)
                        g_exc_stub_from_name = build_exc_logger_stub(
                            ctx, EXC_STUB_BASE, SVC_EXC_FROM_NAME, exp_it->second);
                    tv = g_exc_stub_from_name;
                } else if(trace_exc && exp_it != g_exported_syms.end() &&
                   strcmp(sym,"mono_raise_exception")==0){
                    if(!g_exc_stub_raise)
                        g_exc_stub_raise = build_exc_logger_stub(
                            ctx, EXC_STUB_BASE + 0x40u, SVC_EXC_RAISE, exp_it->second);
                    tv = g_exc_stub_raise;
                } else if(exp_it != g_exported_syms.end() &&
                   strcmp(sym,"mono_jit_init_version")==0 && mjiv_wrap){
                    uint32_t mdg = 0;
                    if (auto dg = g_exported_syms.find("mono_domain_get");
                        dg != g_exported_syms.end())
                        mdg = dg->second;
                    tv = build_mjiv_stub(ctx, exp_it->second, mdg);
                } else if(exp_it != g_exported_syms.end()){
                    tv = exp_it->second;
                    fprintf(stderr,"[arm_exec] xlib: %s -> 0x%08x\n", sym, tv);
                } else if(uint32_t dva = lookup_symbol_direct_va(sym)){
                    tv = dva;
                } else {
                    uint32_t svc = sym_svc(sym);
                    if (svc == UINT32_MAX) {
                        
                        static std::unordered_set<std::string> s_warned;
                        if (s_warned.insert(sym).second)
                            fprintf(stderr, "[unresolved] sym=%s not in SVC map"
                                    " (addr=0x%08x) → returns 0 when called\n", sym, ro);
                        tv = tramp(SVC_UNKNOWN_CALL);
                    } else {
                        tv = tramp(svc);
                    }
                    if(strstr(sym,"unwind")||strstr(sym,"backtrace")||
                       strstr(sym,"pthread")||                               /* pthread trace */
                       (svc>=SVC_EGL_GETDISPLAY&&svc<=SVC_EGL_GETCURCTX) ||  
                       (svc>=SVC_GL_BASE&&svc<SVC_EXT_BASE))
                        fprintf(stderr,"[arm_exec] patching sym=%s svc=%u addr=0x%08x -> 0x%08x\n",
                                sym,svc,ro,tv);
                }
                switch(rt){
                case R_ARM_JUMP_SLOT:
                case R_ARM_GLOB_DAT:
                case R_ARM_ABS32:
                    ctx.mem.write32(ro,tv);
                    break;
                default: break;
                }
            }
        }
    }
    if (getenv("LUNARIA_TRACE_MUTEX") && base_addr == 0xa0000u)
        fprintf(stderr, "[litprobe] after-reloc  mem[0x914524]=0x%08x\n",
                ctx.mem.read32(0x914524u));
    /* Execute DT_INIT_ARRAY (C++ static constructors).
 * In real Android the dynamic linker calls these before any user code.
 * Without them, global singletons stay zero-initialized (BSS). */
    {
        uint32_t init_array_va = 0, init_array_sz = 0;
        for(int i=0;i<ehdr->e_phnum;++i){
            if(phdrs[i].p_type != PT_DYNAMIC) continue;
            const auto *dyn = reinterpret_cast<const Elf32_Dyn *>(buf.data()+phdrs[i].p_offset);
            for(; dyn->d_tag != DT_NULL; ++dyn){
                if(dyn->d_tag == DT_INIT_ARRAY)   init_array_va = base_addr + dyn->d_un.d_ptr;
                if(dyn->d_tag == DT_INIT_ARRAYSZ) init_array_sz = dyn->d_un.d_val;
            }
            break;
        }
        if(init_array_va && init_array_sz >= 4){
            uint32_t count = init_array_sz / 4;
            fprintf(stderr,"[arm_exec] running %u INIT_ARRAY ctors from 0x%08x\n",
                    count, init_array_va);
            for(uint32_t k=0; k<count; ++k){
                /* R_ARM_RELATIVE rebasing has already been applied above */
                uint32_t fn = ctx.mem.read32(init_array_va + k*4);
                if(fn == 0 || fn == 0xFFFFFFFFu) continue;
                if((k % 64) == 0)
                    fprintf(stderr,"[arm_exec] INIT_ARRAY ctor %u/%u\n", k, count);
                if (ctors_via_cb) {
                    /* dlopen-time load: the main JIT is mid-Run() inside an
                     * SVC, so run ctors on the standalone callback JIT. */
                    (void)call_guest_cb(ctx, fn, 0, 0);
                    continue;
                }
                run_arm(ctx, fn, 0, 0, 0, 0, 200'000'000ULL);
                if (getenv("LUNARIA_TRACE_MUTEX") && base_addr == 0xa0000u) {
                    uint32_t v = ctx.mem.read32(0x914524u);
                    if (v != 0x1a0c2fcu) {
                        fprintf(stderr, "[litprobe] CORRUPTED by INIT_ARRAY ctor %u/%u "
                                "(fn=0x%08x): mem[0x914524]=0x%08x\n", k, count, fn, v);
                        break;
                    }
                }
            }
        }
    }
    /* libmono assembly loader: mono_path_resolve_symlinks() returns a plain path
 * without a "file://" prefix, so try_open_assembly() never enters its fopen
 * branch and mscorlib.dll is never opened (mono_assembly_load_corlib → exit).
 * Route the path helpers through SVC handlers that speak host file:// URIs. */
    if (strstr(path, "libmono.so")) {
        g_mono_base = base_addr;
        hook_export_to_svc(ctx, "mono_path_resolve_symlinks", SVC_MONO_PATH_NORM);
        hook_export_to_svc(ctx, "mono_path_canonicalize",     SVC_MONO_PATH_NORM);
        /* Internal g_filename_from_uri is exported as mono_escape_uri_string in
 * this libmono build; decode file:// URIs before fopen. */
        hook_export_to_svc(ctx, "mono_escape_uri_string",     SVC_G_FILENAME_URI);
        /* Unity libmono: mono_file_map_open returns FILE* (fd @ shim+0x0e).
 * We host-open assemblies and let fd/size/map use the guest FILE shim. */
        hook_export_to_svc(ctx, "mono_file_map_open",         SVC_MONO_FILE_MAP_OPEN);
        hook_export_to_svc(ctx, "mono_file_map_fd",           SVC_MONO_FILE_MAP_FD);
        hook_export_to_svc(ctx, "mono_file_map_size",         SVC_MONO_FILE_MAP_SIZE);
        hook_export_to_svc(ctx, "mono_file_map",              SVC_MONO_FILE_MAP);
        hook_export_to_svc(ctx, "mono_file_map_close",        SVC_MONO_FILE_MAP_CLOSE);
        if (auto it = g_exported_syms.find("mono_config_for_assembly");
            it != g_exported_syms.end()) {
            const uint32_t stub = build_mono_cfg_asm_stub(ctx);
            redirect_export(ctx, "mono_config_for_assembly", stub);
        }
        if (auto it = g_exported_syms.find("mono_config_parse");
            it != g_exported_syms.end()) {
            const uint32_t exp_va = it->second & ~1u;
            g_mono_config_parse_tramp = build_mono_export_trampoline(ctx, exp_va);
            const uint32_t stub = build_mono_config_parse_stub(ctx);
            redirect_export(ctx, "mono_config_parse", stub);
        }
        ctx.mem.map(MONO_EMPTY_STR, 4u);
        ctx.mem.write32(MONO_EMPTY_STR, 0);
        {
            const uint32_t gbf = base_addr + 0x2b8914u;
            g_g_build_filename_tramp = build_mono_export_trampoline(ctx, gbf);
            const uint32_t stub = build_g_build_filename_stub(ctx, g_g_build_filename_tramp);
            redirect_va(ctx, gbf, stub, "g_build_filename");
        }
        ensure_mono_icall_hashes(ctx, base_addr);
        
        if (getenv("LUNARIA_TRACE_MONO")) {
            install_inline_detour(ctx, base_addr + 0x13c890u, "mono_assembly_open_full");
            install_inline_detour(ctx, base_addr + 0x13d3a8u, "mono_assembly_load_from_full");
            install_inline_detour(ctx, base_addr + 0x177158u, "mono_get_corlib");
            install_inline_detour(ctx, base_addr + 0x274bd8u, "mono_verify_corlib");
            install_inline_detour(ctx, base_addr + 0x19c00cu, "mono_image_open_full");
            install_inline_detour(ctx, base_addr + 0x19b428u, "mono_image_open_a");
            install_inline_detour(ctx, base_addr + 0x19b288u, "mono_image_init");
        }
        if (getenv("LUNARIA_TRACE_GHASH")) {
            install_inline_detour(ctx, base_addr + 0x29b620u, "g_hash_new(hash,equal)");
            install_inline_detour(ctx, base_addr + 0x29b9d8u, "g_hash_lookup(table,key)");
            install_inline_detour(ctx, base_addr + 0x29c370u, "g_hash_insert_rep(table,key,val)");
            install_inline_detour(ctx, base_addr + 0x29c58cu, "g_hash_insert(table,key,val)");
            install_inline_detour(ctx, base_addr + 0x2b81b9u, "g_assert_fail");
            /* Internal insert (not the export stub); NULL table asserts at line 184. */
            install_inline_detour(ctx, base_addr + 0x2b6a21u, "g_hash_insert_int(table,key,val,rep)");
        }
    }
    /* Generic diagnostic detours: LUNARIA_DETOUR=hexva[:label][,hexva[:label]...] */
    if (const char *dl = getenv("LUNARIA_DETOUR")) {
        char tmp[1024];
        snprintf(tmp, sizeof(tmp), "%s", dl);
        for (char *tok = strtok(tmp, ","); tok; tok = strtok(nullptr, ",")) {
            char *colon = strchr(tok, ':');
            const char *label = "LUNARIA_DETOUR";
            if (colon) { *colon = '\0'; label = colon + 1; }
            uint32_t va = (uint32_t)strtoul(tok, nullptr, 16);
            if (va >= lib_lo && va < lib_hi)
                install_inline_detour(ctx, va, strdup(label));
        }
    }
    if (const char *dm = getenv("LUNARIA_DUMPMEM")) {
        uint32_t va = (uint32_t)strtoul(dm, nullptr, 16);
        if (va >= lib_lo && va < lib_hi) {
            fprintf(stderr, "[dumpmem] 0x%08x:", va);
            for (int i = 0; i < 8; ++i)
                fprintf(stderr, " %08x", ctx.mem.read32(va + (uint32_t)i*4));
            fprintf(stderr, "\n");
        }
    }
    if (strstr(path, "libunity.so")) {
        if (const char *begin = memmem_buf(buf.data(), buf.size(), "<configuration>")) {
            const char *close = strstr(begin, "</configuration>");
            if (close) {
                close += strlen("</configuration>");
                g_unity_machine_config.assign(begin, (size_t)(close - begin));
                fprintf(stderr, "[arm_exec] libunity machine.config: %zu bytes\n",
                        g_unity_machine_config.size());
            }
        }
    }
    g_loaded_lib_paths.insert(path);
    return true;
}
static uint64_t env_ticks(const char *name, uint64_t def) {
    const char *v = getenv(name);
    if (!v || !*v) return def;
    char *end = nullptr;
    uint64_t x = strtoull(v, &end, 10);
    /* Honour the K/M/G suffixes documented in README (e.g. "5G", "200M"). */
    if (end && *end) {
        switch (*end) {
            case 'k': case 'K': x *= 1000ULL; break;
            case 'm': case 'M': x *= 1000'000ULL; break;
            case 'g': case 'G': x *= 1000'000'000ULL; break;
            default: break;
        }
    }
    return x ? x : def;
}

/* -------------------------------------------------------------------------
 * Run ARM code
 * ---------------------------------------------------------------------- */
static int run_arm(ArmExecCtx &ctx, uint32_t entry_va,
                   uint32_t r0_arg=0, uint32_t r1_arg=0,
                   uint32_t r2_arg=0, uint32_t r3_arg=0,
                   uint64_t max_ticks=UINT64_MAX) {
    bool is_thumb=(entry_va&1u)!=0;
    uint32_t pc=entry_va&~1u;

    /* Create the persistent JIT on first use; reuse thereafter to keep JIT cache */
    if (!ctx.jit) {
        ctx.cb = std::make_unique<ArmCallbacks>();
        ctx.cb->ctx = &ctx;
        Dynarmic::A32::UserConfig cfg;
        cfg.callbacks = ctx.cb.get();
        cfg.processor_id = 0;
        cfg.global_monitor = &ctx.excl_mon;
        /* Flat 4GB arena → guest loads/stores compile to direct host accesses.
         * LUNARIA_WATCH disables fastmem so every store routes through MemoryWrite*,
         * letting the watchpoint trap wild stores (no SIGSEGV from mprotect). */
        if (const char *xr = getenv("LUNARIA_TRACE_XCL")) {
            sscanf(xr, "%x:%x", &g_xcl_lo, &g_xcl_hi);
            fprintf(stderr, "[xcl] tracing exclusive writes in [0x%08x,0x%08x)\n",
                    g_xcl_lo, g_xcl_hi);
        }
        if (const char *wr = getenv("LUNARIA_WATCH_RANGE")) {
            sscanf(wr, "%x:%x", &g_wrange_lo, &g_wrange_hi);
            fprintf(stderr, "[wrange] watching [0x%08x,0x%08x)\n",
                    g_wrange_lo, g_wrange_hi);
        }
        if (!getenv("LUNARIA_WATCH") && !getenv("LUNARIA_WATCH_EXC_TYPE") &&
            !g_wrange_hi) {
            cfg.fastmem_pointer = (uintptr_t)ctx.mem.host;
            cfg.recompile_on_fastmem_failure = true;
        }
        install_stub_coprocessors(cfg);
        ctx.jit = std::make_unique<Dynarmic::A32::Jit>(cfg);
        ctx.cb->jit = ctx.jit.get();
    }

    ArmCallbacks &cb = *ctx.cb;
    cb.ticks = 0;
    cb.ticks_limit = max_ticks;
    cb.last_svc = UINT32_MAX;

    /* Clear all halt reasons from the previous call before re-running */
    using HR = Dynarmic::HaltReason;
    ctx.jit->ClearHalt(HR::MemoryAbort | HR::UserDefined1 | HR::UserDefined2 |
                       HR::UserDefined3 | HR::UserDefined4 | HR::Step);

    /* SVC tracing: set LUNARIA_TRACE_SVC=1 to log the first 1000 SVCs per call */
    cb.trace_svcs = getenv("LUNARIA_TRACE_SVC") != nullptr;
    if (const char *wf = getenv("LUNARIA_WATCH_FLAG")) {
        g_watch_flag_va = (uint32_t)strtoul(wf, nullptr, 0);
        g_watch_flag_last = g_watch_flag_va ? ctx.mem.read32(g_watch_flag_va) : 0;
    }
    cb.trace_svc_count = 0;
    cb.trace_blocks = env_ticks("LUNARIA_TRACE_BLOCKS", 0);

    auto &regs=ctx.jit->Regs();
    regs.fill(0);
    regs[0]=r0_arg; regs[1]=r1_arg; regs[2]=r2_arg; regs[3]=r3_arg;
    /* Restore callee-saved registers (R4-R11) from previous JNI call to
     * model JVM thread state — Unity stores singleton pointers in these. */
    if(ctx.regs_valid)
        for(int i=0;i<8;++i) regs[4+i]=ctx.saved_regs[i];
    regs[13]=STACK_BASE+STACK_SIZE-16;
    regs[14]=SENTINEL_ADDR;
    regs[15]=pc;
    ctx.jit->SetCpsr(is_thumb ? 0x00000030u : 0x00000010u);

    /* LUNARIA_STEP_GFX=0x<entrypc>: when this run starts at the given function entry
     * (e.g. nativeRecreateGfxState 0x26f94c1), single-step the whole call so the
     * guard branches inside it report PRECISE pc + r0 — AddTicks only yields
     * block-level/successor PCs and misses one-shot mid-function decisions.
     * libunity base = 0x23d0000 → nativeRecreateGfxState guards:
     *   0x26f94c0 entry, 0x26f94da guard1(r0), 0x26f94e4 guard2(r0),
     *   0x26f94ec real gfx-recreate reached. */
    /* dynarmic's Run() returns to the caller (hr == 0, no halt reason) whenever
     * the cycle budget from GetTicksRemaining runs out.  The budget is refilled
     * only at SVC boundaries, so any CPU-bound guest stretch longer than the
     * refill cap (~1M ticks) with no SVC in it would silently truncate the run
     * mid-function — the guest "returned" with a garbage r0 and never finished
     * (e.g. Unity's engine-init class scan, re-registering every ClassID each
     * frame).  Resume until the guest really returns (sentinel), a genuine halt
     * is raised, or the per-call tick limit is reached. */
    Dynarmic::HaltReason hr;
    uint64_t hb_next = 500'000'000ULL;          /* heartbeat every 500M ticks */
    for (;;) {
        hr = ctx.jit->Run();
        /* CacheInvalidation is raised when InvalidateCacheRange() is called from
         * within a SVC handler while the JIT is mid-Run (e.g. cacheflush after
         * mono JIT backpatch).  Clear it and resume — the invalidated block will
         * be recompiled on the next fetch. */
        if (Dynarmic::Has(hr, HR::CacheInvalidation)) {
            ctx.jit->ClearHalt(HR::CacheInvalidation);
            hr &= ~HR::CacheInvalidation;
        }
        cb.periodic_yield = false;                           /* consumed the yield */
        if (hr != Dynarmic::HaltReason{}) break;            /* fault/yield/step */
        if (pc_in_sentinel(ctx.jit->Regs()[15])) break;     /* guest returned  */
        if (cb.ticks >= max_ticks) break;                    /* tick budget hit */
        /* Heartbeat: print progress when running unlimited (e.g. nativeRender
         * scene-load) so silent JIT compilation phases are visible. */
        if (max_ticks == UINT64_MAX && cb.ticks >= hb_next) {
            fprintf(stderr, "[run_arm] heartbeat: ticks=%llu pc=0x%08x last_svc=%u\n",
                    (unsigned long long)cb.ticks,
                    (unsigned)ctx.jit->Regs()[15],
                    cb.last_svc);
            hb_next += 500'000'000ULL;
            /* Also give background threads a slice so worker deps can complete */
            if (!g_scheduling && !g_threads.empty())
                schedule_threads(env_ticks("LUNARIA_THREAD_TICKS", 200'000'000ULL));
        }
    }
    /* LUNARIA_WATCH precise capture: the first wild-store hit arms g_step_mode, records
     * the clobbered word (g_watch_poll_addr) and halts.  We then single-step and
     * poll that word *between* steps — at a step boundary R[15] is validly written
     * back, so the snapshot taken before the step that flips the word IS the exact
     * faulting instruction.  Continue until capture, sentinel, or tick limit. */
    if (g_step_mode && !g_step_capture_done) {
        uint32_t prev = ctx.mem.read32(g_watch_poll_addr);
        for (uint64_t steps = 0;
             !g_step_capture_done && steps < 4'000'000ull &&
             !pc_in_sentinel(ctx.jit->Regs()[15]) &&
             cb.ticks < cb.ticks_limit;
             ++steps) {
            /* Snapshot register state at the boundary *before* executing the next
             * instruction; R[15] here is the precise address about to run. */
            auto &Rb = ctx.jit->Regs();
            uint32_t snap[16];
            for (int i = 0; i < 16; ++i) snap[i] = (uint32_t)Rb[i];
            bool thumb = (ctx.jit->Cpsr() & (1u << 5)) != 0;

            ctx.jit->ClearHalt(HR::UserDefined1 | HR::Step);
            hr = ctx.jit->Step();

            uint32_t now = ctx.mem.read32(g_watch_poll_addr);
            if (now != prev) {
                uint32_t instr_pc = snap[15] & ~1u;
                fprintf(stderr,
                        "[watch] PRECISE store: addr=0x%08x %08x->%08x by pc=0x%08x (%s) tid=%u\n",
                        g_watch_poll_addr, prev, now, instr_pc, thumb ? "T" : "A",
                        g_current_tid);
                fprintf(stderr, "[watch] regs@store:");
                for (int i = 0; i < 16; ++i) fprintf(stderr, " r%d=0x%08x", i, snap[i]);
                fprintf(stderr, "\n[watch] stack (sp=0x%08x):", snap[13]);
                for (uint32_t k = 0; k < 48; ++k) {
                    uint32_t w = ctx.mem.read32(snap[13] + k * 4);
                    bool code = (w >= 0x000a0000u && w < 0x02320000u) ||
                                (w >= 0x023d0000u && w < 0x02ec8000u);
                    fprintf(stderr, " %s%08x%s", code ? "[" : "", w, code ? "]" : "");
                }
                fprintf(stderr, "\n");
                g_watch_lo = g_watch_hi = 0;   /* one-shot */
                g_step_capture_done = true;
                break;
            }
            prev = now;
        }
        g_step_mode = false;
    }
    if (getenv("LUNARIA_DUMP_LAST_SVC"))
        fprintf(stderr, "[run_arm] exit: hr=0x%x pc=0x%08x sentinel=%d ticks=%llu\n",
                (unsigned)hr, ctx.jit->Regs()[15],
                pc_in_sentinel(ctx.jit->Regs()[15]) ? 1 : 0,
                (unsigned long long)cb.ticks);
    if (hr != Dynarmic::HaltReason{} &&
        !pc_in_sentinel(ctx.jit->Regs()[15]))
        fprintf(stderr, "[arm_exec] run halted: reason=0x%x pc=0x%08x\n",
                (unsigned)hr, ctx.jit->Regs()[15]);

    /* Save callee-saved registers for the next JNI call */
    for(int i=0;i<8;++i) ctx.saved_regs[i]=regs[4+i];
    ctx.regs_valid = true;

    bool timed_out = (cb.ticks >= max_ticks);
    int ret=(int)ctx.jit->Regs()[0];
    if (timed_out)
        fprintf(stderr,"[arm_exec] run_arm: TIMEOUT after %llu ticks, last_svc=%u, pc=0x%08x\n",
                (unsigned long long)cb.ticks, cb.last_svc,
                (unsigned)ctx.jit->Regs()[15]);
    else
        fprintf(stderr,"[arm_exec] run_arm: %llu instructions executed tid=%u pc=0x%08x\n",
                (unsigned long long)cb.ticks, g_current_tid,
                (unsigned)ctx.jit->Regs()[15]);
    if (getenv("LUNARIA_DUMP_LAST_SVC")) {
        auto &fr = ctx.jit->Regs();
        fprintf(stderr, "[arm_exec] final: pc=0x%08x lr=0x%08x sp=0x%08x "
                "r0=0x%08x r7=0x%08x\n", fr[15], fr[14], fr[13], fr[0], fr[7]);
        /* Mutex spin probe: il2cpp os::FastMutex::Lock @ guest 0x8bcd7c.
         * Dump the lock object (r5) and its fields so we can tell whether the
         * lock word (offset 0) is the expected 0/1/2 or uninitialised garbage. */
        uint32_t fpc = fr[15] & ~1u;
        if (fpc >= 0x8bcd7cu && fpc <= 0x8bce40u) {
            uint32_t r5 = fr[5];
            fprintf(stderr, "[mutex] r4=0x%08x r5=0x%08x r6=0x%08x", fr[4], r5, fr[6]);
            if (r5 >= 0x1000u && r5 < 0xfff00000u) {
                fprintf(stderr, " lock[0]=0x%08x owner[0x40]=0x%08x count[0x44]=0x%08x"
                        " hdr[-4]=0x%08x",
                        ctx.mem.read32(r5), ctx.mem.read32(r5 + 0x40),
                        ctx.mem.read32(r5 + 0x44), ctx.mem.read32(r5 - 4));
            }
            fprintf(stderr, "\n");
        }
        svc_ring_dump();
        fprintf(stderr, "[arm_exec] last block PCs (oldest first):");
        for (uint32_t i = 0; i < 64; ++i) {
            uint32_t pc = cb.pc_ring[(cb.pc_ring_pos + i) & 63];
            if (pc) fprintf(stderr, " %08x", pc);
        }
        fprintf(stderr, "\n");
    }
    /* Keep ctx.jit alive to preserve the JIT cache across calls */
    return ret;
}

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */
extern "C" int arm_elf_is_arm32(const char *path) {
    int fd=open(path,O_RDONLY); if(fd<0) return 0;
    unsigned char ident[EI_NIDENT]; uint16_t mach=0;
    read(fd,ident,EI_NIDENT);
    lseek(fd,offsetof(Elf32_Ehdr,e_machine),SEEK_SET);
    read(fd,&mach,2); close(fd);
    return (memcmp(ident,ELFMAG,SELFMAG)==0&&
            ident[EI_CLASS]==ELFCLASS32&&mach==EM_ARM)?1:0;
}

/* Initialize the ARM execution context without loading any ELF.
 * Call this before arm_exec_load_library to pre-load dependency libraries. */
extern "C" int arm_exec_context_init(struct jvm *jvm) {
    if(g_ctx){fprintf(stderr,"arm_exec: context already exists\n"); return -1;}
    g_ctx=new ArmExecCtx();
    g_ctx->jvm=jvm;
    build_jni_tables(*g_ctx);
    g_ctx->mem.map(STACK_BASE, STACK_SIZE);
    g_ctx->mem.map(HEAP_BASE,  HEAP_SIZE);
    g_ctx->mem.map(MMAP_BASE,  MMAP_END - MMAP_BASE);
    /* Boehm GC sbrk arena: sbrk() returns addresses in [BRK_BASE, BRK_END).
     * Without this mapping, any guest write to sbrk-allocated memory faults.
     * Mapping it here makes GC_scratch_alloc / GC_add_to_heap work correctly
     * so GC_init completes and GC_initialized is set. */
    g_ctx->mem.map(BRK_BASE,   BRK_END - BRK_BASE);
    return 0;
}

extern "C" int arm_exec_jni_onload(const char *path, struct jvm *jvm) {
    /* Create context if not already initialized */
    if(!g_ctx){
        if(arm_exec_context_init(jvm) < 0) return -1;
    }

    /* Remember the main library's directory so findLibrary() can return the
     * full path to sibling libs (libil2cpp.so, libmain.so, …). */
    if (path) {
        const char *slash = strrchr(path, '/');
        g_main_lib_dir = slash ? std::string(path, (size_t)(slash - path)) : std::string(".");
    }

    /* Auto-place: if dependencies were pre-loaded, stack the main library after them */
    uint32_t main_base = (g_lib_load_end > 0) ? ((g_lib_load_end + 0xffffu) & ~0xffffu) : 0u;
    uint32_t jni_onload_va=0;
    fprintf(stderr,"[arm_exec] loading main library: %s at base=0x%08x\n", path, main_base);
    if(!load_elf(*g_ctx, path, jni_onload_va, main_base, false)){
        if(!g_ctx->jit){ delete g_ctx; g_ctx=nullptr; } return -1; }

    if(!jni_onload_va){
        fprintf(stderr,"arm_exec: JNI_OnLoad not found in %s\n", path);
        return -1; }

    fprintf(stderr,"[arm_exec] executing JNI_OnLoad @ 0x%08x (%s)\n",
            jni_onload_va, (jni_onload_va&1)?"Thumb":"ARM");
    g_svc_ring_pos = 0;
    memset(g_svc_ring, 0, sizeof(g_svc_ring));

    int ver=run_arm(*g_ctx, jni_onload_va, VM_SLOT_BASE, 0, 0, 0,
                    env_ticks("LUNARIA_ONLOAD_TICKS", 5'000'000'000ULL));

    fprintf(stderr,"[arm_exec] JNI_OnLoad returned 0x%x, %zu native(s) registered\n",
            ver, g_ctx->natives.size());
    /* Dump SVCs called during JNI_OnLoad */
    fprintf(stderr,"[arm_exec] JNI_OnLoad SVCs (last 64):\n");
    svc_ring_dump();
    return ver;
}

/* Load an additional ARM32 ELF library at the given base address.
 * Exports from the library become available for cross-library symbol resolution.
 * If the library has JNI_OnLoad, calls it and returns the JNI version; else 0.
 * Must be called after arm_exec_jni_onload has created the context. */
/* Scan ELF phdrs to find the total span of LOAD segments (max vaddr + memsz). */
static uint32_t elf_load_span(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    Elf32_Ehdr ehdr;
    if (read(fd, &ehdr, sizeof(ehdr)) != sizeof(ehdr)) { close(fd); return 0; }
    uint32_t span = 0;
    lseek(fd, ehdr.e_phoff, SEEK_SET);
    for (int i = 0; i < ehdr.e_phnum; i++) {
        Elf32_Phdr ph;
        if (read(fd, &ph, sizeof(ph)) != sizeof(ph)) break;
        if (ph.p_type == PT_LOAD && ph.p_memsz > 0) {
            uint32_t end = ph.p_vaddr + ph.p_memsz;
            if (end > span) span = end;
        }
    }
    close(fd);
    return span;
}

extern "C" int arm_exec_load_library(const char *path, uint32_t base_addr) {
    if(!g_ctx){fprintf(stderr,"arm_exec: no context — call arm_exec_context_init first\n"); return -1;}
    /* Auto-place PIC libraries to avoid overlapping existing segments.
     * base_addr=0 means "pick a non-overlapping base from g_lib_load_end." */
    if (base_addr == 0 && g_lib_load_end > 0) {
        /* Align to 64KB for dynarmic fastmem compatibility */
        base_addr = (g_lib_load_end + 0xffffu) & ~0xffffu;
    }
    fprintf(stderr,"[arm_exec] loading library: %s at base=0x%08x\n", path, base_addr);
    uint32_t jni_onload_va=0;
    if(!load_elf(*g_ctx, path, jni_onload_va, base_addr, false)) return -1;
    if(!jni_onload_va){
        fprintf(stderr,"[arm_exec] %s: no JNI_OnLoad, skipping\n", path);
        return 0;
    }
    fprintf(stderr,"[arm_exec] %s: JNI_OnLoad @ 0x%08x (%s)\n",
            path, jni_onload_va, (jni_onload_va&1)?"Thumb":"ARM");
    int ver = run_arm(*g_ctx, jni_onload_va, VM_SLOT_BASE, 0, 0, 0,
                      env_ticks("LUNARIA_ONLOAD_TICKS", 5'000'000'000ULL));
    fprintf(stderr,"[arm_exec] %s: JNI_OnLoad returned 0x%x\n", path, ver);
    return ver;
}

extern "C" int arm_exec_call_native(uint32_t fn_va, JNIEnv *env, jobject obj,
                                    const jvalue *args, int nargs, jvalue *ret) {
    if(!g_ctx) return -1;
    (void)env;
    /* ARM32 JNI calling convention: r0=JNIEnv* r1=jobject r2..r3=args[0..1]
 * Additional args go on the guest stack below the initial SP. */
    uint32_t r2 = 0, r3 = 0;
    if (nargs > 0 && args) r2 = (uint32_t)args[0].i;
    if (nargs > 1 && args) r3 = (uint32_t)args[1].i;
    /* run_arm always resets SP to STACK_BASE+STACK_SIZE-16, so write extra args
 * to the memory below that fixed SP address before calling run_arm. */
    if (nargs > 2 && args) {
        int extra = nargs - 2;
        uint32_t sp = STACK_BASE + STACK_SIZE - 16 - (uint32_t)(extra * 4);
        for (int i = 0; i < extra; ++i)
            g_ctx->mem.write32(sp + (uint32_t)(i * 4), (uint32_t)args[2 + i].i);
    }
    int rv = run_arm(*g_ctx, fn_va, ENV_SLOT_BASE, (uint32_t)(uintptr_t)obj, r2, r3);
    if(ret) ret->i=rv;
    return 0;
}

extern "C" uint32_t arm_exec_lookup_native(const char *klass, const char *method) {
    if (!g_ctx || !klass || !method) return 0;
    for (auto &rn : g_ctx->natives)
        if (rn.klass == klass && rn.name == method)
            return rn.fn_va;
    return 0;
}

extern "C" uint32_t arm_exec_lookup_native_sig(const char *klass, const char *method,
                                                char *sig_out, int sig_max) {
    if (!g_ctx || !klass || !method) return 0;
    for (auto &rn : g_ctx->natives) {
        if (rn.klass == klass && rn.name == method) {
            if (sig_out && sig_max > 0)
                snprintf(sig_out, (size_t)sig_max, "%s", rn.sig.c_str());
            return rn.fn_va;
        }
    }
    return 0;
}

extern "C" uint32_t arm_exec_lookup_export(const char *sym) {
    if (!sym || !*sym) return 0;
    auto it = g_exported_syms.find(sym);
    return it != g_exported_syms.end() ? it->second : 0;
}

extern "C" int arm_exec_call(uint32_t fn_va, uint32_t r0, uint32_t r1,
                              uint32_t r2, uint32_t r3) {
    if (!g_ctx || !fn_va) return 0;
    /* Re-bind EGL context before each JNI call: Unity 4.x expects the Java side
 * (GLSurfaceView) to make the context current before each call, so it may have
 * released it via eglMakeCurrent(NO_CONTEXT) — ensure it is always current. */
    if (g_egl_ctx != EGL_NO_CONTEXT && g_egl_dpy != EGL_NO_DISPLAY &&
        g_egl_surf != EGL_NO_SURFACE)
        eglMakeCurrent(g_egl_dpy, g_egl_surf, g_egl_surf, g_egl_ctx);
    /* Reset the swap flag so each call gets a fresh tracking window */
    g_arm_did_swap = false;
    /* Budgeted to avoid infinite loops in stubbed-out code paths;
 * override with LUNARIA_CALL_TICKS for slower machines / heavier games. */
    return run_arm(*g_ctx, fn_va, r0, r1, r2, r3,
                   env_ticks("LUNARIA_CALL_TICKS", 2'000'000'000ULL));
}

/* AAPCS stack args 5–6: run_arm fixes sp at STACK_BASE+STACK_SIZE-16 and prewrites slots.
 * nativeResize (IIII)V needs texW/texH on stack; leaving them zero made scale = w/0 = inf. */
extern "C" int arm_exec_call6(uint32_t fn_va, uint32_t r0, uint32_t r1,
                              uint32_t r2, uint32_t r3,
                              uint32_t stk0, uint32_t stk1) {
    if (!g_ctx || !fn_va) return 0;
    const uint32_t sp = STACK_BASE + STACK_SIZE - 16;
    memcpy(g_ctx->mem.ptr(sp),     &stk0, 4);
    memcpy(g_ctx->mem.ptr(sp + 4), &stk1, 4);
    return arm_exec_call(fn_va, r0, r1, r2, r3);
}

extern "C" int arm_exec_call_unlimited(uint32_t fn_va, uint32_t r0, uint32_t r1,
                                        uint32_t r2, uint32_t r3) {
    if (!g_ctx || !fn_va) return 0;
    if (g_egl_ctx != EGL_NO_CONTEXT && g_egl_dpy != EGL_NO_DISPLAY &&
        g_egl_surf != EGL_NO_SURFACE)
        eglMakeCurrent(g_egl_dpy, g_egl_surf, g_egl_surf, g_egl_ctx);
    g_arm_did_swap = false;
    /* No tick limit: caller guarantees the guest function must complete (e.g.
     * nativeRender — abandoning it mid-PlayerLoop leaves Unity's reentrancy
     * guard set, making every subsequent frame bail immediately). */
    return run_arm(*g_ctx, fn_va, r0, r1, r2, r3, UINT64_MAX);
}


extern "C" uint32_t arm_exec_env_va(void) {
    return g_ctx ? ENV_SLOT_BASE : 0;
}

extern "C" void arm_exec_glfw_poll(void) {
    if (g_glfw) glfwPollEvents();
}

extern "C" int arm_exec_glfw_should_close(void) {
    return (g_glfw && glfwWindowShouldClose(g_glfw)) ? 1 : 0;
}

/* Run one cooperative scheduling pass: every runnable guest thread gets a
 * tick-budgeted slice on the auxiliary JIT. */
extern "C" void arm_exec_run_pending_threads(void) {
    schedule_threads(env_ticks("LUNARIA_THREAD_TICKS", 200'000'000ULL));
}

/* UnitySampleGame (and some Unity 4.x titles) open mscorlib via mono_file_map
 * inside initJni / nativeRender before every generic trampoline slot exists. */
extern "C" void arm_exec_ensure_mono_trampolines(void) {
    static bool done = false;
    if (done || !g_ctx || !g_mono_base) return;

    /* mono_trampoline_init fills mono_trampoline_code[0..11] only. */
    if (uint32_t tramp_init = mono_fn_va(nullptr, 0xf0b04u)) {
        (void)run_arm(*g_ctx, tramp_init, 0, 0, 0, 0,
                      env_ticks("LUNARIA_JITINIT_TICKS", 500'000'000ULL));
        arm_exec_run_pending_threads();
    }
    fprintf(stderr, "[arm_exec] ensure_mono_trampolines: ready\n");
    done = true;
}

/* Trampolines first; do NOT call mono_jit_init_version from the host — Unity
 * registers machine config in initJni and jit-inits from nativeRender. */
extern "C" void arm_exec_ensure_mono_jit_init(void) {
    arm_exec_ensure_mono_trampolines();
}

/* After initJni / before mjiv: register libunity's machine.config and mono etc
 * paths so mono_config_parse(NULL) never calls g_file_open(NULL). */
extern "C" void arm_exec_prepare_mono_config(void) {
    static bool done = false;
    if (done || !g_ctx || !g_mono_base) return;

    auto guest_str = [&](const char *s) -> uint32_t {
        if (!s || !*s) return 0u;
        uint32_t va = arm_malloc(*g_ctx, (uint32_t)(strlen(s) + 1));
        if (va) memcpy(g_ctx->mem.ptr(va), s, strlen(s) + 1);
        return va;
    };

    const char *cfg_dir = getenv("MONO_CFG_DIR");
    const char *apk_root = getenv("ANDROID_PACKAGE_CODE_PATH");
    char managed[PATH_MAX] = {};
    if (apk_root)
        snprintf(managed, sizeof managed, "%s/assets/bin/Data/Managed", apk_root);

    /* v1.1.4322 fallback looks under mono/1.0/machine.config */
    if (cfg_dir && !g_unity_machine_config.empty()) {
        for (const char *ver : {"1.0", "2.0", "4.0"}) {
            char dir[PATH_MAX], path[PATH_MAX];
            snprintf(dir, sizeof dir, "%s/mono/%s", cfg_dir, ver);
            mkdir(dir, 0755);
            snprintf(path, sizeof path, "%s/machine.config", dir);
            if (FILE *f = fopen(path, "wb")) {
                fwrite(g_unity_machine_config.data(), 1, g_unity_machine_config.size(), f);
                fclose(f);
            }
        }
    }

    if (managed[0] && cfg_dir && cfg_dir[0]) {
        uint32_t mva = guest_str(managed);
        uint32_t cva = guest_str(cfg_dir);
        if (mva && cva) {
            if (uint32_t fn = mono_fn_va("mono_set_dirs", 0x13a6acu))
                (void)run_arm(*g_ctx, fn, mva, cva, 0, 0, 20'000'000ULL);
        }
    } else if (cfg_dir && cfg_dir[0]) {
        if (uint32_t cva = guest_str(cfg_dir)) {
            if (uint32_t fn = mono_fn_va("mono_set_config_dir", 0x1ef784u))
                (void)run_arm(*g_ctx, fn, cva, 0, 0, 0, 10'000'000ULL);
        }
    }

    if (!g_unity_machine_config.empty()) {
        if (uint32_t sva = guest_str(g_unity_machine_config.c_str())) {
            if (uint32_t fn = mono_fn_va("mono_register_machine_config", 0x1ef83cu))
                (void)run_arm(*g_ctx, fn, sva, 0, 0, 0, 50'000'000ULL);
        }
    }

    /* Full config init once from the host (top-level run_arm).  Export stub noops mjiv. */
    if (g_mono_config_parse_tramp)
        (void)run_arm(*g_ctx, g_mono_config_parse_tramp, 0, 0, 0, 0, 50'000'000ULL);

    if (getenv("LUNARIA_TRACE_MONO")) {
        uint32_t mc = mono_fn_va("mono_get_machine_config", 0x1ef86cu);
        uint32_t cd = mono_fn_va("mono_get_config_dir", 0x1ef7f8u);
        fprintf(stderr, "[arm_exec] prepare_mono_config: machine_config=%#x config_dir=%#x\n",
                mc ? (uint32_t)arm_exec_call(mc, 0, 0, 0, 0) : 0u,
                cd ? (uint32_t)arm_exec_call(cd, 0, 0, 0, 0) : 0u);
    }

    fprintf(stderr, "[arm_exec] prepare_mono_config: done\n");
    done = true;
}

/* Copy mono_get_root_domain into @0x3bd918 when Unity has inited Mono. */
extern "C" void arm_exec_sync_mono_domain_slot(void) {
    if (g_ctx && g_mono_base) ensure_mono_domain_slot(*g_ctx);
}

extern "C" int arm_exec_host_egl_init(void) {
    return init_host_egl() ? 1 : 0;
}

/* Reset callee-saved register state so subsequent calls start clean.
 * Call after a run that may have had stack overflow / exception corruption. */
extern "C" uint32_t arm_exec_read32(uint32_t va) {
    if (!g_ctx) return 0u;
    return g_ctx->mem.read32(va);
}

extern "C" void arm_exec_write32(uint32_t va, uint32_t val) {
    if (g_ctx) g_ctx->mem.write32(va, val);
}

/* Bytes of guest heap consumed by the bump allocator (excludes freelist reuse). */
extern "C" uint32_t arm_exec_heap_used(void) {
    return g_ctx ? (g_ctx->heap_ptr - HEAP_BASE) : 0u;
}

extern "C" const char *arm_exec_get_main_lib_dir(void) {
    return g_main_lib_dir.empty() ? nullptr : g_main_lib_dir.c_str();
}

extern "C" void arm_exec_reset_saved_regs(void) {
    if (g_ctx) {
        std::fill(std::begin(g_ctx->saved_regs), std::end(g_ctx->saved_regs), 0u);
        g_ctx->regs_valid = false;
    }
}

extern "C" void arm_exec_egl_swap(void) {
    if (g_glfw) glfwPollEvents();
    /* Ensure context is current (Unity 4.x may have released it) */
    if (g_egl_ctx != EGL_NO_CONTEXT && g_egl_dpy != EGL_NO_DISPLAY &&
        g_egl_surf != EGL_NO_SURFACE)
        eglMakeCurrent(g_egl_dpy, g_egl_surf, g_egl_surf, g_egl_ctx);
    /* Skip swap if ARM guest already swapped via eglSwapBuffers SVC */
    if (!g_arm_did_swap &&
        g_egl_dpy != EGL_NO_DISPLAY && g_egl_surf != EGL_NO_SURFACE) {
        eglSwapBuffers(g_egl_dpy, g_egl_surf);
        ++g_host_egl_swap_count;
        if (getenv("LUNARIA_TRACE_EGL")) {
            static uint64_t n = 0;
            if (n < 5 || (n % 500) == 0)
                fprintf(stderr, "[egl] host fallback swap #%llu (guest_swaps=%llu glClear=%llu)\n",
                        (unsigned long long)n, (unsigned long long)g_guest_egl_swap_count,
                        (unsigned long long)g_guest_gl_clear_count);
            ++n;
        }
    }
    g_arm_did_swap = false;
}

extern "C" void arm_exec_svc_ring_dump(void) {
    svc_ring_dump();
    /* Also dump main JIT PC and recent block PCs if available */
    if (g_ctx && g_ctx->jit) {
        auto &jit = *g_ctx->jit;
        fprintf(stderr, "[arm_exec] main-jit PC=0x%08x LR=0x%08x R0=0x%08x tid=%u\n",
                (uint32_t)jit.Regs()[15], (uint32_t)jit.Regs()[14],
                (uint32_t)jit.Regs()[0], g_current_tid);
    }
}
