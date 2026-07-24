/*
 * Copyright © 2026 Yuichiro Nakada / Project Lunaria
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "jvm/jvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Detect whether the ELF at `path` is an ARM 32-bit shared object.
 * Returns 1 if ARM, 0 otherwise.
 */
int arm_elf_is_arm32(const char *path);

/*
 * Load and execute an ARM 32-bit JNI library under dynarmic emulation.
 *
 * Loads all PT_LOAD segments, resolves dynamic imports to SVC trampolines,
 * calls JNI_OnLoad, and registers native methods into `jvm`.
 *
 * Returns the JNI version returned by JNI_OnLoad, or -1 on error.
 */
int arm_exec_jni_onload(const char *path, struct jvm *jvm);

/*
 * Call an ARM native method previously registered via RegisterNatives.
 * `fn_va` is the ARM virtual address of the function.
 * `args`  is the jvalue array (A-variant of the JNI call).
 * `ret`   receives the return value as a jvalue.
 *
 * The caller is responsible for marshalling the correct argument types.
 * Returns 0 on success, -1 if no ARM execution context is active.
 */
int arm_exec_call_native(uint32_t fn_va, JNIEnv *env, jobject obj,
                         const jvalue *args, int nargs, jvalue *ret);

/* Look up the ARM virtual address of a native method registered via RegisterNatives.
 * Returns 0 if no ARM context is active or the method is not found. */
uint32_t arm_exec_lookup_native(const char *klass, const char *method);

/* Like arm_exec_lookup_native but also fills `sig_out` (up to sig_max bytes)
 * with the JNI signature string of the registered method. */
uint32_t arm_exec_lookup_native_sig(const char *klass, const char *method,
                                    char *sig_out, int sig_max);

/* Look up a loaded ELF export by name (e.g. "mono_file_map_override").
 * Returns the guest VA from the first library that exported it, or 0. */
uint32_t arm_exec_lookup_export(const char *sym);

/* Call an ARM function with up to 4 register arguments (r0–r3).
 * Returns the value of r0 after the call. */
int arm_exec_call(uint32_t fn_va, uint32_t r0, uint32_t r1,
                  uint32_t r2, uint32_t r3);

/* arm_exec_call + AAPCS スタック渡し引数 2 個 (JNI の 5 個目以降の引数)。
 * nativeResize(IIII) など、レジスタ 4 本に収まらないシグネチャ用。 */
int arm_exec_call6(uint32_t fn_va, uint32_t r0, uint32_t r1,
                   uint32_t r2, uint32_t r3, uint32_t stk0, uint32_t stk1);

/* Like arm_exec_call but with no tick limit.  Use for calls that MUST run to
 * completion — abandoning them mid-execution (e.g. mid-PlayerLoop) leaves
 * guest state inconsistent.  The caller must ensure the function terminates. */
int arm_exec_call_unlimited(uint32_t fn_va, uint32_t r0, uint32_t r1,
                             uint32_t r2, uint32_t r3);

/* Return the ARM virtual address used as JNIEnv* (ENV_SLOT_BASE),
 * or 0 if no ARM context is active. */
uint32_t arm_exec_env_va(void);

/* Poll GLFW events (call from the render loop). */
void arm_exec_glfw_poll(void);

/* Dump current framebuffer to PPM.  path may be NULL → /tmp/lunaria_NNNN.ppm.
 * Returns 1 on success.  Also: F12 in the window, or `touch /tmp/lunaria-shot`. */
int arm_exec_screenshot(const char *path);

/* Framebuffer / window size (LUNARIA_WIDTH / LUNARIA_HEIGHT, default 1280×720). */
int arm_exec_fb_width(void);
int arm_exec_fb_height(void);

/* Touch input bridge (GLFW mouse → Android MotionEvent).
 * arm_exec_touch_next() pops the next queued event and makes it "current";
 * returns 0 when the queue is empty.  The accessors below return the current
 * event's data — libjvm-android.c's MotionEvent JNI getters call them. */
int       arm_exec_touch_next(void);
int       arm_exec_touch_action(void);  /* 0=DOWN 1=UP 2=MOVE */
float     arm_exec_touch_x(void);
float     arm_exec_touch_y(void);
long long arm_exec_touch_time(void);    /* CLOCK_MONOTONIC ms */
void      arm_exec_touch_push(int action, float x, float y); /* test injection */

/* Returns 1 if the GLFW window close button was pressed, 0 otherwise. */
int arm_exec_glfw_should_close(void);

/* Run all pthread_create-queued ARM thread functions inline (up to 8 passes). */
void arm_exec_run_pending_threads(void);

/* Pre-create Mono generic JIT trampolines before initJni maps mscorlib. */
void arm_exec_ensure_mono_trampolines(void);

/* Alias: trampolines only (no host mono_jit_init_version). */
void arm_exec_ensure_mono_jit_init(void);

/* Register libunity machine.config + mono etc paths before mjiv. */
void arm_exec_prepare_mono_config(void);

/* Copy mono_get_root_domain into @0x3bd918 when Unity has inited Mono. */
void arm_exec_sync_mono_domain_slot(void);

/* Unity 4 GL thread job queue (ConcurrentLinkedQueue.add → executeGLThreadJobs). */
void arm_exec_gl_job_enqueue(uint32_t runnable_handle);
void arm_exec_drain_gl_thread_jobs(void);

/* Initialize the ARM execution context (JNI tables, stack, heap) without
 * loading any ELF.  Call this before arm_exec_load_library to pre-load
 * dependency libraries so their exports are visible when the main library
 * is patched. */
int arm_exec_context_init(struct jvm *jvm);

/* Load an additional ARM32 ELF shared library at `base_addr` into the current
 * ARM execution context. Exported symbols become available for cross-library
 * resolution. Calls JNI_OnLoad if present.
 * Call arm_exec_context_init first, then load dependencies before the main lib. */
int arm_exec_load_library(const char *path, uint32_t base_addr);

/* Initialize host EGL + GLES2 context and make it current.
 * Must be called before nativeRecreateGfxState so Unity's GL calls use a real context.
 * Returns 1 on success, 0 on failure. */
int arm_exec_host_egl_init(void);

/* Call eglSwapBuffers on the host EGL surface (present the rendered frame). */
void arm_exec_egl_swap(void);

/* Read a 32-bit word from the ARM guest address space.
 * Returns 0 if the address is unmapped or no ARM context is active. */
uint32_t arm_exec_read32(uint32_t va);

/* Write a 32-bit word to the ARM guest address space (no-op if no context). */
void arm_exec_write32(uint32_t va, uint32_t val);

/* Bytes of guest heap consumed by the bump allocator (for leak diagnosis). */
uint32_t arm_exec_heap_used(void);

/* How many times guest abort() has been called (mono g_assert, etc.). */
uint64_t arm_exec_guest_abort_count(void);

/* Return the directory of the main ARM library (set when arm_exec_jni_onload
 * is first called).  Used by libjvm-java.c findLibrary to return full paths. */
const char *arm_exec_get_main_lib_dir(void);

/* Reset saved callee-saved registers (R4-R11) to zero.
 * Call after a run that may have corrupted ARM state (stack overflow / exception).
 * Prevents stale register values from being restored into the next JNI call. */
void arm_exec_reset_saved_regs(void);

/* Allocate `size` bytes in the guest heap.  Returns guest VA, or 0 on OOM. */
uint32_t arm_exec_malloc(uint32_t size);

/* Copy a host C string into guest memory (NUL-terminated).  Returns VA or 0. */
uint32_t arm_exec_strdup(const char *s);

/* Build a guest ANativeActivity (+ callbacks) for UE4 NativeActivity entry.
 * Returns the activity VA, or 0 on failure.  `clazz` is a jobject handle. */
uint32_t arm_exec_native_activity_create(uint32_t clazz_handle);

/* Ensure a fake ANativeWindow exists; return its guest VA (for onNativeWindowCreated). */
uint32_t arm_exec_native_window_va(void);

/* -------------------------------------------------------------------------
 * ARM64 (AArch64) execution engine — parallel to the ARM32 engine above.
 * All arm64_exec_* functions mirror their arm_exec_* counterparts but
 * operate on the A64 JIT (Dynarmic::A64::Jit) and load Elf64 binaries.
 * ---------------------------------------------------------------------- */

/* Detect whether the ELF at `path` is an AArch64 (arm64-v8a) shared object. */
int arm64_elf_is_arm64(const char *path);

/* Initialize the ARM64 execution context (JNI tables, stack, heap).
 * Must be called before arm64_exec_load_library or arm64_exec_jni_onload. */
int arm64_exec_context_init(struct jvm *jvm);

/* Load an additional ARM64 ELF shared library into the current A64 context.
 * `base_addr` == 0 → auto-place after previously loaded libraries. */
int arm64_exec_load_library(const char *path, uint64_t base_addr);

/* Load an ARM64 ELF and call JNI_OnLoad. Returns JNI version or -1. */
int arm64_exec_jni_onload(const char *path, struct jvm *jvm);

/* Call an ARM64 function with up to 4 register arguments (x0-x3). */
int64_t arm64_exec_call(uint64_t fn_va, uint64_t x0, uint64_t x1,
                        uint64_t x2, uint64_t x3);

/* A64: nativeResize etc. — first 8 args in x0–x7 (no stack needed for 6). */
int64_t arm64_exec_call6(uint64_t fn_va, uint64_t x0, uint64_t x1,
                         uint64_t x2, uint64_t x3, uint64_t x4, uint64_t x5);

/* Like arm64_exec_call but with no tick limit. */
int64_t arm64_exec_call_unlimited(uint64_t fn_va, uint64_t x0, uint64_t x1,
                                  uint64_t x2, uint64_t x3);

/* Look up an exported symbol VA by name. Returns 0 if not found. */
uint64_t arm64_exec_lookup_export(const char *sym);

/* Look up a RegisterNatives-registered native method VA. */
uint64_t arm64_exec_lookup_native(const char *klass, const char *method);

/* Like arm64_exec_lookup_native but also fills sig_out. */
uint64_t arm64_exec_lookup_native_sig(const char *klass, const char *method,
                                      char *sig_out, int sig_max);

/* Return the A64 JNIEnv* guest VA (ENV_SLOT64_BASE). */
uint64_t arm64_exec_env_va(void);

/* Build and return an ANativeActivity guest struct for UE NativeActivity. */
uint64_t arm64_exec_native_activity_create(uint64_t clazz_handle);

/* Ensure a fake ANativeWindow exists; return its guest VA. */
uint64_t arm64_exec_native_window_va(void);

/* Read / write guest memory (64-bit addresses). */
uint32_t arm64_exec_read32(uint64_t va);
uint64_t arm64_exec_read64(uint64_t va);
void     arm64_exec_write32(uint64_t va, uint32_t val);
void     arm64_exec_write64(uint64_t va, uint64_t val);

/* Allocate `size` bytes in the A64 guest heap. Returns guest VA or 0. */
uint64_t arm64_exec_malloc(uint64_t size);

/* Copy a host C string into A64 guest memory. Returns VA or 0. */
uint64_t arm64_exec_strdup(const char *s);

/* Initialize host EGL+GLES2 context (shared with A32 path). */
int arm64_exec_host_egl_init(void);

/* Poll GLFW events. */
void arm64_exec_glfw_poll(void);

/* Present host EGL surface. */
void arm64_exec_egl_swap(void);

/* Dump framebuffer to PPM. */
int arm64_exec_screenshot(const char *path);

/* Window dimensions. */
int arm64_exec_fb_width(void);
int arm64_exec_fb_height(void);

/* Touch input. */
void arm64_exec_touch_push(int action, float x, float y);

/* Run queued guest threads (cooperative scheduler). */
void arm64_exec_run_pending_threads(void);

/* GLFW window close requested? */
int arm64_exec_glfw_should_close(void);

/* Dump recent SVC ring for diagnostics. */
void arm64_exec_svc_ring_dump(void);

#ifdef __cplusplus
}
#endif
