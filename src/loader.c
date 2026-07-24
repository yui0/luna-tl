/*
 * Copyright © 2026 Yuichiro Nakada / Project Lunaria
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <libgen.h>
#include <dlfcn.h>
#include <elf.h>
#include <err.h>
#include <limits.h>
#include <signal.h>
#include <sys/stat.h>
#include "linker/dlfcn.h"
#include "linker/linker.h"
#include "jvm/jvm.h"
#include "arm_exec.h"
#include <link.h>

/* Exposed from arm_exec.cpp for diagnostic dumps */
extern void arm_exec_svc_ring_dump(void);

static void svc_dump_handler(int sig) {
    (void)sig;
    arm_exec_svc_ring_dump();
}

/* libmono.so @ 0x20000000: mono_defaults struct and key fields */
static uint32_t mono_export_call(const char *sym)
{
   uint32_t va = arm_exec_lookup_export(sym);
   return va ? (uint32_t)arm_exec_call(va, 0, 0, 0, 0) : 0u;
}

static void dump_mono_defaults(const char *when)
{
   if (!getenv("LUNARIA_TRACE_MONO")) return;
   uint32_t base = arm_exec_lookup_export("mono_defaults");
   uint32_t corlib_fn = mono_export_call("mono_get_corlib");
   uint32_t object_fn = mono_export_call("mono_get_object_class");
   uint32_t root_fn  = mono_export_call("mono_get_root_domain");
   uint32_t corlib_asm = mono_export_call("mono_unity_assembly_get_mscorlib");
   if (base) {
      fprintf(stderr,
              "[mono] %s: defaults@%#x corlib=%#x object=%#x void=%#x byte=%#x int32=%#x string=%#x\n",
              when, base,
              arm_exec_read32(base + 0x00u),
              arm_exec_read32(base + 0x04u),
              arm_exec_read32(base + 0x0cu),
              arm_exec_read32(base + 0x08u),
              arm_exec_read32(base + 0x20u),
              arm_exec_read32(base + 0x44u));
   } else {
      fprintf(stderr, "[mono] %s: mono_defaults symbol not found\n", when);
   }
   fprintf(stderr, "[mono] %s: mono_get_corlib()=%#x mono_get_object_class()=%#x mono_get_root_domain()=%#x mono_unity_assembly_get_mscorlib()=%#x\n",
           when, corlib_fn, object_fn, root_fn, corlib_asm);
}

static int
run_jni_game(struct jvm *jvm)
{
   // Works only with unity libs for now
   // XXX: What this basically is that, we port the Java bits to C
   // XXX: This will become unneccessary as we make dalvik interpreter

   struct {
      union {
         void *ptr;
         void (*fun)(JNIEnv*, jobject, jobject);
      } native_init_jni;

      union {
         void *ptr;
         void (*fun)(JNIEnv*, jobject);
      } native_done;

      union {
         void *ptr;
         void (*fun)(JNIEnv*, jobject, jstring);
      } native_file;

      union {
         void *ptr;
         jboolean (*fun)(JNIEnv*, jobject);
      } native_pause;

      union {
         void *ptr;
         void (*fun)(JNIEnv*, jobject, jint, jobject);
      } native_recreate_gfx_state;

      union {
         void *ptr;
         jboolean (*fun)(JNIEnv*, jobject);
      } native_render;

      union {
         void *ptr;
         void (*fun)(JNIEnv*, jobject);
      } native_resume;

      union {
         void *ptr;
         void (*fun)(JNIEnv*, jobject, jboolean);
      } native_focus_changed;

      union {
         void *ptr;
         void (*fun)(JNIEnv*, jobject, jstring);
      } native_set_input_string;

      union {
         void *ptr;
         void (*fun)(JNIEnv*, jobject);
      } native_soft_input_closed;

      union {
         void *ptr;
         void (*fun)(JNIEnv*, jobject, jboolean);
      } native_set_input_canceled;

      union {
         void *ptr;
         void (*fun)(JNIEnv*, jobject, jobject);
      } native_init_www;

      union {
         void *ptr;
         void (*fun)(JNIEnv*, jobject, jobject);
      } native_init_web_request;

      union {
         void *ptr;
         void (*fun)(JNIEnv*, jobject, jlong);
      } native_add_vsync_time;

      union {
         void *ptr;
         void (*fun)(JNIEnv*, jobject, jboolean);
      } native_forward_events_to_dalvik;

      union {
         void *ptr;
         void (*fun)(JNIEnv*, jobject, jobject);
      } native_inject_event;
   } unity;

   static const char *unity_player_class = "com.unity3d.player.UnityPlayer";
   unity.native_init_jni.ptr = jvm_get_native_method(jvm, unity_player_class, "initJni");
   unity.native_done.ptr = jvm_get_native_method(jvm, unity_player_class, "nativeDone");
   unity.native_file.ptr = jvm_get_native_method(jvm, unity_player_class, "nativeFile");
   unity.native_pause.ptr = jvm_get_native_method(jvm, unity_player_class, "nativePause");
   unity.native_recreate_gfx_state.ptr = jvm_get_native_method(jvm, unity_player_class, "nativeRecreateGfxState");
   unity.native_render.ptr = jvm_get_native_method(jvm, unity_player_class, "nativeRender");
   unity.native_resume.ptr = jvm_get_native_method(jvm, unity_player_class, "nativeResume");
   unity.native_focus_changed.ptr = jvm_get_native_method(jvm, unity_player_class, "nativeFocusChanged");
   unity.native_set_input_string.ptr = jvm_get_native_method(jvm, unity_player_class, "nativeSetInputString");
   unity.native_soft_input_closed.ptr = jvm_get_native_method(jvm, unity_player_class, "nativeSoftInputClosed");
   unity.native_set_input_canceled.ptr = jvm_get_native_method(jvm, unity_player_class, "nativeSetInputCanceled");
   unity.native_init_www.ptr = jvm_get_native_method(jvm, unity_player_class, "nativeInitWWW");
   unity.native_init_web_request.ptr = jvm_get_native_method(jvm, unity_player_class, "nativeInitWebRequest");
   unity.native_add_vsync_time.ptr = jvm_get_native_method(jvm, unity_player_class, "nativeAddVSyncTime");
   unity.native_forward_events_to_dalvik.ptr = jvm_get_native_method(jvm, unity_player_class, "nativeForwardEventsToDalvik");
   unity.native_inject_event.ptr = jvm_get_native_method(jvm, unity_player_class, "nativeInjectEvent");

   if (!unity.native_init_jni.ptr)
      errx(EXIT_FAILURE, "not a unity jni lib");

   const jobject context = jvm->native.AllocObject(&jvm->env, jvm->native.FindClass(&jvm->env, "android/app/Activity"));

   if (unity.native_file.ptr) {
      const char *apk = lunaria_apk_mount_path();
      if (apk && *apk)
         unity.native_file.fun(&jvm->env, context, jvm->env->NewStringUTF(&jvm->env, apk));

      DIR *dir;
      const char *obb_dir = getenv("ANDROID_EXTERNAL_OBB_DIR");
      if (obb_dir && (dir = opendir(obb_dir))) {
         for (struct dirent *d; (d = readdir(dir));) {
            if (!strcmp(d->d_name, ".") || !strcmp(d->d_name, ".."))
               continue;

            char path[4096];
            snprintf(path, sizeof(path), "%s/%s", obb_dir, d->d_name);
            unity.native_file.fun(&jvm->env, context, jvm->env->NewStringUTF(&jvm->env, path));
         }
      }
   }

   unity.native_init_jni.fun(&jvm->env, context, context);

   // unity.native_forward_events_to_dalvik.fun(&jvm->env, context, true);
   if (unity.native_init_www.ptr)
      unity.native_init_www.fun(&jvm->env, context, jvm->env->FindClass(&jvm->env, "com/unity3d/player/WWW"));
   if (unity.native_init_web_request.ptr)
      unity.native_init_web_request.fun(&jvm->env, context, jvm->env->FindClass(&jvm->env, "com/unity3d/player/UnityWebRequest"));
   unity.native_recreate_gfx_state.fun(&jvm->env, context, 0, context);
   unity.native_focus_changed.fun(&jvm->env, context, true);
   unity.native_resume.fun(&jvm->env, context);
   unity.native_done.fun(&jvm->env, context);
   // unity.native_add_vsync_time.fun(&jvm->env, context, 0);

   while (unity.native_render.fun(&jvm->env, context)) {
      static int i = 0;
      if (++i >= 10) {
         unity.native_inject_event.fun(&jvm->env, context, jvm->native.AllocObject(&jvm->env, jvm->native.FindClass(&jvm->env, "android/view/MotionEvent")));
         i = 0;
      }
   }

   return EXIT_SUCCESS;
}

static int
run_ue4_game_arm(struct jvm *jvm)
{
   /* UE4 GameActivity is a NativeActivity: entry is ANativeActivity_onCreate,
    * then the engine's android_main / looper thread drives the frame loop. */
   uint32_t va_oncreate = arm_exec_lookup_export("ANativeActivity_onCreate");
   if (!va_oncreate)
      errx(EXIT_FAILURE, "UE4: ANativeActivity_onCreate not found");

   if (!arm_exec_host_egl_init())
      fprintf(stderr, "[loader] host EGL re-init failed\n");

   jobject activity = jvm->native.AllocObject(&jvm->env,
         jvm->native.FindClass(&jvm->env, "com/epicgames/ue4/GameActivity"));
   if (!activity)
      activity = jvm->native.AllocObject(&jvm->env,
            jvm->native.FindClass(&jvm->env, "android/app/NativeActivity"));
   uint32_t act_va = arm_exec_native_activity_create(
         (uint32_t)(uintptr_t)activity);
   if (!act_va)
      errx(EXIT_FAILURE, "UE4: failed to allocate ANativeActivity");

   /* Optional JNI hooks GameActivity calls before/during native startup */
   uint32_t va_set_global = arm_exec_lookup_native(
         "com.epicgames.ue4.GameActivity", "nativeSetGlobalActivity");
   uint32_t va_set_win = arm_exec_lookup_native(
         "com.epicgames.ue4.GameActivity", "nativeSetWindowInfo");
   uint32_t va_set_surf = arm_exec_lookup_native(
         "com.epicgames.ue4.GameActivity", "nativeSetSurfaceViewInfo");
   uint32_t va_resume_init = arm_exec_lookup_native(
         "com.epicgames.ue4.GameActivity", "nativeResumeMainInit");
   uint32_t env = arm_exec_env_va();
   uint32_t ctx = (uint32_t)(uintptr_t)activity;

   if (va_set_global) {
      fprintf(stderr, "[loader] UE4 nativeSetGlobalActivity\n");
      arm_exec_call(va_set_global, env, ctx, ctx, 0);
   }

   fprintf(stderr, "[loader] UE4 ANativeActivity_onCreate @0x%08x act=0x%08x\n",
           va_oncreate, act_va);
   arm_exec_call_unlimited(va_oncreate, act_va, 0, 0, 0);
   arm_exec_run_pending_threads();
   fprintf(stderr, "[loader] UE4 onCreate returned\n");

   /* Do NOT call activity->callbacks->onStart/onResume/onNativeWindowCreated
    * synchronously: the NDK glue's onNativeWindowCreated waits on a cond until
    * android_main processes APP_CMD_INIT_WINDOW, which deadlocks if we hold the
    * main JIT.  Instead write APP_CMD_* into the android_app command pipe and
    * let the event thread drain them while we pump. */
   {
      uint32_t instance = arm_exec_read32(act_va + 28); /* ANativeActivity.instance */
      uint32_t win = arm_exec_native_window_va();
      fprintf(stderr, "[loader] UE4 android_app instance=0x%08x win=0x%08x\n",
              instance, win);
      if (instance) {
         /* Heuristic: scan android_app for a writable pipe fd pair.
          * Observed NDK layout (32-bit bionic): msgread @+0x48. */
         int msgwrite = -1;
         uint32_t pipe_off = 0;
         for (uint32_t off = 64; off < 256; off += 4) {
            int a = (int)arm_exec_read32(instance + off);
            int b = (int)arm_exec_read32(instance + off + 4);
            if (a > 2 && b > 2 && a < 1024 && b < 1024 && a != b) {
               struct stat sa, sb;
               if (fstat(a, &sa) == 0 && fstat(b, &sb) == 0 &&
                   S_ISFIFO(sa.st_mode) && S_ISFIFO(sb.st_mode)) {
                  msgwrite = b;
                  pipe_off = off;
                  fprintf(stderr, "[loader] UE4 cmd pipe @+0x%x read=%d write=%d\n",
                          off, a, b);
                  break;
               }
            }
         }
         if (win) {
            /* Public window @36; pendingWindow sits after mutex/cond/pipe/
             * thread/poll_sources/flags — typically msgread+0x38 (=0x80 when
             * msgread is 0x48).  process_cmd copies pendingWindow → window. */
            arm_exec_write32(instance + 36, win);
            uint32_t pend = pipe_off ? pipe_off + 0x38u : 0x80u;
            arm_exec_write32(instance + pend, win);
            fprintf(stderr, "[loader] UE4 set window@36 pendingWindow@+0x%x = 0x%08x\n",
                    pend, win);
         }
         if (msgwrite >= 0) {
            /* APP_CMD_START=10, RESUME=11, INIT_WINDOW=1, GAINED_FOCUS=6 */
            static const int8_t cmds[] = { 10, 11, 1, 6 };
            for (size_t i = 0; i < sizeof cmds; ++i) {
               int8_t c = cmds[i];
               if (write(msgwrite, &c, 1) != 1)
                  fprintf(stderr, "[loader] UE4 write APP_CMD %d failed\n", c);
               else
                  fprintf(stderr, "[loader] UE4 wrote APP_CMD %d\n", c);
               arm_exec_run_pending_threads();
            }
         }
      }
   }

   int w = arm_exec_fb_width(), h = arm_exec_fb_height();
   if (va_set_win) {
      fprintf(stderr, "[loader] UE4 nativeSetWindowInfo %dx%d\n", w, h);
      arm_exec_call6(va_set_win, env, ctx, (uint32_t)w, (uint32_t)h, 0, 0);
   }
   if (va_set_surf) {
      fprintf(stderr, "[loader] UE4 nativeSetSurfaceViewInfo %dx%d\n", w, h);
      arm_exec_call6(va_set_surf, env, ctx, (uint32_t)w, (uint32_t)h, 0, 0);
   }
   if (va_resume_init) {
      fprintf(stderr, "[loader] UE4 nativeResumeMainInit\n");
      arm_exec_call(va_resume_init, env, ctx, 0, 0);
   }

   signal(SIGUSR1, svc_dump_handler);
   signal(SIGALRM, svc_dump_handler);
   alarm(30);

   int max_frames = 0;
   {
      const char *mf = getenv("LUNARIA_MAX_FRAMES");
      if (mf && *mf) max_frames = atoi(mf);
   }
   if (max_frames <= 0) max_frames = 300; /* default cap for first bring-up */

   fprintf(stderr, "[loader] UE4 entering pump loop (max_frames=%d)\n", max_frames);
   for (int frame = 0; frame < max_frames; ++frame) {
      if (arm_exec_guest_abort_count() > 0) {
         fprintf(stderr, "[loader] guest abort — stopping UE4 loop (frame %d)\n", frame);
         break;
      }
      arm_exec_run_pending_threads();
      arm_exec_egl_swap();
      arm_exec_glfw_poll();
      if (frame < 5 || frame % 50 == 0)
         fprintf(stderr, "[loader] UE4 pump frame %d\n", frame);
      if (arm_exec_glfw_should_close()) break;
      usleep(16000);
   }
   return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------
 * ARM64 UE NativeActivity pump
 * ---------------------------------------------------------------------- */
static int
run_ue4_game_arm64(struct jvm *jvm)
{
   uint64_t va_oncreate = arm64_exec_lookup_export("ANativeActivity_onCreate");
   if (!va_oncreate)
      errx(EXIT_FAILURE, "UE arm64: ANativeActivity_onCreate not found");

   if (!arm64_exec_host_egl_init())
      fprintf(stderr, "[loader] arm64 host EGL init failed\n");

   jobject activity = jvm->native.AllocObject(&jvm->env,
         jvm->native.FindClass(&jvm->env, "com/epicgames/unreal/GameActivity"));
   if (!activity)
      activity = jvm->native.AllocObject(&jvm->env,
            jvm->native.FindClass(&jvm->env, "android/app/NativeActivity"));
   uint64_t act_va = arm64_exec_native_activity_create((uint64_t)(uintptr_t)activity);
   if (!act_va)
      errx(EXIT_FAILURE, "UE arm64: failed to allocate ANativeActivity");

   fprintf(stderr, "[loader] UE arm64 ANativeActivity_onCreate @0x%llx act=0x%llx\n",
           (unsigned long long)va_oncreate, (unsigned long long)act_va);
   arm64_exec_call_unlimited(va_oncreate, act_va, 0, 0, 0);
   arm64_exec_run_pending_threads();
   fprintf(stderr, "[loader] UE arm64 onCreate returned\n");

   /* Deliver APP_CMD_* via the android_app command pipe */
   {
      uint64_t instance_va = arm64_exec_read64(act_va + 56); /* ANativeActivity.instance */
      uint64_t win_va      = arm64_exec_native_window_va();
      fprintf(stderr, "[loader] UE arm64 android_app instance=0x%llx win=0x%llx\n",
              (unsigned long long)instance_va, (unsigned long long)win_va);
      if (instance_va) {
         int msgwrite = -1;
         uint32_t pipe_off = 0;
         /* Scan android_app for a writable pipe fd pair (arm64 layout: offsets bigger) */
         for (uint32_t off = 64; off < 512; off += 8) {
            int a = (int)arm64_exec_read32(instance_va + off);
            int b = (int)arm64_exec_read32(instance_va + off + 4);
            if (a > 2 && b > 2 && a < 1024 && b < 1024 && a != b) {
               struct stat sa, sb;
               if (fstat(a, &sa) == 0 && fstat(b, &sb) == 0 &&
                   S_ISFIFO(sa.st_mode) && S_ISFIFO(sb.st_mode)) {
                  msgwrite = b; pipe_off = off;
                  fprintf(stderr, "[loader] UE arm64 cmd pipe @+0x%x write=%d\n", off, b);
                  break;
               }
            }
         }
         if (win_va) {
            arm64_exec_write64(instance_va + 36, win_va);
            uint32_t pend = pipe_off ? pipe_off + 0x38u : 0x80u;
            arm64_exec_write64(instance_va + pend, win_va);
            fprintf(stderr, "[loader] UE arm64 set window=0x%llx\n",
                    (unsigned long long)win_va);
         }
         if (msgwrite >= 0) {
            static const int8_t cmds[] = { 10, 11, 1, 6 };
            for (size_t i = 0; i < sizeof cmds; ++i) {
               int8_t c = cmds[i];
               if (write(msgwrite, &c, 1) != 1)
                  fprintf(stderr, "[loader] UE arm64 write APP_CMD %d failed\n", c);
               arm64_exec_run_pending_threads();
            }
         }
      }
   }

   signal(SIGUSR1, svc_dump_handler);
   signal(SIGALRM, svc_dump_handler);
   alarm(30);

   int max_frames = 0;
   { const char *mf = getenv("LUNARIA_MAX_FRAMES"); if (mf && *mf) max_frames = atoi(mf); }
   if (max_frames <= 0) max_frames = 300;

   fprintf(stderr, "[loader] UE arm64 entering pump loop (max_frames=%d)\n", max_frames);
   for (int frame = 0; frame < max_frames; ++frame) {
      arm64_exec_run_pending_threads();
      arm64_exec_glfw_poll();
      if (frame < 5 || frame % 50 == 0)
         fprintf(stderr, "[loader] UE arm64 pump frame %d\n", frame);
      if (arm64_exec_glfw_should_close()) break;
      usleep(16000);
   }
   return EXIT_SUCCESS;
}

static int
run_unity_game_arm64(struct jvm *jvm)
{
   static const char *cls     = "com.unity3d.player.UnityPlayer";
   static const char *cls_svc = "com.unity3d.player.UnityPlayerForActivityOrService";

#define LOOKUP2(name) \
   (arm64_exec_lookup_native(cls, name) ?: arm64_exec_lookup_native(cls_svc, name))
#define LOOKUP2_SIG(name, sig_buf) \
   (arm64_exec_lookup_native_sig(cls, name, sig_buf, sizeof(sig_buf)) ?: \
    arm64_exec_lookup_native_sig(cls_svc, name, sig_buf, sizeof(sig_buf)))

   uint64_t va_init_jni = arm64_exec_lookup_native(cls, "initJni");
   uint64_t va_done     = LOOKUP2("nativeDone");
   uint64_t va_render   = LOOKUP2("nativeRender");
   uint64_t va_resume   = LOOKUP2("nativeResume");
   uint64_t va_focus    = LOOKUP2("nativeFocusChanged");
   char recreate_sig[128] = {0};
   uint64_t va_recreate = LOOKUP2_SIG("nativeRecreateGfxState", recreate_sig);
   uint64_t va_inject   = arm64_exec_lookup_native(cls, "nativeInjectEvent");
   uint64_t va_file     = arm64_exec_lookup_native(cls, "nativeFile");
   uint64_t va_resize   = LOOKUP2("nativeResize");
   uint64_t va_fwd_dalv = LOOKUP2("nativeForwardEventsToDalvik");

#undef LOOKUP2
#undef LOOKUP2_SIG

   if (!va_init_jni || !va_render)
      errx(EXIT_FAILURE, "not a unity jni lib (arm64)");

   uint64_t env = arm64_exec_env_va();
   const jobject context = jvm->native.AllocObject(&jvm->env,
         jvm->native.FindClass(&jvm->env, "android/app/Activity"));
   uint64_t ctx = (uint64_t)(uintptr_t)context;

   if (va_file) {
      const char *apk = lunaria_apk_mount_path();
      if (apk && *apk) {
         fprintf(stderr, "[loader] arm64 calling nativeFile (%s)...\n", apk);
         jobject str = jvm->native.NewStringUTF(&jvm->env, apk);
         arm64_exec_call(va_file, env, ctx, (uint64_t)(uintptr_t)str, 0);
         arm64_exec_run_pending_threads();
      }
   }

   fprintf(stderr, "[loader] arm64 calling initJni (va=0x%llx)...\n",
           (unsigned long long)va_init_jni);
   arm64_exec_call(va_init_jni, env, ctx, ctx, 0);
   arm64_exec_run_pending_threads();
   fprintf(stderr, "[loader] arm64 initJni done\n");

   if (!arm64_exec_host_egl_init())
      fprintf(stderr, "[loader] arm64 host EGL re-init failed\n");

   if (va_recreate) {
      const jobject fake_surf = jvm->native.AllocObject(&jvm->env,
            jvm->native.FindClass(&jvm->env, "android/view/Surface"));
      int unity4_sig = (recreate_sig[0] == '(' && recreate_sig[1] == 'L');
      if (unity4_sig) {
         fprintf(stderr, "[loader] arm64 nativeRecreateGfxState (Unity4)...\n");
         arm64_exec_call(va_recreate, env, ctx, (uint64_t)(uintptr_t)fake_surf, 0);
      } else {
         fprintf(stderr, "[loader] arm64 nativeRecreateGfxState (displayId=0)...\n");
         arm64_exec_call(va_recreate, env, ctx, 0, (uint64_t)(uintptr_t)fake_surf);
      }
      arm64_exec_run_pending_threads();
   }

   if (va_resize) {
      int w = arm64_exec_fb_width(), h = arm64_exec_fb_height();
      fprintf(stderr, "[loader] arm64 nativeResize(%d,%d)...\n", w, h);
      arm64_exec_call6(va_resize, env, ctx, (uint64_t)w, (uint64_t)h,
                       (uint64_t)w, (uint64_t)h);
   }
   if (va_focus)
      arm64_exec_call(va_focus, env, ctx, 1, 0);
   if (va_fwd_dalv)
      arm64_exec_call(va_fwd_dalv, env, ctx, 0, 0);
   if (va_resume)
      arm64_exec_call(va_resume, env, ctx, 0, 0);
   arm64_exec_run_pending_threads();

   fprintf(stderr, "[loader] arm64 entering Unity render loop\n");
   signal(SIGUSR1, svc_dump_handler);
   signal(SIGALRM, svc_dump_handler);
   alarm(30);

   int frame_count = 0, fail_streak = 0, last_ok = -1, resized_after_init = 0;
   int max_frames = 0;
   {
      const char *mf = getenv("LUNARIA_MAX_FRAMES");
      if (mf && *mf) max_frames = atoi(mf);
   }
   if (max_frames <= 0) max_frames = 300;

   for (;;) {
      if (max_frames > 0 && frame_count >= max_frames) {
         fprintf(stderr, "[loader] LUNARIA_MAX_FRAMES=%d reached\n", max_frames);
         break;
      }
      if (va_inject && arm_exec_touch_next()) {
         static jobject motion_ev;
         if (!motion_ev)
            motion_ev = jvm->native.AllocObject(&jvm->env,
                  jvm->native.FindClass(&jvm->env, "android/view/MotionEvent"));
         if (va_fwd_dalv)
            arm64_exec_call(va_fwd_dalv, env, ctx, 0, 0);
         arm64_exec_call(va_inject, env, ctx, (uint64_t)(uintptr_t)motion_ev, 0);
      }

      arm_exec_drain_gl_thread_jobs();
      int ok = (int)arm64_exec_call_unlimited(va_render, env, ctx, 0, 0);

      if (!resized_after_init && frame_count >= 1 && va_resize) {
         int w = arm64_exec_fb_width(), h = arm64_exec_fb_height();
         arm64_exec_call6(va_resize, env, ctx, (uint64_t)w, (uint64_t)h,
                          (uint64_t)w, (uint64_t)h);
         resized_after_init = 1;
      }

      arm64_exec_run_pending_threads();
      arm64_exec_egl_swap();
      arm64_exec_glfw_poll();
      ++frame_count;
      if (ok != last_ok || frame_count <= 5 || (frame_count % 50 == 0)) {
         fprintf(stderr, "[loader] arm64 nativeRender → %d (frame %d)\n",
                 ok, frame_count);
         last_ok = ok;
      }
      fail_streak = ok ? 0 : fail_streak + 1;
      if (fail_streak > 3)
         usleep(16000);
      if (arm64_exec_glfw_should_close()) break;
   }

   if (va_done)
      arm64_exec_call(va_done, env, ctx, 0, 0);
   return EXIT_SUCCESS;
}

static int
run_jni_game_arm64(struct jvm *jvm)
{
   /* UE NativeActivity path */
   if (arm64_exec_lookup_export("ANativeActivity_onCreate"))
      return run_ue4_game_arm64(jvm);
   /* UnityPlayer.initJni path (IL2CPP / Mono) */
   if (arm64_exec_lookup_native("com.unity3d.player.UnityPlayer", "initJni"))
      return run_unity_game_arm64(jvm);
   fprintf(stderr, "[loader] arm64: no known entry point\n");
   return EXIT_FAILURE;
}

static int
run_jni_game_arm(struct jvm *jvm)
{
   /* UE4 NativeActivity path (no UnityPlayer.initJni) */
   if (arm_exec_lookup_export("ANativeActivity_onCreate") &&
       !arm_exec_lookup_native("com.unity3d.player.UnityPlayer", "initJni"))
      return run_ue4_game_arm(jvm);

   /* Unity <= 2019: all methods on UnityPlayer.
    * Unity 2020+:  render/lifecycle moved to UnityPlayerForActivityOrService. */
   static const char *cls      = "com.unity3d.player.UnityPlayer";
   static const char *cls_svc  = "com.unity3d.player.UnityPlayerForActivityOrService";

   /* Helper: look up from primary class, fall back to the service class */
#define LOOKUP2(name) \
   (arm_exec_lookup_native(cls, name) ?: arm_exec_lookup_native(cls_svc, name))
#define LOOKUP2_SIG(name, sig_buf) \
   (arm_exec_lookup_native_sig(cls, name, sig_buf, sizeof(sig_buf)) ?: \
    arm_exec_lookup_native_sig(cls_svc, name, sig_buf, sizeof(sig_buf)))

   uint32_t va_init_jni = arm_exec_lookup_native(cls, "initJni");
   uint32_t va_done     = LOOKUP2("nativeDone");
   uint32_t va_render   = LOOKUP2("nativeRender");
   uint32_t va_resume   = LOOKUP2("nativeResume");
   uint32_t va_focus    = LOOKUP2("nativeFocusChanged");
   char recreate_sig[128] = {0};
   uint32_t va_recreate = LOOKUP2_SIG("nativeRecreateGfxState", recreate_sig);
   uint32_t va_inject   = arm_exec_lookup_native(cls, "nativeInjectEvent");
   uint32_t va_file     = arm_exec_lookup_native(cls, "nativeFile");
   uint32_t va_resize   = LOOKUP2("nativeResize");
   uint32_t va_fwd_dalv = LOOKUP2("nativeForwardEventsToDalvik");
   uint32_t fwd_flag_va = 0; /* Unity 5.3 ForwardEventsToDalvik BSS byte */

#undef LOOKUP2
#undef LOOKUP2_SIG

   if (!va_init_jni || !va_render)
      errx(EXIT_FAILURE, "not a unity jni lib");

   uint32_t env = arm_exec_env_va();
   const jobject context = jvm->native.AllocObject(&jvm->env, jvm->native.FindClass(&jvm->env, "android/app/Activity"));
   uint32_t ctx = (uint32_t)(uintptr_t)context;
   uint32_t mono_root = mono_export_call("mono_get_root_domain");

   /* JIT trampolines must exist before initJni — Unity 4.x maps mscorlib inside
    * initJni, and mono branches into uninitialized codeman slots without mini_init. */
   arm_exec_ensure_mono_trampolines();
   dump_mono_defaults("after mono trampolines");

   /* Mount the APK before initJni.  On a real device UnityPlayer passes
    * getPackageCodePath() (the .apk file) via nativeFile during construction,
    * before initJni reads assets/bin/Data.  Calling nativeFile after initJni
    * leaves ArchiveFileSystem unmounted and splash upload runs with unset
    * texture callbacks. */
   if (va_file) {
      const char *apk = lunaria_apk_mount_path();
      if (apk && *apk) {
         fprintf(stderr, "[loader] calling nativeFile (%s)...\n", apk);
         jobject str = jvm->native.NewStringUTF(&jvm->env, apk);
         arm_exec_call(va_file, env, ctx, (uint32_t)(uintptr_t)str, 0);
         arm_exec_run_pending_threads();
      }
   }

   fprintf(stderr, "[loader] calling initJni (va=0x%08x) mono_root_domain=0x%08x...\n",
           va_init_jni, mono_root);

   arm_exec_call(va_init_jni, env, ctx, ctx, 0);
   arm_exec_run_pending_threads();
   fprintf(stderr, "[loader] initJni done: mono_root_domain=0x%08x\n",
           mono_export_call("mono_get_root_domain"));
   dump_mono_defaults("after initJni");

   /* Unity registers paths in initJni; register embedded machine.config before
    * nativeRender calls mono_jit_init_version (gmisc-unix.c:69 otherwise). */
   arm_exec_prepare_mono_config();
   arm_exec_sync_mono_domain_slot();

   /* mono_file_map_open/… are hooked via SVC; leave Unity's override from initJni
    * unless it blocks our hooks (cleared on libunity load if needed). */

   /* EGL は main() の arm_exec_jni_onload より前に初期化済み。
    * 念のため再度 make-current を試みる（arm_exec_host_egl_init は冪等）。 */
   if (!arm_exec_host_egl_init())
      fprintf(stderr, "[loader] host EGL re-init failed — GL calls may be no-ops\n");

   /* initJni/threads may overflow the ARM stack if Mono recurses deeply.
    * Reset saved R4-R11 so nativeRecreateGfxState starts with a clean register state. */
   arm_exec_reset_saved_regs();

   if (va_recreate) {
      const jobject fake_surf = jvm->native.AllocObject(&jvm->env,
            jvm->native.FindClass(&jvm->env, "android/view/Surface"));
      /* Unity 4.x: nativeRecreateGfxState(Landroid/view/Surface;)V  — Surface only
       * Unity 5+ : nativeRecreateGfxState(ILandroid/view/Surface;)V — displayId + Surface
       * Java 側は updateGLDisplay(0, surface) で主ディスプレイ=0 を渡す。
       * 1 を渡すと Surface が G[1] に格納され、メインスレッドが待つ G[0]
       * (libunity 0x02e601a0 相当) が永遠に 0 のまま cond_wait でデッドロック
       * する (Unity 2023 IL2CPP で確認)。 */
      int unity4_sig = (recreate_sig[0] == '(' && recreate_sig[1] == 'L');
      if (unity4_sig) {
         fprintf(stderr, "[loader] calling nativeRecreateGfxState (Unity4 surface-only)...\n");
         arm_exec_call(va_recreate, env, ctx, (uint32_t)(uintptr_t)fake_surf, 0);
      } else {
         fprintf(stderr, "[loader] calling nativeRecreateGfxState (displayId=0)...\n");
         arm_exec_call(va_recreate, env, ctx, 0, (uint32_t)(uintptr_t)fake_surf);
      }
      arm_exec_run_pending_threads();
      fprintf(stderr, "[loader] nativeRecreateGfxState done\n");
   }
   dump_mono_defaults("after nativeRecreateGfxState");
   /* ウィンドウサイズを通知。nativeResize は (IIII)V = (w, h, texW, texH):
    * libunity は touch スケールを xscale=w/texW, yscale=h/texH で計算する
    * (libunity+0x39c7d0)。texW/texH は AAPCS でスタック渡しのため
    * arm_exec_call (レジスタ 4 本のみ) だと 0 を読み scale=inf になり、
    * 以後の全タッチ座標が inf に化ける (GUI.Button が反応しない真因)。 */
   fprintf(stderr, "[loader] calling nativeResize...\n");
   if (va_resize) {
      int w = arm_exec_fb_width(), h = arm_exec_fb_height();
      arm_exec_call6(va_resize, env, ctx, (uint32_t)w, (uint32_t)h,
                     (uint32_t)w, (uint32_t)h);
   }
   fprintf(stderr, "[loader] calling nativeFocusChanged...\n");
   if (va_focus)
      arm_exec_call(va_focus, env, ctx, 1, 0);
   /* APK meta-data unityplayer.ForwardNativeEventsToDalvik=true makes
    * Unity 5.x nativeInjectEvent skip the native queue (return 0) when the
    * forward-to-Dalvik flag byte is set.  We have no Dalvik touch dispatch —
    * inject is our only path — so force the flag clear.  The JNI call itself
    * can no-op if Unity's Scoped* TLS (+0x104) is busy; poke the BSS byte
    * when the Unity 5.3 strb pattern matches. */
   if (va_fwd_dalv) {
      fprintf(stderr, "[loader] nativeForwardEventsToDalvik(false)\n");
      arm_exec_call(va_fwd_dalv, env, ctx, 0, 0);
      /* Unity 5.3.3: strb r7,[r0,#0xc] at fwd+0x118; literals at +0x154/+0x158 */
      if (arm_exec_read32(va_fwd_dalv + 0x118u) == 0xe5c0700cu) {
         uint32_t lit0 = arm_exec_read32(va_fwd_dalv + 0x154u);
         uint32_t lit1 = arm_exec_read32(va_fwd_dalv + 0x158u);
         fwd_flag_va = (va_fwd_dalv + 0x118u) + lit0 + lit1 + 0xcu;
         uint32_t word = arm_exec_read32(fwd_flag_va & ~3u);
         unsigned sh = (fwd_flag_va & 3u) * 8u;
         unsigned cur = (word >> sh) & 0xffu;
         if (cur) {
            arm_exec_write32(fwd_flag_va & ~3u, word & ~(0xffu << sh));
            fprintf(stderr, "[loader] cleared ForwardEventsToDalvik flag "
                    "@ 0x%08x (was %u)\n", fwd_flag_va, cur);
         } else {
            fprintf(stderr, "[loader] ForwardEventsToDalvik flag @ 0x%08x "
                    "already 0\n", fwd_flag_va);
         }
      }
   }
   fprintf(stderr, "[loader] calling nativeResume...\n");
   if (va_resume)
      arm_exec_call(va_resume, env, ctx, 0, 0);
   fprintf(stderr, "[loader] nativeResume done, running pending threads...\n");
   arm_exec_run_pending_threads();
   fprintf(stderr, "[loader] entering render loop\n");
   /* SIGUSR1: dump SVC ring buffer on demand (kill -USR1 <pid>) */
   signal(SIGUSR1, svc_dump_handler);
   /* SIGALRM: auto-dump after 15s to diagnose first-frame hang */
   signal(SIGALRM, svc_dump_handler);
   alarm(15);

   /* 注意: nativeDone() はここでは呼ばない。Unity 5+ では nativeDone() は
    * UnityPlayer.destroy() からの終了処理であり、レンダーループ前に呼ぶと
    * エンジンが quit 状態になり nativeRender が即 return する。 */

   /* limp mode: nativeRender が失敗(0)を返してもループを続ける。
    * スタブ未実装による一過性の失敗後も他のサブシステムは前進し得る。 */
   int frame_count = 0, fail_streak = 0, last_ok = -1, resized_after_init = 0;
   int max_frames = 0;
   {
      const char *mf = getenv("LUNARIA_MAX_FRAMES");
      if (mf && *mf) max_frames = atoi(mf);
   }
   for (;;) {
      if (max_frames > 0 && frame_count >= max_frames) {
         fprintf(stderr, "[loader] LUNARIA_MAX_FRAMES=%d reached — exiting render loop\n",
                 max_frames);
         break;
      }
      /* LUNARIA_TOUCH_TEST=x,y: auto DOWN/UP (default frames 60/70).
       * LUNARIA_TOUCH_FRAME=N  : DOWN frame (UP = N + hold).
       * LUNARIA_TOUCH_HOLD=N   : hold duration in frames (default 10).
       *                          Each frame during the hold injects ACTION_MOVE
       *                          so Unity sees a continuous touch. */
      {
         static float tt_x = -1, tt_y = -1; static int tt_parsed = 0;
         static int tt_frame = 60, tt_hold = 10;
         if (!tt_parsed) {
            tt_parsed = 1;
            const char *tt = getenv("LUNARIA_TOUCH_TEST");
            if (tt) sscanf(tt, "%f,%f", &tt_x, &tt_y);
            const char *tf = getenv("LUNARIA_TOUCH_FRAME");
            if (tf) { int v = atoi(tf); if (v > 0) tt_frame = v; }
            const char *th = getenv("LUNARIA_TOUCH_HOLD");
            if (th) { int v = atoi(th); if (v > 0) tt_hold = v; }
         }
         if (tt_x >= 0) {
            /* フォーカス再送: 実機では surfaceChanged 後に focus が届く。
             * ループ前の nativeFocusChanged はエンジン初期化で上書きされる
             * 疑いがあるため、タップ前に再送して入力ゲートを開く */
            if (frame_count == tt_frame - 10 && tt_frame > 10 && va_focus) {
               arm_exec_call(va_focus, env, ctx, 1, 0);
               fprintf(stderr, "[loader] nativeFocusChanged(1) re-sent (frame %d)\n",
                       frame_count);
            }
            if (frame_count == tt_frame) {
               arm_exec_touch_push(0, tt_x, tt_y);  /* ACTION_DOWN */
               fprintf(stderr, "[loader] TOUCH_TEST DOWN (%.0f,%.0f) frame %d\n",
                       tt_x, tt_y, frame_count);
            }
            /* ACTION_MOVE: send every frame while held so Unity keeps the touch active */
            if (frame_count > tt_frame && frame_count < tt_frame + tt_hold) {
               arm_exec_touch_push(2, tt_x, tt_y);
            }
            if (frame_count == tt_frame + tt_hold) {
               arm_exec_touch_push(1, tt_x, tt_y);  /* ACTION_UP */
               fprintf(stderr, "[loader] TOUCH_TEST UP (%.0f,%.0f) frame %d (hold=%d)\n",
                       tt_x, tt_y, frame_count, tt_hold);
            }
         }
      }
      /* GLFW マウス → MotionEvent 注入 (UnityPlayer.onTouchEvent 相当)。
       * MotionEvent の中身 (action/x/y) は libjvm-android.c の JNI getter が
       * arm_exec_touch_* アクセサ経由で読む。1 フレーム 1 イベント: 実機の
       * タップは DOWN と UP が別フレームに届く。同一フレームに両方入れると
       * Unity の Input 集計でタップと認識されないことがある。 */
      if (va_inject && arm_exec_touch_next()) {
         /* Re-clear in case Java/meta-data path set the flag during startup. */
         if (fwd_flag_va) {
            uint32_t word = arm_exec_read32(fwd_flag_va & ~3u);
            unsigned sh = (fwd_flag_va & 3u) * 8u;
            if ((word >> sh) & 0xffu)
               arm_exec_write32(fwd_flag_va & ~3u, word & ~(0xffu << sh));
         }
         if (va_fwd_dalv)
            arm_exec_call(va_fwd_dalv, env, ctx, 0, 0);
         static jobject motion_ev;
         if (!motion_ev)
            motion_ev = jvm->native.AllocObject(&jvm->env,
                  jvm->native.FindClass(&jvm->env, "android/view/MotionEvent"));
         int handled = arm_exec_call(va_inject, env, ctx,
                                     (uint32_t)(uintptr_t)motion_ev, 0);
         static int inj_log = 0;
         if (inj_log < 100) {
            fprintf(stderr, "[loader] injectEvent action=%d x=%.0f y=%.0f → %d\n",
                    arm_exec_touch_action(), arm_exec_touch_x(),
                    arm_exec_touch_y(), handled);
            ++inj_log;
         }
      }
      /* UnityPlayer GL thread loop: executeGLThreadJobs() then nativeRender().
       * nativeRender MUST run to completion: abandoning it mid-PlayerLoop leaves
       * Unity's reentrancy guard set, making every subsequent frame bail with
       * "PlayerLoop called recursively!".  Use the unlimited variant. */
      arm_exec_drain_gl_thread_jobs();
      int ok = arm_exec_call_unlimited(va_render, env, ctx, 0, 0);
      /* Guest abort() (e.g. mono g_assert after mmap OOM) leaves PlayerLoop
       * inconsistent; clearing a hardcoded guard VA then re-entering floods
       * "PlayerLoop called recursively".  Stop the loop after the first abort. */
      if (arm_exec_guest_abort_count() > 0) {
         fprintf(stderr, "[loader] guest abort seen — stopping render loop (frame %d)\n",
                 frame_count);
         break;
      }
      /* If nativeRender was cut short by a guest fault (NULL call, NoExecuteFault),
       * PlayerLoop's re-entry guard byte at 0x20f1ac90 may still be set to 1,
       * causing every subsequent frame to bail with "PlayerLoop called recursively!".
       * Reset it so the next frame can enter PlayerLoop normally. */
      if (!ok) {
         static const uint32_t PLAYERLOOP_GUARD_VA = 0x20f1ac90u;
         uint32_t guard_word = arm_exec_read32(PLAYERLOOP_GUARD_VA & ~3u);
         if (guard_word & 0xffu) {
            arm_exec_write32(PLAYERLOOP_GUARD_VA & ~3u,
                             guard_word & ~0xffu);
            fprintf(stderr, "[loader] nativeRender fault: cleared PlayerLoop guard (frame %d)\n",
                    frame_count);
         }
      }
      /* Android では surfaceChanged → nativeResize がエンジン初期化後にも
       * 届く。ループ前の nativeResize はエンジン未初期化で無視されるため
       * (画面が 128x128 の既定値のままになる)、初回フレーム完了後に再送する。 */
      if (!resized_after_init && frame_count >= 1 && va_resize) {
         int w = arm_exec_fb_width(), h = arm_exec_fb_height();
         arm_exec_call6(va_resize, env, ctx, (uint32_t)w, (uint32_t)h,
                        (uint32_t)w, (uint32_t)h);
         resized_after_init = 1;
         fprintf(stderr, "[loader] nativeResize(%d,%d,%d,%d) re-sent after first frame\n",
                 w, h, w, h);
      }
      /* LUNARIA_TOUCH_DIAG=cntSyncVA,cntPhaseVA,getTouchVA:
       * 毎フレーム libunity のタッチカウント関数をゲスト呼び出しして
       * 「C# スクリプトが見る値」を直接観測する (診断用)。 */
      {
         static uint32_t dg_cnt_sync, dg_cnt_phase, dg_get; static int dg_parsed;
         if (!dg_parsed) {
            dg_parsed = 1;
            const char *d = getenv("LUNARIA_TOUCH_DIAG");
            if (d) sscanf(d, "%x,%x,%x", &dg_cnt_sync, &dg_cnt_phase, &dg_get);
         }
         if (dg_cnt_sync) {
            int cs = arm_exec_call(dg_cnt_sync, 0, 0, 0, 0);
            int cp = dg_cnt_phase ? arm_exec_call(dg_cnt_phase, 0, 0, 0, 0) : -1;
            static int last_cs = -1, last_cp = -1;
            if (cs != last_cs || cp != last_cp) {
               fprintf(stderr, "[diag] frame=%d touchCount sync=%d phase=%d\n",
                       frame_count, cs, cp);
               last_cs = cs; last_cp = cp;
            }
            if (cs > 0 && dg_get) {
               const uint32_t out = 0x41013800; /* STR_SCRATCH 後半 */
               int ok2 = arm_exec_call(dg_get, 0, out, 0, 0);
               fprintf(stderr, "[diag]   GetTouch(0)=%d id=%d x=%f y=%f "
                       "phase=%u f34=%u f38=%u f3c=%u tap=%u\n", ok2,
                       (int)arm_exec_read32(out),
                       (double)*(float *)&(uint32_t){arm_exec_read32(out + 4)},
                       (double)*(float *)&(uint32_t){arm_exec_read32(out + 8)},
                       arm_exec_read32(out + 0x24), arm_exec_read32(out + 0x34),
                       arm_exec_read32(out + 0x38), arm_exec_read32(out + 0x3c),
                       arm_exec_read32(out + 0x20));
            }
         }
      }
      /* UnityMain などのゲストスレッドにも実行時間を与える */
      arm_exec_run_pending_threads();
      /* Unity はeglSwapBuffersをJava側に任せる場合があるのでここで呼ぶ */
      arm_exec_egl_swap();
      ++frame_count;
      if (ok != last_ok || (frame_count <= 5) || (frame_count % 100 == 0)) {
         fprintf(stderr, "[loader] nativeRender → %d (frame %d)\n", ok, frame_count);
         last_ok = ok;
      }
      if (getenv("LUNARIA_TRACE_HEAP"))
         fprintf(stderr, "[loader] heap used = %u MB (frame %d)\n",
                 arm_exec_heap_used() >> 20, frame_count);
      if (getenv("LUNARIA_TRACE_MONO") &&
          (frame_count == 1 || frame_count == 10 || frame_count == 100))
         dump_mono_defaults("render loop");
      fail_streak = ok ? 0 : fail_streak + 1;
      /* 失敗が続いたらフレームペーシングして CPU/スワップ暴走を防ぐ */
      if (fail_streak > 3)
         usleep(16000);
      if (arm_exec_glfw_should_close()) break;
   }

   /* 終了処理: nativeDone() は UnityPlayer.destroy() 相当 */
   if (va_done)
      arm_exec_call(va_done, env, ctx, 0, 0);

   return EXIT_SUCCESS;
}

__attribute__((optimize(0))) static void
raw_start(void *entry, int argc, const char *argv[])
{
   // XXX: make this part of the linker when it's rewritten
#if ANDROID_X86_LINKER && defined(__i386__)
   __asm__("mov 2*4(%ebp),%eax"); /* entry */
   __asm__("mov 3*4(%ebp),%ecx"); /* argc */
   __asm__("mov 4*4(%ebp),%edx"); /* argv */
   __asm__("mov %edx,%esp"); /* trim stack. */
   __asm__("push %edx"); /* push argv */
   __asm__("push %ecx"); /* push argc */
   __asm__("sub %edx,%edx"); /* no rtld_fini function */
   __asm__("jmp *%eax"); /* goto entry */
#else
   warnx("raw_start not implemented for this asm platform, can't execute binaries.");
#endif
}

int
main(int argc, const char *argv[])
{
   if (argc < 2)
      errx(EXIT_FAILURE, "usage: <elf file or jni library>");

   printf("loading module: %s\n", argv[1]);

   /* ARM64 ELF: use A64 dynarmic emulation path */
   if (arm64_elf_is_arm64(argv[1])) {
      printf("detected ARM64 ELF — using A64 dynarmic emulation\n");
      setenv("GC_DONT_GC", "1", 0);
      setenv("GC_MAXIMUM_HEAP_SIZE", "268435456", 0);
      setenv("GC_INITIAL_HEAP_SIZE", "67108864",  0);
      static struct jvm jvm;
      jvm_init(&jvm);

      if (arm64_exec_context_init(&jvm) < 0)
         errx(EXIT_FAILURE, "arm64_exec_context_init failed");

      /* Pre-load companion libraries from the same directory */
      {
         char dir[4096], libpath[4096];
         struct stat stbuf;
         snprintf(dir, sizeof(dir), "%s", argv[1]);
         char *slash = strrchr(dir, '/');
         if (slash) *(slash + 1) = '\0'; else dir[0] = '\0';

         /* libc++_shared.so first */
         snprintf(libpath, sizeof(libpath), "%s%s", dir, "libc++_shared.so");
         if (stat(libpath, &stbuf) == 0 && arm64_elf_is_arm64(libpath)) {
            printf("preloading arm64 libc++_shared: %s\n", libpath);
            arm64_exec_load_library(libpath, 0);
         }
         /* Unity IL2CPP + Frame Pacing (must precede libunity PLT bind) */
         static const char *unity_deps[] = {
            "libil2cpp.so", "libswappywrapper.so", NULL
         };
         for (int k = 0; unity_deps[k]; k++) {
            snprintf(libpath, sizeof(libpath), "%s%s", dir, unity_deps[k]);
            if (stat(libpath, &stbuf) == 0 && arm64_elf_is_arm64(libpath)) {
               printf("preloading arm64 dep: %s\n", libpath);
               arm64_exec_load_library(libpath, 0);
            }
         }
         /* libpsoservice.so and other UE companion libs */
         static const char *ue_deps[] = {
            "libpsoservice.so", "libhwcpipe.so", NULL
         };
         for (int k = 0; ue_deps[k]; k++) {
            snprintf(libpath, sizeof(libpath), "%s%s", dir, ue_deps[k]);
            if (stat(libpath, &stbuf) == 0 && arm64_elf_is_arm64(libpath)) {
               printf("preloading arm64 dep: %s\n", libpath);
               arm64_exec_load_library(libpath, 0);
            }
         }
      }

      if (!arm64_exec_host_egl_init())
         fprintf(stderr, "[loader] early arm64 host EGL init failed\n");

      int jni_ver = arm64_exec_jni_onload(argv[1], &jvm);
      if (jni_ver < 0)
         errx(EXIT_FAILURE, "arm64_exec_jni_onload failed");
      int ret = run_jni_game_arm64(&jvm);
      jvm_release(&jvm);
      printf("exiting\n");
      return ret;
   }

   /* ARM 32-bit ELF: use dynarmic emulation path */
   if (arm_elf_is_arm32(argv[1])) {
      printf("detected ARM32 ELF — using dynarmic emulation\n");
      /* Boehm GC の stop-the-world はシグナルでスレッドを止めるが、協調
       * スレッドモデルではシグナル配送がなく suspend ack を永遠に待って
       * ハングする。bdwgc が GC_init で参照する GC_DONT_GC で抑止する
       * (環境変数で明示指定されていれば尊重する)。
       * LUNARIA_GC_ENABLE=1 のときは回収を有効化する（リーク抑止の実験/本対応）。
       * 注意: bdwgc は GC_DONT_GC の「存在」で判定するため、有効化時は
       * setenv せず unsetenv しておく。 */
      if (getenv("LUNARIA_GC_ENABLE"))
         unsetenv("GC_DONT_GC");
      else
         setenv("GC_DONT_GC", "1", 0);
      /* Boehm GC computes max_heap_size from the 32-bit address space (~4 GB),
       * producing requests of ~3.7 GB which our mmap bump allocator must reject.
       * With zero heap the GC calls GC_scratch_alloc(0) → ABORT("Bad GET_MEM arg").
       * Cap the heap to 256 MB so the GC gets usable memory without flooding. */
      setenv("GC_MAXIMUM_HEAP_SIZE", "268435456", 0); /* 256 MB */
      setenv("GC_INITIAL_HEAP_SIZE", "67108864",  0); /* 64 MB */
      static struct jvm jvm;
      jvm_init(&jvm);

      /* ARM context を先に初期化して依存ライブラリをプリロードする。
       * libmono.so のエクスポートシンボルを libunity.so のパッチより先に収集する。 */
      if (arm_exec_context_init(&jvm) < 0)
         errx(EXIT_FAILURE, "arm_exec_context_init failed");

      /* 依存ライブラリを引数より同一ディレクトリから探してロードする */
      {
         char dir[4096], libpath[4096];
         struct stat stbuf;
         snprintf(dir, sizeof(dir), "%s", argv[1]);
         /* dirname相当 (末尾スラッシュまで) */
         char *slash = strrchr(dir, '/');
         if (slash) *(slash + 1) = '\0';
         else dir[0] = '\0';

         /* libmono.so または libmonobdwgc-2.0.so を探してロード */
         static const char *mono_candidates[] = {
            "libmono.so", "libmonobdwgc-2.0.so", NULL
         };
         for (int k = 0; mono_candidates[k]; k++) {
            snprintf(libpath, sizeof(libpath), "%s%s", dir, mono_candidates[k]);
            if (stat(libpath, &stbuf) == 0 && arm_elf_is_arm32(libpath)) {
               printf("preloading mono: %s at base 0x20000000\n", libpath);
               arm_exec_load_library(libpath, 0x20000000u);
               break;
            }
         }
         /* libmain.so はロードしない: 中身は Java の NativeLoader 経由で
          * libunity.so を dlopen するだけのスタブで、エミュレーション環境では
          * NativeLoader 待ちでハングする */

         /* libc++_shared.so → libil2cpp.so の順 (IL2CPP ゲーム対応) */
         snprintf(libpath, sizeof(libpath), "%s%s", dir, "libc++_shared.so");
         if (stat(libpath, &stbuf) == 0 && arm_elf_is_arm32(libpath)) {
            printf("preloading libc++_shared: %s\n", libpath);
            arm_exec_load_library(libpath, 0);
         }

         /* libil2cpp.so: libunity.so より先にシンボルテーブルへ登録 */
         snprintf(libpath, sizeof(libpath), "%s%s", dir, "libil2cpp.so");
         if (stat(libpath, &stbuf) == 0 && arm_elf_is_arm32(libpath)) {
            printf("preloading il2cpp: %s\n", libpath);
            arm_exec_load_library(libpath, 0);
         }

         /* libswappywrapper.so (Android Frame Pacing): libunity.so が SwappyGL_* を
          * 呼ぶため、先にロードして PLT を解決しておかないと NULL 呼び出しになる */
         snprintf(libpath, sizeof(libpath), "%s%s", dir, "libswappywrapper.so");
         if (stat(libpath, &stbuf) == 0 && arm_elf_is_arm32(libpath)) {
            printf("preloading swappywrapper: %s\n", libpath);
            arm_exec_load_library(libpath, 0);
         }

         /* UE4 companion libs (DT_NEEDED of libUE4.so / optional plugins) */
         static const char *ue4_deps[] = {
            "libplaycore.so", "libhwcpipe.so", "libtry-alloc-lib.so",
            "libOVRPlugin.so", "libvrapi.so", NULL
         };
         for (int k = 0; ue4_deps[k]; k++) {
            snprintf(libpath, sizeof(libpath), "%s%s", dir, ue4_deps[k]);
            if (stat(libpath, &stbuf) == 0 && arm_elf_is_arm32(libpath)) {
               printf("preloading UE4 dep: %s\n", libpath);
               arm_exec_load_library(libpath, 0);
            }
         }
      }

      /* ホスト側 EGL/GLES2 コンテキストを libunity.so の INIT_ARRAY / JNI_OnLoad よりも
       * 前に作成する。Unity の INIT_ARRAY コンストラクタが eglGetCurrentContext() を
       * チェックして EGL 準備済みなら eglGetProcAddress() で GL 関数ポインタを取得する
       * ため、ここで初期化しておく必要がある。 */
      if (!arm_exec_host_egl_init())
         fprintf(stderr, "[loader] early host EGL init failed\n");

      /* メインライブラリ (libunity.so) をロード & JNI_OnLoad 実行 */
      int jni_ver = arm_exec_jni_onload(argv[1], &jvm);
      if (jni_ver < 0)
         errx(EXIT_FAILURE, "arm_exec_jni_onload failed");
      int ret = run_jni_game_arm(&jvm);
      jvm_release(&jvm);
      printf("exiting\n");
      return ret;
   }

   {
      char abs[PATH_MAX], paths[4096];
      realpath(argv[1], abs);
      snprintf(paths, sizeof(paths), "%s", dirname(abs));
      dl_parse_library_path(paths, ":");
   }

   void *handle;
   if (!(handle = bionic_dlopen(argv[1], RTLD_LOCAL | RTLD_NOW)))
      errx(EXIT_FAILURE, "dlopen failed: %s", bionic_dlerror());

   struct {
      union {
         void *ptr;
         jint (*fun)(void*, void*);
      } JNI_OnLoad;

      union {
         void *ptr;
      } start;
   } entry = {0};

   {
      union {
         char bytes[sizeof(Elf32_Ehdr)];
         Elf32_Ehdr hdr;
      } elf;

      FILE *f;
      if (!(f = fopen(argv[1], "rb")))
         err(EXIT_FAILURE, "fopen(%s)", argv[1]);

      fread(elf.bytes, 1, sizeof(elf.bytes), f);
      fclose(f);

      struct soinfo *si = handle;
      if (elf.hdr.e_entry)
         entry.start.ptr = (void*)(intptr_t)(si->base + elf.hdr.e_entry);
   }

   int ret = EXIT_FAILURE;
   if (entry.start.ptr) {
      printf("jumping to %p\n", entry.start.ptr);
      raw_start(entry.start.ptr, argc - 1, &argv[1]);
   } else if ((entry.JNI_OnLoad.ptr = bionic_dlsym(handle, "JNI_OnLoad"))) {
      struct jvm jvm;
      jvm_init(&jvm);
      entry.JNI_OnLoad.fun(&jvm.vm, NULL);
      ret = run_jni_game(&jvm);
      jvm_release(&jvm);
   } else {
      warnx("no entrypoint found in %s", argv[1]);
   }

   printf("unloading module: %s\n", argv[1]);
   bionic_dlclose(handle);
   printf("exiting\n");
   return ret;
}
