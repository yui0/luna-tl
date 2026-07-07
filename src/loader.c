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
#include <sys/stat.h>
#include "linker/dlfcn.h"
#include "linker/linker.h"
#include "jvm/jvm.h"
#include "arm_exec.h"
#include <link.h>

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
   unity.native_init_jni.fun(&jvm->env, context, context);

   if (unity.native_file.ptr) {
      unity.native_file.fun(&jvm->env, context, jvm->env->NewStringUTF(&jvm->env, getenv("ANDROID_PACKAGE_CODE_PATH")));

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
run_jni_game_arm(struct jvm *jvm)
{
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

#undef LOOKUP2
#undef LOOKUP2_SIG

   if (!va_init_jni || !va_render)
      errx(EXIT_FAILURE, "not a unity jni lib");

   uint32_t env = arm_exec_env_va();
   const jobject context = jvm->native.AllocObject(&jvm->env, jvm->native.FindClass(&jvm->env, "android/app/Activity"));
   uint32_t ctx = (uint32_t)(uintptr_t)context;

   /* libmono.so is preloaded at 0x20000000; log root domain via export lookup. */
   uint32_t mono_root = mono_export_call("mono_get_root_domain");
   fprintf(stderr, "[loader] calling initJni (va=0x%08x) mono_root_domain=0x%08x...\n",
           va_init_jni, mono_root);

   /* JIT trampolines must exist before initJni — Unity 4.x maps mscorlib inside
    * initJni, and mono branches into uninitialized codeman slots without mini_init. */
   arm_exec_ensure_mono_trampolines();
   dump_mono_defaults("after mono trampolines");

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

   /* nativeFile: Android 実機では getPackageCodePath() の「APK ファイル」が
    * 渡され、Unity はそれを ZIP アーカイブとしてマウントする。展開ディレクトリ
    * を渡すと ZIP 署名チェックに失敗してマウントされず、ArchiveFileSystem 経由
    * の相対パス (assets/bin/Data/splash.png 等) が NULL を返す。実 APK パス
    * (ANDROID_APK_FILE) を優先して渡す。 */
   if (va_file) {
      const char *apk = getenv("ANDROID_APK_FILE");
      if (!apk) apk = getenv("ANDROID_PACKAGE_CODE_PATH");
      if (apk) {
         jobject str = jvm->native.NewStringUTF(&jvm->env, apk);
         arm_exec_call(va_file, env, ctx, (uint32_t)(uintptr_t)str, 0);
      }
   }

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
       * Unity 5+ : nativeRecreateGfxState(ILandroid/view/Surface;)V — mode + Surface */
      int unity4_sig = (recreate_sig[0] == '(' && recreate_sig[1] == 'L');
      if (unity4_sig) {
         fprintf(stderr, "[loader] calling nativeRecreateGfxState (Unity4 surface-only)...\n");
         arm_exec_call(va_recreate, env, ctx, (uint32_t)(uintptr_t)fake_surf, 0);
      } else {
         fprintf(stderr, "[loader] calling nativeRecreateGfxState (mode=1)...\n");
         arm_exec_call(va_recreate, env, ctx, 1, (uint32_t)(uintptr_t)fake_surf);
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
   if (va_resize)
      arm_exec_call6(va_resize, env, ctx, 1280, 720, 1280, 720);
   if (va_focus)
      arm_exec_call(va_focus, env, ctx, 1, 0);
   if (va_resume)
      arm_exec_call(va_resume, env, ctx, 0, 0);
   arm_exec_run_pending_threads();

   /* 注意: nativeDone() はここでは呼ばない。Unity 5+ では nativeDone() は
    * UnityPlayer.destroy() からの終了処理であり、レンダーループ前に呼ぶと
    * エンジンが quit 状態になり nativeRender が即 return する。 */

   /* limp mode: nativeRender が失敗(0)を返してもループを続ける。
    * スタブ未実装による一過性の失敗後も他のサブシステムは前進し得る。 */
   int frame_count = 0, fail_streak = 0, last_ok = -1, resized_after_init = 0;
   for (;;) {
      /* LUNARIA_TOUCH_TEST=x,y: フレーム 300 で DOWN、310 で UP を自動注入
       * (X11 フォーカスに依存しないタップ検証用) */
      {
         static float tt_x = -1, tt_y = -1; static int tt_parsed = 0;
         if (!tt_parsed) {
            tt_parsed = 1;
            const char *tt = getenv("LUNARIA_TOUCH_TEST");
            if (tt) sscanf(tt, "%f,%f", &tt_x, &tt_y);
         }
         if (tt_x >= 0) {
            /* フォーカス再送: 実機では surfaceChanged 後に focus が届く。
             * ループ前の nativeFocusChanged はエンジン初期化で上書きされる
             * 疑いがあるため、タップ前に再送して入力ゲートを開く */
            if (frame_count == 250 && va_focus) {
               arm_exec_call(va_focus, env, ctx, 1, 0);
               fprintf(stderr, "[loader] nativeFocusChanged(1) re-sent (frame 250)\n");
            }
            if (frame_count == 300) arm_exec_touch_push(0, tt_x, tt_y);
            if (frame_count == 310) arm_exec_touch_push(1, tt_x, tt_y);
         }
      }
      /* GLFW マウス → MotionEvent 注入 (UnityPlayer.onTouchEvent 相当)。
       * MotionEvent の中身 (action/x/y) は libjvm-android.c の JNI getter が
       * arm_exec_touch_* アクセサ経由で読む。1 フレーム 1 イベント: 実機の
       * タップは DOWN と UP が別フレームに届く。同一フレームに両方入れると
       * Unity の Input 集計でタップと認識されないことがある。 */
      if (va_inject && arm_exec_touch_next()) {
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
         arm_exec_call6(va_resize, env, ctx, 1280, 720, 1280, 720);
         resized_after_init = 1;
         fprintf(stderr, "[loader] nativeResize(1280,720,1280,720) re-sent after first frame\n");
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
      if (ok != last_ok) {
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
