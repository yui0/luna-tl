/*
 * Copyright © 2026 Yuichiro Nakada / Project Lunaria
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/* Unit tests for Unity ReflectionHelper stubs in jni_stubs.c.
 *
 * Build: cc -std=c11 -g -Isrc -D_GNU_SOURCE \
 *            test/test_unity.c src/jvm/jni_stubs.c -o test/test_unity
 * Run:   ./test/test_unity
 */
#include <stdio.h>
#include <string.h>
#include "jvm/jni.h"

/* jni_stubs.c references these; provide no-op stubs for the unit test. */
void arm_exec_drain_gl_thread_jobs(void) {}
const char *arm_exec_get_main_lib_dir(void) { return "."; }
int arm_exec_touch_action(void) { return 0; }
float arm_exec_touch_x(void) { return 0.f; }
float arm_exec_touch_y(void) { return 0.f; }
long long arm_exec_touch_time(void) { return 0; }

/* functions under test (no public header — forward-declare here) */
jmethodID com_unity3d_player_ReflectionHelper_getMethodID(JNIEnv *env, jobject object, jvalue *values);
jfieldID  com_unity3d_player_ReflectionHelper_getFieldID (JNIEnv *env, jobject object, jvalue *values);

/* ---------- test harness ---------- */
static int s_pass, s_fail;
#define PASS(n)      do { printf("  PASS %s\n", (n)); s_pass++; } while (0)
#define FAIL(n, msg) do { printf("  FAIL %s -- %s\n", (n), (msg)); s_fail++; } while (0)

/* ---------- mock JNI ---------- */
static const char *g_method_name, *g_method_sig;
static const char *g_field_name,  *g_field_sig;
/* opaque types — use a plain int whose address serves as a unique sentinel */
static int fake_mid_storage;
static int fake_fid_storage;
#define FAKE_MID ((jmethodID)&fake_mid_storage)
#define FAKE_FID ((jfieldID)&fake_fid_storage)
static struct JNINativeInterface g_funcs;
/* JNIEnv == const struct JNINativeInterface* */
static JNIEnv g_env = &g_funcs;
static int    g_dummy_obj; /* non-NULL stand-in for jobject */

static const char *mock_GetStringUTFChars(JNIEnv *e, jstring s, jboolean *c)
    { (void)e; (void)c; return (const char *)s; }

static jmethodID mock_GetMethodID(JNIEnv *e, jclass k, const char *n, const char *sig)
    { (void)e; (void)k; g_method_name = n; g_method_sig = sig; return FAKE_MID; }

static jfieldID mock_GetFieldID(JNIEnv *e, jclass k, const char *n, const char *sig)
    { (void)e; (void)k; g_field_name = n; g_field_sig = sig; return FAKE_FID; }

static void reset(void)
{
    memset(&g_funcs, 0, sizeof(g_funcs));
    g_funcs.GetStringUTFChars = mock_GetStringUTFChars;
    g_funcs.GetMethodID       = mock_GetMethodID;
    g_funcs.GetFieldID        = mock_GetFieldID;
    g_method_name = g_method_sig = g_field_name = g_field_sig = NULL;
}

/* ---------- tests ---------- */

static void test_getMethodID_retval_and_args(void)
{
    reset();
    static int dummy_class;
    jvalue vals[4] = {
        { .l = (jobject)&dummy_class },
        { .l = (jobject)"myMethod"   },
        { .l = (jobject)"()V"        },
        { .z = JNI_FALSE },
    };
    jmethodID ret = com_unity3d_player_ReflectionHelper_getMethodID(
        &g_env, (jobject)&g_dummy_obj, vals);

    if (ret != FAKE_MID)
        FAIL("getMethodID/retval", "wrong jmethodID returned");
    else if (!g_method_name || strcmp(g_method_name, "myMethod") != 0)
        FAIL("getMethodID/name", "wrong method name passed to GetMethodID");
    else if (!g_method_sig || strcmp(g_method_sig, "()V") != 0)
        FAIL("getMethodID/sig", "wrong signature passed to GetMethodID");
    else
        PASS("getMethodID");
}

static void test_getMethodID_sig_forwarded(void)
{
    reset();
    static int dummy_class;
    jvalue vals[4] = {
        { .l = (jobject)&dummy_class               },
        { .l = (jobject)"doSomething"               },
        { .l = (jobject)"(Ljava/lang/String;)I"    },
        { .z = JNI_FALSE },
    };
    com_unity3d_player_ReflectionHelper_getMethodID(
        &g_env, (jobject)&g_dummy_obj, vals);

    if (!g_method_sig || strcmp(g_method_sig, "(Ljava/lang/String;)I") != 0)
        FAIL("getMethodID/sig_complex", "complex signature not forwarded correctly");
    else
        PASS("getMethodID/sig_complex");
}

static void test_getFieldID_retval_and_args(void)
{
    reset();
    static int dummy_class;
    jvalue vals[4] = {
        { .l = (jobject)&dummy_class          },
        { .l = (jobject)"myField"             },
        { .l = (jobject)"Ljava/lang/String;"  },
        { .z = JNI_FALSE },
    };
    jfieldID ret = com_unity3d_player_ReflectionHelper_getFieldID(
        &g_env, (jobject)&g_dummy_obj, vals);

    if (ret != FAKE_FID)
        FAIL("getFieldID/retval", "wrong jfieldID returned");
    else if (!g_field_name || strcmp(g_field_name, "myField") != 0)
        FAIL("getFieldID/name", "wrong field name passed to GetFieldID");
    else if (!g_field_sig || strcmp(g_field_sig, "Ljava/lang/String;") != 0)
        FAIL("getFieldID/sig", "wrong signature passed to GetFieldID");
    else
        PASS("getFieldID");
}

static void test_getFieldID_primitive_sig(void)
{
    reset();
    static int dummy_class;
    jvalue vals[4] = {
        { .l = (jobject)&dummy_class },
        { .l = (jobject)"count"      },
        { .l = (jobject)"I"          },
        { .z = JNI_FALSE },
    };
    com_unity3d_player_ReflectionHelper_getFieldID(
        &g_env, (jobject)&g_dummy_obj, vals);

    if (!g_field_sig || strcmp(g_field_sig, "I") != 0)
        FAIL("getFieldID/primitive_sig", "primitive sig not forwarded correctly");
    else
        PASS("getFieldID/primitive_sig");
}

/* ---------- main ---------- */
int main(void)
{
    printf("jni_stubs unity tests\n");
    test_getMethodID_retval_and_args();
    test_getMethodID_sig_forwarded();
    test_getFieldID_retval_and_args();
    test_getFieldID_primitive_sig();
    printf("\n%d passed, %d failed\n", s_pass, s_fail);
    return s_fail ? 1 : 0;
}
