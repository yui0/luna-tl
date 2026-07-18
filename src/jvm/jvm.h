/*
 * Copyright © 2026 Yuichiro Nakada / Project Lunaria
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "jni.h"
#include <stdbool.h>
#include <stdlib.h>

/* Android 実機の getPackageCodePath() / ApplicationInfo.sourceDir は APK ファイル
 * パスを返す。lunaria-apk.sh は展開先を ANDROID_PACKAGE_CODE_PATH に置くので、
 * JNI では ANDROID_APK_FILE を優先する。 */
static inline const char *lunaria_apk_mount_path(void)
{
   const char *apk = getenv("ANDROID_APK_FILE");
   if (apk && *apk)
      return apk;
   return getenv("ANDROID_PACKAGE_CODE_PATH");
}

struct jvm_string {
   const char *data;
   size_t size;
   bool heap; // if on heap, then `data` should be freed
};

struct jvm_array {
   void *data;
   size_t element_sz, size; // `size` == in elements, `size * element_sz` for bytes
};

struct jvm_class {
   struct jvm_string name;
};

struct jvm_method {
   jclass klass;
   struct jvm_string name, signature;
};

struct jvm_object {
   jclass this_klass;

   union {
      struct jvm_array array;
      struct jvm_method method;
      struct jvm_class klass;
      struct jvm_string string;
   };

   enum jvm_object_type {
      JVM_OBJECT_NONE,
      JVM_OBJECT_OPAQUE,
      JVM_OBJECT_ARRAY,
      JVM_OBJECT_METHOD,
      JVM_OBJECT_CLASS,
      JVM_OBJECT_STRING,
      JVM_OBJECT_LAST,
   } type;
};

struct jvm_native_method {
   struct jvm_method method;
   void *function;
};

struct jvm {
   // [0] object is created on `jvm_init` and it's a class object for defining the class of a class
   // every class object's `this_class` member points back to [0], causing recursion.
   // Every other object or class definition is created lazily as needed, only [0] is special.
   // `jobject`'s we return through JNI are actually (index+1) to this array, not pointers.
   struct jvm_object objects[65536];

   // Native methods registered by the application.
   // Nothing special, but there's no need to access this array either really.
   // You can use `jvm_get_native_method` instead.
   struct jvm_native_method methods[255];

   // These hold the function pointers for our JNI implementation.
   struct JNINativeInterface native;
   struct JNIInvokeInterface invoke;

   // JNI's api is weird.. pointer to a reference of a struct, OK!
   // Developers have to dereference these pointers to call methods from an ... reference.
   // NOTE: These are pointers, and JNI interface passes pointers to these pointers!
   JNIEnv env; // points to native
   JavaVM vm; // points to invoke
};

const char*
jvm_get_class_name(struct jvm *jvm, jobject object);

void*
jvm_get_native_method(struct jvm *jvm, const char *klass, const char *method);

void
jvm_release(struct jvm *jvm);

void
jvm_init(struct jvm *jvm);

struct jvm*
jnienv_get_jvm(JNIEnv *env);

/* Per-File path storage (java/io/File). jobject handles index this table. */
void jni_file_set_path(jobject file, const char *path);
const char *jni_file_get_path(jobject file);
void jni_file_bind_ctor(JNIEnv *env, jobject file, jmethodID ctor, va_list ap);
void jni_file_bind_ctor_a(JNIEnv *env, jobject file, jmethodID ctor, const jvalue *args);
