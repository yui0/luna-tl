# Copyright © 2026 Yuichiro Nakada / Project Lunaria
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

PREFIX ?= /usr/local
BINDIR ?= /bin
LIBDIR ?= /lib
RUNTIMEDIR ?= /lunaria

.SUFFIXES:

RM = rm -f

WARNINGS = -Wall -Wextra -Wpedantic -Wformat=2 -Wstrict-aliasing=3 -Wstrict-overflow=3 -Wstack-usage=4096000 \
	-Wfloat-equal -Wcast-align -Wpointer-arith -Wchar-subscripts -Warray-bounds=2 -Wno-unused-parameter

CFLAGS ?= -g -O2 $(WARNINGS)
CFLAGS += -std=c11
CPPFLAGS ?= -D_FORTIFY_SOURCE=2
CPPFLAGS += -Isrc -DANDROID_X86_LINKER # -DVERBOSE_FUNCTIONS
# Default: x86 (32-bit). Use targets below for other ABIs.

bins = lunaria
libs = runtime/libpthread.so runtime/libdl.so runtime/libc.so runtime/libandroid.so \
       runtime/liblog.so runtime/libEGL.so runtime/libOpenSLES.so runtime/libjvm.so \
       runtime/libm.so runtime/libz.so runtime/libmediandk.so runtime/libGLESv3.so

all: $(bins)

# https://developer.android.com/ndk/guides/abis
# https://android.googlesource.com/platform/ndk/+/ics-mr0/docs/STANDALONE-TOOLCHAIN.html
# https://android.googlesource.com/platform/ndk/+/ics-mr0/docs/CPU-ARCH-ABIS.html
# you can also try compiling with your custom ABI with the all target,
# but compatibility with android binaries is not guaranteed

x86:
	$(MAKE) all \
	    CFLAGS="$(CFLAGS) -march=i686 -mtune=intel -mssse3 -mstackrealign -mfpmath=sse -m32" \
	    LDFLAGS="$(LDFLAGS) -march=i686 -m32"

x86_64:
	$(MAKE) all \
	    CFLAGS="$(CFLAGS) -march=x86-64 -msse4.2 -mpopcnt -m64 -mtune=intel -fPIC" \
	    LDFLAGS="$(LDFLAGS) -march=x86-64 -m64" \
	    CPPFLAGS="$(CPPFLAGS) -UANDROID_X86_LINKER -DANDROID_X86_64_LINKER"

armeabi:
	$(MAKE) all \
	    CFLAGS="$(CFLAGS) -march=armv5te -mthumb" \
	    LDFLAGS="$(LDFLAGS) -march=armv5te"

armeabi-v7a:
	$(MAKE) all \
	    CFLAGS="$(CFLAGS) -march=armv7-a -mfloat-abi=softfp -mthumb" \
	    LDFLAGS="$(LDFLAGS) -march=armv7-a -Wl,--fix-cortex-a8"

armeabi-v7a-neon:
	$(MAKE) all \
	    CFLAGS="$(CFLAGS) -march=armv7-a -mfloat-abi=softfp -mthumb -mfpu=neon" \
	    LDFLAGS="$(LDFLAGS) -march=armv7-a -Wl,--fix-cortex-a8"

arm64-v8a:
	$(MAKE) all \
	    CPPFLAGS="$(CPPFLAGS) -UANDROID_X86_LINKER -DANDROID_AARCH64_LINKER"

trace.o: src/trace.c src/trace.h
	$(CC) $(CFLAGS) -fvisibility=hidden -fPIC $(CPPFLAGS) -D_GNU_SOURCE -c src/trace.c -o $@

runtime/libpthread.so: src/lib/pthread.c
	mkdir -p runtime
	$(CC) $(CFLAGS) -fPIC $(CPPFLAGS) -D_GNU_SOURCE $(LDFLAGS) -shared src/lib/pthread.c -lpthread -lrt -o $@

runtime/libdl.so: trace.o src/linker/dlfcn.c src/linker/linker.c src/linker/linker_environ.c src/linker/rt.c src/linker/strlcpy.c
	mkdir -p runtime
	$(CC) $(CFLAGS) -fPIC $(CPPFLAGS) -D_GNU_SOURCE -DLINKER_DEBUG=1 -DRUNTIMEPATH='"$(PREFIX)$(LIBDIR)$(RUNTIMEDIR)"' \
	    -Wno-pedantic -Wno-variadic-macros -Wno-pointer-to-int-cast -Wno-int-to-pointer-cast -Wno-incompatible-pointer-types \
	    $(LDFLAGS) -shared trace.o \
	    src/linker/dlfcn.c src/linker/linker.c src/linker/linker_environ.c src/linker/rt.c src/linker/strlcpy.c \
	    -ldl -lpthread -o $@

runtime/libc.so: trace.o src/lib/libc.c src/lib/libc-ctype.h src/lib/libc-sysconf.h src/lib/libc-verbose.h
	mkdir -p runtime
	$(CC) $(CFLAGS) -fPIC -Wno-deprecated-declarations $(CPPFLAGS) -D_GNU_SOURCE \
	    -Wl,-wrap,_IO_file_xsputn $(LDFLAGS) -shared \
	    trace.o src/lib/libc.c \
	    `pkg-config --libs libbsd libunwind` -o $@

# Small Android API stubs: one driver (src/lib/stub.c), one -DLUNARIA_STUB_* per .so
STUB_SO = $(CC) $(CFLAGS) -fPIC $(CPPFLAGS) $(LDFLAGS) -Isrc/lib -shared src/lib/stub.c

runtime/libandroid.so:
	mkdir -p runtime
	$(STUB_SO) -DLUNARIA_STUB_ANDROID -o $@ `pkg-config --libs glfw3`

runtime/liblog.so:
	mkdir -p runtime
	$(STUB_SO) -DLUNARIA_STUB_LOG -o $@

runtime/libEGL.so:
	mkdir -p runtime
	$(STUB_SO) -DLUNARIA_STUB_EGL -D_GNU_SOURCE -o $@ -lEGL `pkg-config --libs glfw3`

runtime/libOpenSLES.so: trace.o
	mkdir -p runtime
	$(CC) $(CFLAGS) -Wno-pedantic -fPIC $(CPPFLAGS) $(LDFLAGS) -Isrc/lib -shared trace.o \
	    src/lib/stub.c -DLUNARIA_STUB_OPENSLES -o $@

runtime/libjvm.so: trace.o src/jvm/jvm.c src/jvm/jni_stubs.c
	mkdir -p runtime
	$(CC) $(CFLAGS) -fPIC $(CPPFLAGS) -D_GNU_SOURCE -Wno-pedantic $(LDFLAGS) -shared \
	    trace.o src/jvm/jvm.c src/jvm/jni_stubs.c -o $@

runtime/libm.so:
	mkdir -p runtime
	$(STUB_SO) -DLUNARIA_STUB_MATH -D_GNU_SOURCE -o $@ -ldl -lm

runtime/libz.so:
	mkdir -p runtime
	$(STUB_SO) -DLUNARIA_STUB_ZLIB -o $@ -ldl -lz

runtime/libmediandk.so:
	mkdir -p runtime
	$(STUB_SO) -DLUNARIA_STUB_MEDIANDK -o $@

runtime/libGLESv3.so:
	mkdir -p runtime
	$(STUB_SO) -DLUNARIA_STUB_GLESV3 -D_GNU_SOURCE -o $@ -lGLESv2

# trick linker to link against unversioned libs
libdl.so: runtime/libdl.so
	ln -s runtime/libdl.so $@
libpthread.so: runtime/libpthread.so
	ln -s runtime/libpthread.so $@

# arm_exec.o: compiled with C++20 and dynarmic headers; linked into lunaria
arm_exec.o: src/arm_exec.cpp src/arm_exec.h src/jvm/jvm.h $(DYNARMIC_LIB)
	$(CXX) -std=c++20 -O2 -g -fPIC \
	    $(DYNARMIC_INCS) \
	    -Isrc -D_GNU_SOURCE \
	    -c src/arm_exec.cpp -o $@

# loader.o: compiled as C11 (arm_exec.h is C-compatible)
loader.o: src/loader.c src/arm_exec.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -D_GNU_SOURCE -c src/loader.c -o $@

# lunaria: link with g++ so arm_exec.o (C++) and dynarmic (C++) are handled correctly
lunaria: loader.o arm_exec.o trace.o libdl.so libpthread.so \
       runtime/libpthread.so runtime/libc.so \
       runtime/libandroid.so runtime/liblog.so \
       runtime/libEGL.so runtime/libOpenSLES.so \
       runtime/libjvm.so runtime/libm.so runtime/libz.so
	$(CXX) -std=c++20 -O2 -g \
	    -L. -Wl,-Y,runtime,-rpath,$(PREFIX)$(LIBDIR)$(RUNTIMEDIR) $(LDFLAGS) \
	    loader.o arm_exec.o trace.o \
	    $(DYNARMIC_LIBS) \
	    -ldl -lpthread -ljvm \
	    `pkg-config --libs glfw3` -lEGL -lGLESv2 -lz -o $@

install-bin: $(bins)
	install -Dm755 $(bins) -t "$(DESTDIR)$(PREFIX)$(BINDIR)"

install-lib: $(libs)
	install -Dm755 $(libs) -t "$(DESTDIR)$(PREFIX)$(LIBDIR)$(RUNTIMEDIR)"

install: install-bin install-lib

clean:
	$(RM) $(bins) trace.o arm_exec.o loader.o libdl.so libpthread.so
	$(RM) -r runtime
	$(RM) test/test_dynarmic_arm test/test_unity

test: lunaria test/libunity.so test/test_dynarmic_arm
	sh test/run_tests.sh

# Download a real Unity APK and extract libunity.so for static-analysis tests.
# Source: Daggerfall Unity Android port (open source, MIT-licensed Unity wrapper)
# https://github.com/Vwing/daggerfall-unity-android
LIBUNITY_APK_URL = https://github.com/Vwing/daggerfall-unity-android/releases/download/v1.1.1.8/dfu-mono-32bit-v1.1.1.8_mods-supported.apk
LIBUNITY_APK     = test/dfu-mono-32bit.apk

fetch-libunity: $(LIBUNITY_APK)
	unzip -jo $(LIBUNITY_APK) "lib/armeabi-v7a/*.so" -d test/
	@printf 'Extracted ARM libraries to test/\n'

$(LIBUNITY_APK):
	curl -L "$(LIBUNITY_APK_URL)" -o $@

# Auto-extract libunity.so if APK is already present, or download+extract
test/libunity.so: $(LIBUNITY_APK)
	unzip -jo $(LIBUNITY_APK) "lib/armeabi-v7a/libunity.so" -d test/

# Build dynarmic A32 JIT library (Release, A32 frontend only)
DYNARMIC_DIR     = dynarmic
DYNARMIC_BUILD   = $(DYNARMIC_DIR)/build
DYNARMIC_LIB     = $(DYNARMIC_BUILD)/src/dynarmic/libdynarmic.a
DYNARMIC_FMT_LIB = $(DYNARMIC_BUILD)/externals/fmt/libfmt.a
DYNARMIC_MCL_LIB = $(DYNARMIC_BUILD)/externals/mcl/src/libmcl.a
DYNARMIC_ZYD_LIB = $(DYNARMIC_BUILD)/externals/zydis/libZydis.a
DYNARMIC_ZYC_LIB = $(DYNARMIC_BUILD)/externals/zydis/zycore/libZycore.a

DYNARMIC_INCS = \
	-I$(DYNARMIC_DIR)/src \
	-I$(DYNARMIC_DIR)/externals/mcl/include \
	-I$(DYNARMIC_DIR)/externals/fmt/include \
	-I$(DYNARMIC_DIR)/externals/zydis/include \
	-I$(DYNARMIC_DIR)/externals/zydis/zycore/include \
	-I$(DYNARMIC_BUILD)/externals/zydis \
	-I$(DYNARMIC_BUILD)/externals/zydis/zycore

DYNARMIC_LIBS = \
	$(DYNARMIC_LIB) $(DYNARMIC_FMT_LIB) $(DYNARMIC_MCL_LIB) \
	$(DYNARMIC_ZYD_LIB) $(DYNARMIC_ZYC_LIB)

$(DYNARMIC_LIB):
	cmake -B $(DYNARMIC_BUILD) -S $(DYNARMIC_DIR) \
	    -DDYNARMIC_WARNINGS_AS_ERRORS=OFF \
	    -DDYNARMIC_TESTS=OFF \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DDYNARMIC_FRONTENDS="A32;A64" \
	    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
	cmake --build $(DYNARMIC_BUILD) -j$(shell nproc)

test/test_dynarmic_arm: test/test_dynarmic_arm.cpp $(DYNARMIC_LIB)
	$(CXX) -std=c++20 -O2 -g \
	    $(DYNARMIC_INCS) \
	    test/test_dynarmic_arm.cpp \
	    $(DYNARMIC_LIBS) \
	    -lpthread -o $@

dynarmic-build: $(DYNARMIC_LIB)

# ---------------------------------------------------------------------------
# Between-two-worlds (ShutovKS) — Unity IL2CPP ARM32 APK (MIT-licensed game)
# https://github.com/ShutovKS/Between-two-worlds
# ---------------------------------------------------------------------------
BTW_APK_URL = https://github.com/ShutovKS/Between-two-worlds/releases/download/1.0.5/Android_1.0.5.apk
BTW_APK     = test/btw-android.apk

fetch-btw: $(BTW_APK)
	unzip -p $(BTW_APK) "base/lib/armeabi-v7a/libunity.so" > test/btw_libunity.so
	@printf 'Extracted btw libunity.so to test/btw_libunity.so\n'

$(BTW_APK):
	curl -L "$(BTW_APK_URL)" -o $@

test/btw_libunity.so: $(BTW_APK)
	unzip -p $< "base/lib/armeabi-v7a/libunity.so" > $@

.PHONY: all x86 x86_64 armeabi armeabi-v7a armeabi-v7a-neon arm64-v8a \
        clean install install-bin install-lib test fetch-libunity fetch-btw dynarmic-build
