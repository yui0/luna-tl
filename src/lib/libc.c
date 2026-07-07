/*
 * Copyright © 2026 Yuichiro Nakada / Project Lunaria
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <limits.h>
#include <math.h>
#include <dirent.h>
#include <assert.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/user.h> // PAGE_SIZE, PAGE_SHIFT
#include <netdb.h> // h_errno
#include "trace.h"

struct bionic_dirent {
   uint64_t d_ino;
   int64_t d_off;
   unsigned short d_reclen;
   unsigned char d_type;
   char d_name[256];
};

#if defined(ANDROID_X86_LINKER) || defined(ANDROID_X86_64_LINKER)
typedef unsigned long bionic_sigset_t;
struct bionic_sigaction {
   union {
      void (*bsa_handler)(int);
      void (*bsa_sigaction)(int, void*, void*);
   };
   bionic_sigset_t sa_mask;
   int sa_flags;
   void (*sa_restorer)(void);
};
#else
#  error "not implemented for this platform"
#endif

// Stuff that doesn't exist in glibc

#define PROP_NAME_MAX   32
#define PROP_VALUE_MAX  92

int
__system_property_get(const char *name, char *value)
{
   verbose("%s", name);

   if (!strcmp(name, "ro.build.version.sdk"))
      return snprintf(value, PROP_VALUE_MAX, "%d", 15);

   *value = 0;
   return 0;
}

pid_t
gettid(void)
{
   return syscall(SYS_gettid);
}

int
tgkill(int tgid, int tid, int sig)
{
   verbose("%d, %d, %d", tgid, tid, sig);
   return syscall(SYS_tgkill, tgid, tid, sig);
}

int
tkill(int tid, int sig)
{
   verbose("%d, %d", tid, sig);
   return syscall(SYS_tkill, tid, sig);
}

// Stuff needed for runtime compatibility, but not neccessary for linking
// Also stuff that exists in glibc, but needs to be wrapped for runtime compatibility

// Some defines from app-stdio.c as per GNU linker's manual for --wrap:
//    You may wish to provide a __real_malloc function as well, so that links without the
//    --wrap option will succeed. If you do this, you should not put the definition of
//    __real_malloc in the same file as __wrap_malloc; if you do, the assembler may resolve
//    the call before the linker has a chance to wrap it to malloc.

size_t __real_IO_file_xsputn(FILE *f, const void *buf, size_t n) { return 0; }

#include "libc-ctype.h"

const unsigned int bionic___page_size = PAGE_SIZE;

__attribute_const__ int*
bionic___errno(void)
{
   return __errno_location();
}

__attribute_const__ int*
bionic___get_h_errno(void)
{
   return &h_errno;
}

int
bionic_execv(const char *pathname, char *const argv[])
{
   verbose("%s", pathname);
   char const* args[255];
   size_t i = 0;
   args[i++] = "/proc/self/exe";
   args[i++] = pathname;
   for (const size_t x = i; i < sizeof(args) - 1 && argv[i - x]; ++i) args[i] = argv[i - x];
   assert(argv[i] == NULL);
   args[i] = NULL;
   return execv("/proc/self/exe", (char*const*)args);
}

int
bionic_stat(const char *restrict path, struct stat *restrict buf)
{
   verbose("%s", path);
   return stat(path, buf);
}

int
bionic_lstat(const char *restrict path, struct stat *restrict buf)
{
   verbose("%s", path);
   return lstat(path, buf);
}

int
bionic_fstat(int fd, struct stat *buf)
{
   verbose("%d", fd);
   return fstat(fd, buf);
}

int
bionic_fstatat(int dirfd, const char *pathname, void *buf, int flags)
{
   verbose("%d, %s", dirfd, pathname);
   return fstatat(dirfd, pathname, buf, flags);
}

static void
glibc_dirent_to_bionic_dirent(const struct dirent *de, struct bionic_dirent *bde)
{
   assert(bde && de);
   *bde = (struct bionic_dirent){
      .d_ino = de->d_ino,
      .d_off = de->d_off,
      .d_reclen = de->d_reclen,
      .d_type = de->d_type,
   };
   _Static_assert(sizeof(bde->d_name) >= sizeof(de->d_name), "bionic_dirent can't hold dirent's d_name");
   memcpy(bde->d_name, de->d_name, sizeof(bde->d_name));
}

struct bionic_dirent*
bionic_readdir(DIR *dirp)
{
   assert(dirp);
   static struct bionic_dirent bde;
   struct dirent *de;
   if (!(de = readdir(dirp)))
      return NULL;
   glibc_dirent_to_bionic_dirent(de, &bde);
   return &bde;
}

int
bionic_readdir_r(DIR *dirp, struct bionic_dirent *entry, struct bionic_dirent **result)
{
   assert(dirp && entry && result);
   struct dirent de, *der = NULL;

   int ret;
   if ((ret = readdir_r(dirp, &de, &der)) != 0 || !der) {
      *result = NULL;
      return ret;
   }

   glibc_dirent_to_bionic_dirent(der, entry);
   *result = entry;
   return 0;
}

// Need to wrap bunch of signal crap
// https://android.googlesource.com/platform/bionic/+/master/docs/32-bit-abi.md

int
bionic_sigaddset(const bionic_sigset_t *set, int sig)
{
   int bit = sig - 1; // Signal numbers start at 1, but bit positions start at 0.
   unsigned long *local_set = (unsigned long*)set;
   if (!set || bit < 0 || bit >= (int)(8 * sizeof(*set))) {
      errno = EINVAL;
      return -1;
   }
   local_set[bit / LONG_BIT] |= 1UL << (bit % LONG_BIT);
   return 0;
}

int
bionic_sigismember(const bionic_sigset_t *set, int sig)
{
   int bit = sig - 1; // Signal numbers start at 1, but bit positions start at 0.
   const unsigned long *local_set = (const unsigned long*)set;
   if (!set || bit < 0 || bit >= (int)(8 * sizeof(*set))) {
      errno = EINVAL;
      return -1;
   }
   return (int)((local_set[bit / LONG_BIT] >> (bit % LONG_BIT)) & 1);
}

int
bionic_sigaction(int sig, const struct bionic_sigaction *restrict act, struct bionic_sigaction *restrict oact)
{
   verbose("%d, %p, %p", sig, (void*)act, (void*)oact);

   // THREAD_SIGNAL on android used by libbacktrace
   if (sig == 33)
      sig = SIGRTMIN;

   struct sigaction goact = {0}, gact = {0};
   if (act) {
      gact.sa_handler = act->bsa_handler;
      gact.sa_flags = act->sa_flags;
      gact.sa_restorer = act->sa_restorer;

      // delete reserved signals
      // 32 (__SIGRTMIN + 0)        POSIX timers
      // 33 (__SIGRTMIN + 1)        libbacktrace
      // 34 (__SIGRTMIN + 2)        libcore
      // 35 (__SIGRTMIN + 3)        debuggerd -b
      assert(35 < SIGRTMAX);
      for (int signo = 35; signo < SIGRTMAX; ++signo) {
         if (bionic_sigismember(&act->sa_mask, signo))
            sigaddset(&gact.sa_mask, signo);
      }
   }

   const int ret = sigaction(sig, (act ? &gact : NULL), (oact ? &goact : NULL));

   if (oact) {
      *oact = (struct bionic_sigaction){0};
      oact->bsa_handler = goact.sa_handler;
      oact->sa_flags = goact.sa_flags;
      oact->sa_restorer = goact.sa_restorer;

      for (int signo = SIGRTMIN + 3; signo < SIGRTMAX; ++signo) {
         if (sigismember(&goact.sa_mask, signo))
            bionic_sigaddset(&oact->sa_mask, signo);
      }
   }

   return ret;
}

int
bionic___isfinitef(float f)
{
   return isfinite(f);
}

int
bionic___isfinite(float f)
{
   return isfinite(f);
}

void
bionic___assert2(const char* file, int line, const char* function, const char* failed_expression)
{
   fprintf(stderr, "%s:%d: %s: assertion \"%s\" failed\n", file, line, function, failed_expression);
   abort();
}

uintptr_t bionic___stack_chk_guard = 4;

__attribute__((noreturn)) void
bionic___stack_chk_fail(void)
{
   abort();
}

size_t
bionic___strlen_chk(const char *s, size_t s_len)
{
   const size_t ret = strlen(s);
   if (__builtin_expect(ret >= s_len, 0)) {
      fprintf(stderr, "*** strlen read overflow detected ***\n");
      abort();
   }
   return ret;
}

size_t
bionic___fwrite_chk(const void * __restrict buf, size_t size, size_t count, FILE * __restrict stream, size_t buf_size)
{
   size_t total;
   if (__builtin_expect(__builtin_mul_overflow(size, count, &total), 0)) {
      // overflow: trigger the error path in fwrite
      return fwrite(buf, size, count, stream);
   }

   if (__builtin_expect(total > buf_size, 0)) {
      fprintf(stderr, "*** fwrite read overflow detected ***\n");
      abort();
   }

   return fwrite(buf, size, count, stream);
}

char*
bionic___strchr_chk(const char* p, int ch, size_t s_len)
{
   for (;; ++p, s_len--) {
      if (__builtin_expect(s_len == 0, 0)) {
         fprintf(stderr, "*** strchr buffer overrun detected ***\n");
         abort();
      }

      if (*p == ch)
         return (char*)p;
      else if (!*p)
         return NULL;
   }
   assert(0 && "should not happen");
}

char*
bionic___strrchr_chk(const char* p, int ch, size_t s_len)
{
   const char *save;
   for (save = NULL;; ++p, s_len--) {
      if (__builtin_expect(s_len == 0, 0)) {
         fprintf(stderr, "*** strchr buffer overrun detected ***\n");
         abort();
      }

      if (*p == ch)
         save = p;
      else if (!*p)
         return (char*)save;
   }
   assert(0 && "should not happen");
}

#include "libc-sysconf.h"

long
bionic_sysconf(int name)
{
   verbose("0x%x", name);
   return sysconf(bionic_sysconf_to_glibc_sysconf(name));
}

static void
__libc_fini(int signal, void *array)
{
   void** fini_array = (void**)array;

   if (!array || (size_t)fini_array[0] != (size_t)~0)
      return;

   fini_array += 1;

   int count;
   for (count = 0; fini_array[count]; ++count);

   for (; count > 0; --count) {
      const union {
         void *ptr;
         void (*fun)(void);
      } fini = { .ptr = fini_array[count] };

      if ((size_t)fini.ptr != (size_t)~0)
         fini.fun();
   }
}

struct bionic_structors {
   void (**preinit_array)(void);
   void (**init_array)(void);
   void (**fini_array)(void);
};

__attribute__((noreturn)) void
bionic___libc_init(void *raw_args, void (*onexit)(void), int (*slingshot)(int, char**, char**), struct bionic_structors const *const structors)
{
   // linker has already called the constructors

   union {
      struct s {
         uintptr_t argc;
         char **argv;
      } s;
      char bytes[sizeof(struct s)];
   } arg;

   memcpy(arg.bytes, raw_args, sizeof(arg.bytes));

   if (structors->fini_array && on_exit(__libc_fini, structors->fini_array)) {
      fprintf(stderr, "__cxa_atexit failed\n");
      abort();
   }

   exit(slingshot(arg.s.argc, arg.s.argv, arg.s.argv + arg.s.argc + 1));
}

#ifdef VERBOSE_FUNCTIONS
#  include "libc-verbose.h"
#endif

/* ---- stdio (bionic pre-M standard stream compatibility) ---- */

struct bionic___sFILE {
#if defined(__LP64__)
   char _pad[152];
#else
   char _pad[84];
#endif
} __attribute__((aligned(sizeof(void*))));

// Bionic standard stream support pre-M Android
// Post-M it's saner and they point to stdin/stdout/stderr symbols instead
const struct bionic___sFILE bionic___sF[3] = {
   {{ 's', 't', 'd', 'i', 'n' }},
   {{ 's', 't', 'd', 'o', 'u', 't' }},
   {{ 's', 't', 'd', 'e', 'r', 'r' }}
};

static inline FILE*
bionic_file_to_glibc_file(FILE *f)
{
   if (f == (void*)&bionic___sF[0])
      return stdin;
   else if (f == (void*)&bionic___sF[1])
      return stdout;
   else if (f == (void*)&bionic___sF[2])
      return stderr;
   return f;
}

// libstdc++ uses these directly for standard streams, thus we need to wrap em
// and IO_file wraps aren't enough.

int
bionic_fflush(FILE *f)
{
   return fflush(bionic_file_to_glibc_file(f));
}

size_t
bionic_fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)
{
   return fwrite(ptr, size, nmemb, bionic_file_to_glibc_file(stream));
}

int
bionic_putc(int ch, FILE *f)
{
   return putc(ch, bionic_file_to_glibc_file(f));
}

// Wrapping internal glibc VTABLE functions to handle bionic's pre-M crap
// We define __real_IO_file_xsputn above so linker will link our library.

size_t
__wrap_IO_file_xsputn(FILE *f, const void *buf, size_t n)
{
   return __real_IO_file_xsputn(bionic_file_to_glibc_file(f), buf, n);
}

/* ---- SHA-1 (Public Domain, Steve Reid <steve@edmweb.com>) ---- */

typedef struct
{
    uint32_t state[5];
    uint32_t count[2];
    unsigned char buffer[64];
} SHA1_CTX;

#define SHA1HANDSOFF

#define rol(value, bits) (((value) << (bits)) | ((value) >> (32 - (bits))))

#if BYTE_ORDER == LITTLE_ENDIAN
#define blk0(i) (block->l[i] = (rol(block->l[i],24)&0xFF00FF00) \
    |(rol(block->l[i],8)&0x00FF00FF))
#elif BYTE_ORDER == BIG_ENDIAN
#define blk0(i) block->l[i]
#else
#error "Endianness not defined!"
#endif
#define blk(i) (block->l[i&15] = rol(block->l[(i+13)&15]^block->l[(i+8)&15] \
    ^block->l[(i+2)&15]^block->l[i&15],1))

#define R0(v,w,x,y,z,i) z+=((w&(x^y))^y)+blk0(i)+0x5A827999+rol(v,5);w=rol(w,30);
#define R1(v,w,x,y,z,i) z+=((w&(x^y))^y)+blk(i)+0x5A827999+rol(v,5);w=rol(w,30);
#define R2(v,w,x,y,z,i) z+=(w^x^y)+blk(i)+0x6ED9EBA1+rol(v,5);w=rol(w,30);
#define R3(v,w,x,y,z,i) z+=(((w|x)&y)|(w&x))+blk(i)+0x8F1BBCDC+rol(v,5);w=rol(w,30);
#define R4(v,w,x,y,z,i) z+=(w^x^y)+blk(i)+0xCA62C1D6+rol(v,5);w=rol(w,30);

void SHA1Transform(
    uint32_t state[5],
    const unsigned char buffer[64]
)
{
    uint32_t a, b, c, d, e;

    typedef union
    {
        unsigned char c[64];
        uint32_t l[16];
    } CHAR64LONG16;

#ifdef SHA1HANDSOFF
    CHAR64LONG16 block[1];
    memcpy(block, buffer, 64);
#else
    CHAR64LONG16 *block = (const CHAR64LONG16 *) buffer;
#endif
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    R0(a,b,c,d,e, 0); R0(e,a,b,c,d, 1); R0(d,e,a,b,c, 2); R0(c,d,e,a,b, 3);
    R0(b,c,d,e,a, 4); R0(a,b,c,d,e, 5); R0(e,a,b,c,d, 6); R0(d,e,a,b,c, 7);
    R0(c,d,e,a,b, 8); R0(b,c,d,e,a, 9); R0(a,b,c,d,e,10); R0(e,a,b,c,d,11);
    R0(d,e,a,b,c,12); R0(c,d,e,a,b,13); R0(b,c,d,e,a,14); R0(a,b,c,d,e,15);
    R1(e,a,b,c,d,16); R1(d,e,a,b,c,17); R1(c,d,e,a,b,18); R1(b,c,d,e,a,19);
    R2(a,b,c,d,e,20); R2(e,a,b,c,d,21); R2(d,e,a,b,c,22); R2(c,d,e,a,b,23);
    R2(b,c,d,e,a,24); R2(a,b,c,d,e,25); R2(e,a,b,c,d,26); R2(d,e,a,b,c,27);
    R2(c,d,e,a,b,28); R2(b,c,d,e,a,29); R2(a,b,c,d,e,30); R2(e,a,b,c,d,31);
    R2(d,e,a,b,c,32); R2(c,d,e,a,b,33); R2(b,c,d,e,a,34); R2(a,b,c,d,e,35);
    R2(e,a,b,c,d,36); R2(d,e,a,b,c,37); R2(c,d,e,a,b,38); R2(b,c,d,e,a,39);
    R3(a,b,c,d,e,40); R3(e,a,b,c,d,41); R3(d,e,a,b,c,42); R3(c,d,e,a,b,43);
    R3(b,c,d,e,a,44); R3(a,b,c,d,e,45); R3(e,a,b,c,d,46); R3(d,e,a,b,c,47);
    R3(c,d,e,a,b,48); R3(b,c,d,e,a,49); R3(a,b,c,d,e,50); R3(e,a,b,c,d,51);
    R3(d,e,a,b,c,52); R3(c,d,e,a,b,53); R3(b,c,d,e,a,54); R3(a,b,c,d,e,55);
    R3(e,a,b,c,d,56); R3(d,e,a,b,c,57); R3(c,d,e,a,b,58); R3(b,c,d,e,a,59);
    R4(a,b,c,d,e,60); R4(e,a,b,c,d,61); R4(d,e,a,b,c,62); R4(c,d,e,a,b,63);
    R4(b,c,d,e,a,64); R4(a,b,c,d,e,65); R4(e,a,b,c,d,66); R4(d,e,a,b,c,67);
    R4(c,d,e,a,b,68); R4(b,c,d,e,a,69); R4(a,b,c,d,e,70); R4(e,a,b,c,d,71);
    R4(d,e,a,b,c,72); R4(c,d,e,a,b,73); R4(b,c,d,e,a,74); R4(a,b,c,d,e,75);
    R4(e,a,b,c,d,76); R4(d,e,a,b,c,77); R4(c,d,e,a,b,78); R4(b,c,d,e,a,79);
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    a = b = c = d = e = 0;
#ifdef SHA1HANDSOFF
    memset(block, '\0', sizeof(block));
#endif
}

void SHA1Init(SHA1_CTX *context)
{
    context->state[0] = 0x67452301;
    context->state[1] = 0xEFCDAB89;
    context->state[2] = 0x98BADCFE;
    context->state[3] = 0x10325476;
    context->state[4] = 0xC3D2E1F0;
    context->count[0] = context->count[1] = 0;
}

void SHA1Update(SHA1_CTX *context, const unsigned char *data, uint32_t len)
{
    uint32_t i, j;

    j = context->count[0];
    if ((context->count[0] += len << 3) < j)
        context->count[1]++;
    context->count[1] += (len >> 29);
    j = (j >> 3) & 63;
    if ((j + len) > 63) {
        memcpy(&context->buffer[j], data, (i = 64 - j));
        SHA1Transform(context->state, context->buffer);
        for (; i + 63 < len; i += 64)
            SHA1Transform(context->state, &data[i]);
        j = 0;
    } else {
        i = 0;
    }
    memcpy(&context->buffer[j], &data[i], len - i);
}

void SHA1Final(unsigned char digest[20], SHA1_CTX *context)
{
    unsigned i;
    unsigned char finalcount[8];
    unsigned char c;

    for (i = 0; i < 8; i++)
        finalcount[i] = (unsigned char)((context->count[(i >= 4 ? 0 : 1)] >> ((3 - (i & 3)) * 8)) & 255);
    c = 0200;
    SHA1Update(context, &c, 1);
    while ((context->count[0] & 504) != 448) {
        c = 0000;
        SHA1Update(context, &c, 1);
    }
    SHA1Update(context, finalcount, 8);
    for (i = 0; i < 20; i++)
        digest[i] = (unsigned char)((context->state[i >> 2] >> ((3 - (i & 3)) * 8)) & 255);
    memset(context, '\0', sizeof(*context));
    memset(&finalcount, '\0', sizeof(finalcount));
}

void SHA1(char *hash_out, const char *str, int len)
{
    SHA1_CTX ctx;
    int ii;

    SHA1Init(&ctx);
    for (ii = 0; ii < len; ii += 1)
        SHA1Update(&ctx, (const unsigned char*)str + ii, 1);
    SHA1Final((unsigned char*)hash_out, &ctx);
    hash_out[20] = '\0';
}

/* ---- BSD compat (getprogname / setprogname) ---- */

extern const char *__progname;

static char*
strrchr_safe(const char *s, int c)
{
   char *r = (char*)strrchr(s, c);
   return (r ? r : (char*)s);
}

const char*
bionic_getprogname(void)
{
   return __progname;
}

void
bionic_setprogname(const char *progname)
{
   __progname = strrchr_safe(progname, '/') + 1;
}

/* ---- anti-anti-debug (hide TracerPid from /proc/self/status) ---- */

int
bionic_open(const char *path, int oflag, ...)
{
   if (!strcmp(path, "/proc/self/status")) {
      static FILE *faked = NULL;

      if (!faked) {
         static char status[4096];

         {
            FILE *f = fopen(path, "rb");
            assert(f && "/proc/self/status failed to open :/");
            const size_t ret = fread(status, 1, sizeof(status), f);
            assert(ret <= sizeof(status) && "/proc/self/status doesn't fit in 4096 bytes :/");
            fclose(f);
         }

         for (char *s, *e; (s = strstr(status, "TracerPid:\t"));) {
            for (e = s; (size_t)(e - status) < sizeof(status) && *e && *e != '\n'; ++e);
            memmove(s, e, sizeof(status) - (e - status));
            break;
         }

         faked = fmemopen(status, sizeof(status), "rb");
         assert(faked && "fmemopen failed :/");
      }

      return fileno(faked);
   }

   return open(path, oflag);
}
