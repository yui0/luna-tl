
/*
 * libz.c — Android libz wrapper for Lunaria
 *
 * Provides zlib functions that bionic-linked ARM32 ELFs import from libz.so.
 * The ARM32 execution path uses SVC trampolines with manual ARM32↔host
 * z_stream layout translation; this file covers the host-side path where
 * struct layouts match and direct forwarding to the system zlib is safe.
 *
 * All functions forward to the system zlib via dlsym(RTLD_NEXT).
 */
#include <dlfcn.h>
#include <zlib.h>

/* ISO C forbids void* → function-pointer casts directly; use a union. */
#define ZCAST(ftype, sym) ({ \
    static union { void *p; ftype f; } _u; \
    if (!_u.p) _u.p = dlsym(RTLD_NEXT, sym); \
    _u.f; })

/* ---- inflate family ---- */

int
inflateInit_(z_streamp strm, const char *version, int stream_size)
{
    static union { void *p; int (*f)(z_streamp, const char *, int); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "inflateInit_");
    return u.f ? u.f(strm, version, stream_size) : Z_STREAM_ERROR;
}

int
inflateInit2_(z_streamp strm, int windowBits, const char *version, int stream_size)
{
    static union { void *p; int (*f)(z_streamp, int, const char *, int); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "inflateInit2_");
    return u.f ? u.f(strm, windowBits, version, stream_size) : Z_STREAM_ERROR;
}

int
inflate(z_streamp strm, int flush)
{
    static union { void *p; int (*f)(z_streamp, int); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "inflate");
    return u.f ? u.f(strm, flush) : Z_STREAM_ERROR;
}

int
inflateEnd(z_streamp strm)
{
    static union { void *p; int (*f)(z_streamp); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "inflateEnd");
    return u.f ? u.f(strm) : Z_STREAM_ERROR;
}

int
inflateReset(z_streamp strm)
{
    static union { void *p; int (*f)(z_streamp); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "inflateReset");
    return u.f ? u.f(strm) : Z_STREAM_ERROR;
}

/* ---- deflate family ---- */

int
deflateInit_(z_streamp strm, int level, const char *version, int stream_size)
{
    static union { void *p; int (*f)(z_streamp, int, const char *, int); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "deflateInit_");
    return u.f ? u.f(strm, level, version, stream_size) : Z_STREAM_ERROR;
}

int
deflateInit2_(z_streamp strm, int level, int method, int windowBits,
              int memLevel, int strategy, const char *version, int stream_size)
{
    static union { void *p; int (*f)(z_streamp, int, int, int, int, int, const char *, int); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "deflateInit2_");
    return u.f ? u.f(strm, level, method, windowBits, memLevel, strategy,
                     version, stream_size) : Z_STREAM_ERROR;
}

int
deflate(z_streamp strm, int flush)
{
    static union { void *p; int (*f)(z_streamp, int); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "deflate");
    return u.f ? u.f(strm, flush) : Z_STREAM_ERROR;
}

int
deflateEnd(z_streamp strm)
{
    static union { void *p; int (*f)(z_streamp); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "deflateEnd");
    return u.f ? u.f(strm) : Z_STREAM_ERROR;
}

int
deflateReset(z_streamp strm)
{
    static union { void *p; int (*f)(z_streamp); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "deflateReset");
    return u.f ? u.f(strm) : Z_STREAM_ERROR;
}

/* ---- checksum ---- */

uLong
crc32(uLong crc, const Bytef *buf, uInt len)
{
    static union { void *p; uLong (*f)(uLong, const Bytef *, uInt); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "crc32");
    return u.f ? u.f(crc, buf, len) : crc;
}

uLong
adler32(uLong adler, const Bytef *buf, uInt len)
{
    static union { void *p; uLong (*f)(uLong, const Bytef *, uInt); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "adler32");
    return u.f ? u.f(adler, buf, len) : adler;
}

/* ---- utility ---- */

const char *
zlibVersion(void)
{
    static union { void *p; const char *(*f)(void); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "zlibVersion");
    return u.f ? u.f() : ZLIB_VERSION;
}

uLong
compressBound(uLong sourceLen)
{
    static union { void *p; uLong (*f)(uLong); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "compressBound");
    return u.f ? u.f(sourceLen) : sourceLen + (sourceLen >> 12) + (sourceLen >> 14) + 11;
}

int
compress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen)
{
    static union { void *p; int (*f)(Bytef *, uLongf *, const Bytef *, uLong); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "compress");
    return u.f ? u.f(dest, destLen, source, sourceLen) : Z_STREAM_ERROR;
}

int
compress2(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen, int level)
{
    static union { void *p; int (*f)(Bytef *, uLongf *, const Bytef *, uLong, int); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "compress2");
    return u.f ? u.f(dest, destLen, source, sourceLen, level) : Z_STREAM_ERROR;
}

int
uncompress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen)
{
    static union { void *p; int (*f)(Bytef *, uLongf *, const Bytef *, uLong); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "uncompress");
    return u.f ? u.f(dest, destLen, source, sourceLen) : Z_STREAM_ERROR;
}
