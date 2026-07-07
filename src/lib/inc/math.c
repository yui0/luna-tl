
/*
 * libm.c — Android libm wrapper for Lunaria
 *
 * Provides float/double math functions that bionic-linked ARM32 ELFs import
 * from libm.so.  The ARM32 execution path uses SVC trampolines; this file
 * covers the host-side (bionic-linker, dlsym) path.
 *
 * All functions forward to the system libm via dlsym(RTLD_NEXT).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <stdint.h>

/* Avoid pulling in glibc's macro-based math.h declarations that conflict
 * with our explicit function definitions below. */
#include <math.h>

/* ISO C forbids casting void* to function pointer directly; use a union. */
#define FWD1(ret, name, T0) \
    ret name(T0 a) { \
        static union { void *p; ret (*f)(T0); } u; \
        if (!u.p) u.p = dlsym(RTLD_NEXT, #name); \
        return u.f ? u.f(a) : (ret)0; \
    }
#define FWD2(ret, name, T0, T1) \
    ret name(T0 a, T1 b) { \
        static union { void *p; ret (*f)(T0, T1); } u; \
        if (!u.p) u.p = dlsym(RTLD_NEXT, #name); \
        return u.f ? u.f(a, b) : (ret)0; \
    }

/* ---- float (f-suffix) ---- */
FWD1(float,  sinf,       float)
FWD1(float,  cosf,       float)
FWD1(float,  tanf,       float)
FWD1(float,  asinf,      float)
FWD1(float,  acosf,      float)
FWD1(float,  atanf,      float)
FWD1(float,  sqrtf,      float)
FWD1(float,  expf,       float)
FWD1(float,  exp2f,      float)
FWD1(float,  logf,       float)
FWD1(float,  log2f,      float)
FWD1(float,  log10f,     float)
FWD1(float,  floorf,     float)
FWD1(float,  ceilf,      float)
FWD1(float,  roundf,     float)
FWD1(float,  truncf,     float)
FWD1(float,  fabsf,      float)
FWD1(float,  cbrtf,      float)
FWD1(float,  sinhf,      float)
FWD1(float,  coshf,      float)
FWD1(float,  tanhf,      float)
FWD1(float,  asinhf,     float)
FWD1(float,  acoshf,     float)
FWD1(float,  atanhf,     float)
FWD1(float,  nearbyintf, float)
FWD1(float,  rintf,      float)
FWD1(float,  log1pf,     float)
FWD2(float,  atan2f,     float, float)
FWD2(float,  powf,       float, float)
FWD2(float,  fmodf,      float, float)
FWD2(float,  fminf,      float, float)
FWD2(float,  fmaxf,      float, float)
FWD2(float,  copysignf,  float, float)
FWD2(float,  hypotf,     float, float)
FWD2(float,  remainderf, float, float)

/* ---- double ---- */
FWD1(double, sin,        double)
FWD1(double, cos,        double)
FWD1(double, tan,        double)
FWD1(double, asin,       double)
FWD1(double, acos,       double)
FWD1(double, atan,       double)
FWD1(double, sqrt,       double)
FWD1(double, exp,        double)
FWD1(double, exp2,       double)
FWD1(double, log,        double)
FWD1(double, log2,       double)
FWD1(double, log10,      double)
FWD1(double, floor,      double)
FWD1(double, ceil,       double)
FWD1(double, round,      double)
FWD1(double, trunc,      double)
FWD1(double, fabs,       double)
FWD1(double, cbrt,       double)
FWD1(double, sinh,       double)
FWD1(double, cosh,       double)
FWD1(double, tanh,       double)
FWD1(double, log1p,      double)
FWD1(double, nearbyint,  double)
FWD1(double, rint,       double)
FWD2(double, atan2,      double, double)
FWD2(double, pow,        double, double)
FWD2(double, fmod,       double, double)
FWD2(double, fmin,       double, double)
FWD2(double, fmax,       double, double)
FWD2(double, hypot,      double, double)

double
ldexp(double x, int exp)
{
    static union { void *p; double (*f)(double, int); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "ldexp");
    return u.f ? u.f(x, exp) : x;
}

float
ldexpf(float x, int exp)
{
    static union { void *p; float (*f)(float, int); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "ldexpf");
    return u.f ? u.f(x, exp) : x;
}

double
frexp(double x, int *exp)
{
    static union { void *p; double (*f)(double, int *); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "frexp");
    return u.f ? u.f(x, exp) : x;
}

void
sincosf(float x, float *s, float *c)
{
    static union { void *p; void (*f)(float, float *, float *); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "sincosf");
    if (u.f) { u.f(x, s, c); return; }
    if (s) *s = sinf(x);
    if (c) *c = cosf(x);
}

void
sincos(double x, double *s, double *c)
{
    static union { void *p; void (*f)(double, double *, double *); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "sincos");
    if (u.f) { u.f(x, s, c); return; }
    if (s) *s = sin(x);
    if (c) *c = cos(x);
}

/* isinf / isinff — glibc defines these as macros, but Android bionic has them
 * as real functions.  Implement without calling the macro versions to avoid
 * redefinition conflicts. */
#undef isinf
#undef isinff
#undef __isinf
int
isinf(double x)
{
    uint64_t bits;
    __builtin_memcpy(&bits, &x, 8);
    return ((bits & 0x7fffffffffffffffULL) == 0x7ff0000000000000ULL) ? ((bits >> 63) ? -1 : 1) : 0;
}

int
isinff(float x)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &x, 4);
    return ((bits & 0x7fffffff) == 0x7f800000) ? ((bits >> 31) ? -1 : 1) : 0;
}

int
__isinf(double x)
{
    return isinf(x);
}

/* __fpclassifyf — used by some bionic code for float classification */
int
__fpclassifyf(float x)
{
    return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, (double)x);
}

long double
strtold(const char *nptr, char **endptr)
{
    static union { void *p; long double (*f)(const char *, char **); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "strtold");
    return u.f ? u.f(nptr, endptr) : 0.0L;
}

#include <wchar.h>
double
wcstod(const wchar_t *nptr, wchar_t **endptr)
{
    static union { void *p; double (*f)(const wchar_t *, wchar_t **); } u;
    if (!u.p) u.p = dlsym(RTLD_NEXT, "wcstod");
    return u.f ? u.f(nptr, endptr) : 0.0;
}
