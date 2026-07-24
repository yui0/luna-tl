/* libGLESv3.so stub — forwards GLES3 entry points from the host libGLESv2.
 * All symbols are resolved via the host GL dispatch; the guest uses SVC
 * trampolines so this file mainly satisfies DT_NEEDED linking. */
#define GL_GLEXT_PROTOTYPES
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <dlfcn.h>
#include <stddef.h>

/* Provide weak aliases for GLES3 entry points not in gl2.h.
 * These are backed by the host GLES driver loaded at startup. */
static void *glesv3_pfn(const char *name) {
    static void *h;
    if (!h) { h = dlopen("libGLESv2.so.2", RTLD_NOW | RTLD_GLOBAL);
               if (!h) h = dlopen("libGLESv2.so",   RTLD_NOW | RTLD_GLOBAL); }
    return h ? dlsym(h, name) : NULL;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

#define GLES3_FWD(ret, name, ...) \
    __attribute__((visibility("default"))) ret name(__VA_ARGS__)

/* GLES3 symbols that libUnreal.so needs but are absent from GLES2 headers */
GLES3_FWD(void, glGenVertexArrays, int n, unsigned *a)
    { static void(*f)(int,unsigned*); if(!f) f=glesv3_pfn("glGenVertexArrays"); if(f) f(n,a); }
GLES3_FWD(void, glBindVertexArray, unsigned a)
    { static void(*f)(unsigned); if(!f) f=glesv3_pfn("glBindVertexArray"); if(f) f(a); }
GLES3_FWD(void, glDeleteVertexArrays, int n, const unsigned *a)
    { static void(*f)(int,const unsigned*); if(!f) f=glesv3_pfn("glDeleteVertexArrays"); if(f) f(n,a); }
GLES3_FWD(void, glTexStorage2D, unsigned t, int l, unsigned f2, int w, int h)
    { static void(*fn)(unsigned,int,unsigned,int,int); if(!fn) fn=glesv3_pfn("glTexStorage2D"); if(fn) fn(t,l,f2,w,h); }
GLES3_FWD(void, glBlitFramebuffer, int sx0,int sy0,int sx1,int sy1,int dx0,int dy0,int dx1,int dy1,unsigned mask,unsigned filter)
    { static void(*fn)(int,int,int,int,int,int,int,int,unsigned,unsigned); if(!fn) fn=glesv3_pfn("glBlitFramebuffer"); if(fn) fn(sx0,sy0,sx1,sy1,dx0,dy0,dx1,dy1,mask,filter); }
