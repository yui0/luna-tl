
/*
 * libmediandk.c — Android libmediandk stub for Lunaria
 *
 * Unity 4.x (Mono) does not call Media NDK functions at runtime; the NEEDED
 * entry exists because libunity.so was linked against it.  All functions are
 * no-op stubs so the bionic dynamic linker can resolve the NEEDED entry
 * without error.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* ---- opaque handle types ---- */
typedef struct AMediaCodec       AMediaCodec;
typedef struct AMediaExtractor   AMediaExtractor;
typedef struct AMediaFormat      AMediaFormat;
typedef struct AMediaMuxer       AMediaMuxer;
typedef struct AMediaCrypto      AMediaCrypto;
typedef struct AMediaDrm         AMediaDrm;

typedef int32_t media_status_t;
#define AMEDIA_OK 0
#define AMEDIA_ERROR_UNKNOWN (-10000)

/* ---- AMediaFormat ---- */

AMediaFormat *
AMediaFormat_new(void)
{
    return calloc(1, sizeof(void *));
}

void
AMediaFormat_delete(AMediaFormat *fmt)
{
    free(fmt);
}

int32_t
AMediaFormat_getInt32(AMediaFormat *fmt, const char *name, int32_t *out)
{
    (void)fmt; (void)name; (void)out;
    return 0;
}

int32_t
AMediaFormat_getInt64(AMediaFormat *fmt, const char *name, int64_t *out)
{
    (void)fmt; (void)name; (void)out;
    return 0;
}

int32_t
AMediaFormat_getFloat(AMediaFormat *fmt, const char *name, float *out)
{
    (void)fmt; (void)name; (void)out;
    return 0;
}

int32_t
AMediaFormat_getString(AMediaFormat *fmt, const char *name, const char **out)
{
    (void)fmt; (void)name; (void)out;
    return 0;
}

void
AMediaFormat_setInt32(AMediaFormat *fmt, const char *name, int32_t value)
{
    (void)fmt; (void)name; (void)value;
}

void
AMediaFormat_setInt64(AMediaFormat *fmt, const char *name, int64_t value)
{
    (void)fmt; (void)name; (void)value;
}

void
AMediaFormat_setFloat(AMediaFormat *fmt, const char *name, float value)
{
    (void)fmt; (void)name; (void)value;
}

void
AMediaFormat_setString(AMediaFormat *fmt, const char *name, const char *value)
{
    (void)fmt; (void)name; (void)value;
}

const char *
AMediaFormat_toString(AMediaFormat *fmt)
{
    (void)fmt;
    return "{}";
}

/* ---- AMediaCodec ---- */

AMediaCodec *
AMediaCodec_createDecoderByType(const char *mime_type)
{
    (void)mime_type;
    return NULL;
}

AMediaCodec *
AMediaCodec_createEncoderByType(const char *mime_type)
{
    (void)mime_type;
    return NULL;
}

AMediaCodec *
AMediaCodec_createCodecByName(const char *name)
{
    (void)name;
    return NULL;
}

media_status_t
AMediaCodec_delete(AMediaCodec *codec)
{
    free(codec);
    return AMEDIA_OK;
}

media_status_t
AMediaCodec_configure(AMediaCodec *codec, const AMediaFormat *fmt,
                      void *surface, AMediaCrypto *crypto, uint32_t flags)
{
    (void)codec; (void)fmt; (void)surface; (void)crypto; (void)flags;
    return AMEDIA_ERROR_UNKNOWN;
}

media_status_t
AMediaCodec_start(AMediaCodec *codec)
{
    (void)codec;
    return AMEDIA_ERROR_UNKNOWN;
}

media_status_t
AMediaCodec_stop(AMediaCodec *codec)
{
    (void)codec;
    return AMEDIA_OK;
}

media_status_t
AMediaCodec_flush(AMediaCodec *codec)
{
    (void)codec;
    return AMEDIA_OK;
}

uint8_t *
AMediaCodec_getInputBuffer(AMediaCodec *codec, size_t idx, size_t *out_size)
{
    (void)codec; (void)idx; (void)out_size;
    return NULL;
}

uint8_t *
AMediaCodec_getOutputBuffer(AMediaCodec *codec, size_t idx, size_t *out_size)
{
    (void)codec; (void)idx; (void)out_size;
    return NULL;
}

/* ---- AMediaExtractor ---- */

AMediaExtractor *
AMediaExtractor_new(void)
{
    return calloc(1, sizeof(void *));
}

media_status_t
AMediaExtractor_delete(AMediaExtractor *ex)
{
    free(ex);
    return AMEDIA_OK;
}

media_status_t
AMediaExtractor_setDataSource(AMediaExtractor *ex, const char *location)
{
    (void)ex; (void)location;
    return AMEDIA_ERROR_UNKNOWN;
}

size_t
AMediaExtractor_getTrackCount(AMediaExtractor *ex)
{
    (void)ex;
    return 0;
}

AMediaFormat *
AMediaExtractor_getTrackFormat(AMediaExtractor *ex, size_t idx)
{
    (void)ex; (void)idx;
    return NULL;
}

media_status_t
AMediaExtractor_selectTrack(AMediaExtractor *ex, size_t idx)
{
    (void)ex; (void)idx;
    return AMEDIA_ERROR_UNKNOWN;
}

int64_t
AMediaExtractor_getSampleTime(AMediaExtractor *ex)
{
    (void)ex;
    return -1;
}

int
AMediaExtractor_readSampleData(AMediaExtractor *ex, uint8_t *buf, size_t capacity)
{
    (void)ex; (void)buf; (void)capacity;
    return -1;
}

int32_t
AMediaExtractor_advance(AMediaExtractor *ex)
{
    (void)ex;
    return 0;
}
