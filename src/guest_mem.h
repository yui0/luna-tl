/*
 * Guest-visible RAM profile for ActivityManager.getMemoryInfo and /proc/meminfo.
 *
 * Env (all optional):
 *   LUNARIA_MEM_TOTAL_MB      — total RAM advertised to the app (default 6144 = 6 GiB)
 *   LUNARIA_MEM_AVAIL_MB      — available RAM (default ≈ 5/6 of total)
 *   LUNARIA_MEM_THRESHOLD_MB  — low-memory threshold (default 48)
 *   LUNARIA_MEM_FREE_MB       — MemFree in /proc/meminfo (default ≈ 2/3 of avail)
 *
 * Note: arm32/arm64-as-32bit guest VA space is still ~4 GiB, so mmap arenas
 * cannot back a full 6 GiB mapping; the reported size drives Unity heuristics
 * while LUNARIA_HEAP_MB / MMAP2 layout control actual bump allocation.
 */
#pragma once

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline long lunaria_env_long(const char *name, long def)
{
    const char *e = getenv(name);
    if (!e || !e[0])
        return def;
    char *end = NULL;
    long v = strtol(e, &end, 0);
    return (end != e) ? v : def;
}

/* Clamp phone-class totals into a sane range for the emulator. */
static inline long lunaria_mem_total_mb(void)
{
    long mb = lunaria_env_long("LUNARIA_MEM_TOTAL_MB", 6144);
    if (mb < 512)
        mb = 512;
    if (mb > 16384)
        mb = 16384;
    return mb;
}

static inline long lunaria_mem_avail_mb(void)
{
    long mb = lunaria_env_long("LUNARIA_MEM_AVAIL_MB", 0);
    if (mb <= 0)
        mb = lunaria_mem_total_mb() * 5 / 6;
    if (mb < 256)
        mb = 256;
    if (mb > lunaria_mem_total_mb())
        mb = lunaria_mem_total_mb();
    return mb;
}

static inline long lunaria_mem_free_mb(void)
{
    long mb = lunaria_env_long("LUNARIA_MEM_FREE_MB", 0);
    if (mb <= 0)
        mb = lunaria_mem_avail_mb() * 2 / 3;
    if (mb < 128)
        mb = 128;
    if (mb > lunaria_mem_avail_mb())
        mb = lunaria_mem_avail_mb();
    return mb;
}

static inline long lunaria_mem_threshold_mb(void)
{
    long mb = lunaria_env_long("LUNARIA_MEM_THRESHOLD_MB", 48);
    if (mb < 8)
        mb = 8;
    return mb;
}

static inline uint64_t lunaria_mem_total_bytes(void)
{
    return (uint64_t)lunaria_mem_total_mb() * 1024ull * 1024ull;
}

static inline uint64_t lunaria_mem_avail_bytes(void)
{
    return (uint64_t)lunaria_mem_avail_mb() * 1024ull * 1024ull;
}

static inline uint64_t lunaria_mem_threshold_bytes(void)
{
    return (uint64_t)lunaria_mem_threshold_mb() * 1024ull * 1024ull;
}

static inline uint64_t lunaria_mem_total_kb(void)
{
    return (uint64_t)lunaria_mem_total_mb() * 1024ull;
}

static inline uint64_t lunaria_mem_avail_kb(void)
{
    return (uint64_t)lunaria_mem_avail_mb() * 1024ull;
}

static inline uint64_t lunaria_mem_free_kb(void)
{
    return (uint64_t)lunaria_mem_free_mb() * 1024ull;
}

#ifdef __cplusplus
}
#endif
