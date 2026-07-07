/*
 * Copyright © 2026 Yuichiro Nakada / Project Lunaria
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <stddef.h>

#ifdef VERBOSE_FUNCTIONS
void verbose_log(const char *fmt, ...);
#  define verbose verbose_log
#else
#  define verbose(...) ((void)0)
#endif

/* Identity passthrough — no mmap/asm trampolines in the default build. */
static inline void *
wrapper_create(const char *symbol, void *function)
{
   (void)symbol;
   return function;
}

static inline void
wrapper_set_cpp_demangler(void *function)
{
   (void)function;
}
