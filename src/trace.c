/*
 * Copyright © 2026 Yuichiro Nakada / Project Lunaria
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "trace.h"

#ifdef VERBOSE_FUNCTIONS

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>

void
verbose_log(const char *fmt, ...)
{
   static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
   pthread_mutex_lock(&mutex);
   va_list ap;
   va_start(ap, fmt);
   static char buf[1024];
   vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);
   fprintf(stderr, "%lu: %s\n", (unsigned long)pthread_self(), buf);
   pthread_mutex_unlock(&mutex);
}

#endif
