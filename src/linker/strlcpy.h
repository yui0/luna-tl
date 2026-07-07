/*
 * Copyright © 2026 Yuichiro Nakada / Project Lunaria
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef STRLCPY_H
#define STRLCPY_H

#include <sys/types.h>
#include <string.h>

size_t
apkenv_strlcpy(char *dst, const char *src, size_t siz);

#endif
