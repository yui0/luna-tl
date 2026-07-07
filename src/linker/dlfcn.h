/*
 * Copyright © 2026 Yuichiro Nakada / Project Lunaria
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

void dl_parse_library_path(const char *path, char *delim);
void *bionic_dlopen(const char *filename, int flag);
const char *bionic_dlerror(void);
void *bionic_dlsym(void *handle, const char *symbol);
int bionic_dlclose(void *handle);
