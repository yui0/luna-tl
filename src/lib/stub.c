/*
 * Copyright © 2026 Yuichiro Nakada / Project Lunaria
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * Single build driver for small Android runtime stubs (liblog, libEGL, …).
 * Compile with exactly one -DLUNARIA_STUB_<NAME> to select the module.
 */

#if defined(LUNARIA_STUB_LOG)
# include "inc/log.c"
#elif defined(LUNARIA_STUB_EGL)
# include "inc/egl.c"
#elif defined(LUNARIA_STUB_MEDIANDK)
# include "inc/mediandk.c"
#elif defined(LUNARIA_STUB_MATH)
# include "inc/math.c"
#elif defined(LUNARIA_STUB_ZLIB)
# include "inc/zlib.c"
#elif defined(LUNARIA_STUB_ANDROID)
# include "inc/android.c"
#elif defined(LUNARIA_STUB_OPENSLES)
# include "inc/opensles.c"
#elif defined(LUNARIA_STUB_GLESV3)
# include "inc/glesv3.c"
#else
# error "Define one of LUNARIA_STUB_LOG, LUNARIA_STUB_EGL, LUNARIA_STUB_MEDIANDK, LUNARIA_STUB_MATH, LUNARIA_STUB_ZLIB, LUNARIA_STUB_ANDROID, LUNARIA_STUB_OPENSLES, LUNARIA_STUB_GLESV3"
#endif
