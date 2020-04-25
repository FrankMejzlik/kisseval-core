#pragma once

/* MSVC does not define SSE* flags therefore we just try to turn in on,
 * if your CPU does not support this, you'll get compilation error. */
#if (defined(__SSE4_2__) || defined(_MSC_VER))
#define USE_INTRINS
#else
#pragma message("Fix your CXXFLAGS or get a better CPU!")
#endif

#ifdef USE_INTRINS
#if defined(_MSC_VER)
#include <wmmintrin.h>
#elif
#include <xmmintrin.h>
#endif
#endif
