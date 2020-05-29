#pragma once

/**********************************
 * Compiler settings
 ***********************************/

/**
 * Macros to manage error warning levels.
 */
// MSVC compiler
#if defined(_MSC_VER)
#  define DISABLE_WARNING_PUSH __pragma(warning(push))
#  define DISABLE_ALL_WARNINGS_PUSH __pragma(warning(push, 0)) __pragma(warning(disable : 5045))
#  define DISABLE_WARNING_POP __pragma(warning(pop))

#  define DISABLE_WARNING(warn_num) __pragma(warning(disable : warn_num))

#  define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER DISABLE_WARNING(4100)
#  define DISABLE_WARNING_UNREFERENCED_FUNCTION DISABLE_WARNING(4505)

// Clang & GCC
#elif defined(__GNUC__) || defined(__clang__)
#  define DO_PRAGMA(X) _Pragma(#  X)
#  define DISABLE_WARNING_PUSH DO_PRAGMA(GCC diagnostic push)
#  define DISABLE_WARNING_PUSH DO_PRAGMA(GCC diagnostic push) DO_PRAGMA(GCC diagnostic ignored "-Wall")
#  define DISABLE_WARNING_POP DO_PRAGMA(GCC diagnostic pop)
#  define DISABLE_WARNING(warn_name) DO_PRAGMA(GCC diagnostic ignored #  warn_name)

#  define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER DISABLE_WARNING(-Wunused - parameter)
#  define DISABLE_WARNING_UNREFERENCED_FUNCTION DISABLE_WARNING(-Wunused - function)

// Anything else
#else
#  define DISABLE_WARNING_PUSH
#  define DISABLE_ALL_WARNINGS_PUSH
#  define DISABLE_WARNING_POP

#  define DISABLE_WARNING

#  define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
#  define DISABLE_WARNING_UNREFERENCED_FUNCTION

#endif
