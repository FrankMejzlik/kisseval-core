
#pragma once

/**
 * Basic loging mechanisms.
 *
 * \todo Is there some more modern way to do this?
 */

#include <iostream>

/** Macro used for throwing exceptions in production mode */
#define THROW_PROD(x) throw std::runtime_error(x);

/** Basic log error macro */
#if THROW_ON_ERROR

#define LOGE(x)                                                                             \
  do                                                                                        \
  {                                                                                         \
    std::cout << "ERROR: " << x << "(" << __FILE__ << ": " << __LINE__ << ")" << std::endl; \
    throw std::runtime_error(std::string(x));                                               \
  } while (0);

#else

do
{
  std::cout << "ERROR: " << x << "(" << __FILE__ << ": " << __LINE__ << ")" << std::endl;
} while (0);

#endif

/** Shorthand for warning logging */
#define LOGW(fmt) LOG(2, fmt)  // NOLINT

/** Shorthand for info logging */
#define LOGI(fmt) LOG(3, fmt)  // NOLINT

/** Shorthand for debug info logging */
#define LOGD(fmt) LOG(4, fmt)  // NOLINT

/** Shorthand for verbose logging */
#define LOGV(fmt) LOG(5, fmt)  // NOLINT

/**
 * Universal logging macro
 */
#if (LOG_LEVEL > 0)
#define LOG(level, x)                                                                         \
  do                                                                                          \
  {                                                                                           \
    if (level <= LOG_LEVEL)                                                                   \
    {                                                                                         \
      switch (level)                                                                          \
      {                                                                                       \
        case 2:                                                                               \
          std::cout << "W: " << x << "(" << __FILE__ << ": " << __LINE__ << ")" << std::endl; \
          break;                                                                              \
                                                                                              \
        case 3:                                                                               \
          std::cout << "I: " << x << "(" << __FILE__ << ": " << __LINE__ << ")" << std::endl; \
          break;                                                                              \
                                                                                              \
        case 4:                                                                               \
          std::cout << "D: " << x << "(" << __FILE__ << ": " << __LINE__ << ")" << std::endl; \
          break;                                                                              \
                                                                                              \
        case 5:                                                                               \
          std::cout << "V: " << x << "(" << __FILE__ << ": " << __LINE__ << ")" << std::endl; \
          break;                                                                              \
      }                                                                                       \
    }                                                                                         \
  } while (0);
#else
#define LOG(level, x)
#endif
