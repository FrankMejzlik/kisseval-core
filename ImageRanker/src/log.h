
#pragma once

#include <chrono>
#include <iostream>
#include <string>

/** Gets current UNIX timestamp in miliseconds */
#define UNIX_timestamp \
  std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

/**********************************
 * Error handling
 ***********************************/
/** Macro used for throwing exceptions in production mode. */
#if THROW_ON_ERROR
#  define PROD_THROW(x)  // To avoid unreachable code

#else
#  define PROD_THROW(x)                                                                                        \
    do                                                                                                         \
    {                                                                                                          \
      std::string msg{ std::string{ x } + "| Report it with code '" + std::to_string(UNIX_timestamp) + "'." }; \
      throw std::runtime_error(msg);                                                                           \
    } while (false)
#endif

/** Macro used for throwing not supported exception. */
#if THROW_ON_ERROR
#  define PROD_THROW_NOT_SUPP(x) throw NotSuportedModelOptionExcept(x);
#else
#  define PROD_THROW_NOT_SUPP(x)  // To avoid unreachable code
#endif

/**
 * Basic loging macros.
 *
 * \todo Is there some more modern way to do logging?
 */

/** Basic log error macro */
#if THROW_ON_ERROR

#  define LOGE(x)                                                                             \
    do                                                                                        \
    {                                                                                         \
      std::cerr << "<" << UNIX_timestamp << "> "                                              \
                << "ERROR: " << x << "(" << __FILE__ << ": " << __LINE__ << ")" << std::endl; \
      throw std::runtime_error(std::string(x));                                               \
    } while (false)

#else
#  define LOGE(x)                                                                \
    do                                                                           \
    {                                                                            \
      std::cerr << "<" << UNIX_timestamp << "> "                                 \
                << "ERROR:\n"                                                    \
                << x << "(" << __FILE__ << ": " << __LINE__ << ")" << std::endl; \
    } while (false)

#endif

/** Plain log without any meta-data */
#define LOGP(x) std::cout << x << std::endl

/** Shorthand for warning logging */
#define LOGW(fmt) LOG(2, fmt)

/** Shorthand for info logging */
#define LOGI(fmt) LOG(3, fmt)

/** Shorthand for debug info logging */
#define LOGD(fmt) LOG(4, fmt)

/** Shorthand for verbose logging */
#define LOGV(fmt) LOG(5, fmt)

/**
 * Universal logging macro
 */
#if (LOG_LEVEL > 0)
#  define LOG(level, x)                                                                \
    do                                                                                 \
    {                                                                                  \
      if constexpr (level <= LOG_LEVEL)                                                \
      {                                                                                \
        switch (level)                                                                 \
        {                                                                              \
          case 2:                                                                      \
            std::cout << "<" << UNIX_timestamp << "> "                                 \
                      << "W:\n"                                                        \
                      << x << "(" << __FILE__ << ": " << __LINE__ << ")" << std::endl; \
            break;                                                                     \
                                                                                       \
          case 3:                                                                      \
            std::cout << "<" << UNIX_timestamp << "> "                                 \
                      << "I:\n"                                                        \
                      << x << "(" << __FILE__ << ": " << __LINE__ << ")" << std::endl; \
            break;                                                                     \
                                                                                       \
          case 4:                                                                      \
            std::cout << "<" << UNIX_timestamp << "> "                                 \
                      << "D:\n"                                                        \
                      << x << "(" << __FILE__ << ": " << __LINE__ << ")" << std::endl; \
            break;                                                                     \
                                                                                       \
          case 5:                                                                      \
            std::cout << "<" << UNIX_timestamp << "> "                                 \
                      << "V:\n"                                                        \
                      << x << "(" << __FILE__ << ": " << __LINE__ << ")" << std::endl; \
            break;                                                                     \
        }                                                                              \
      }                                                                                \
    } while (0);
#else
#  define LOG(level, x)
#endif
