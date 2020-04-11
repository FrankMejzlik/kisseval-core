#pragma once

#include <iostream>

//! Will throw exception on LOG_ERROR
#define THROW_ON_ERROR 1

//! Error value for size_t types
#define SIZE_T_ERROR_VALUE SIZE_MAX

#define LOG_CALLS 1

//! Standard logging macro
#define LOG(x) std::cout << x << std::endl

#define LOG_WARN(x) std::cout << x << "(" << __LINE__ << ", " << __FILE__ << ")" << std::endl

#define LOG_NO_ENDL(x) std::cout << x

//! Basic log error macro
#if THROW_ON_ERROR

#define LOG_ERROR(x)                                                                        \
  do                                                                                        \
  {                                                                                         \
    std::cout << "ERROR: " << x << "(" << __LINE__ << ", " << __FILE__ << ")" << std::endl; \
    throw std::runtime_error(std::string(x));                                               \
  } while (0);

#elif

#define LOG_ERROR(x) std::cout << "ERROR: " << x << std::endl;

#endif
