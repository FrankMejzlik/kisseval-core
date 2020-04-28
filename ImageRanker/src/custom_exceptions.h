/**
 * Custom exception.
 */

#ifndef _IR_CUSTOM_EXCEPTIONS_H_
#define _IR_CUSTOM_EXCEPTIONS_H_

#include <exception>
#include <stdexcept>

class UnableToCreateFileExcept : public std::runtime_error
{
 public:
  UnableToCreateFileExcept(const std::string& msg) : std::runtime_error(msg) {}
};

class NotSuportedModelOption : public std::runtime_error
{
 public:
  NotSuportedModelOption(const std::string& msg) : std::runtime_error(msg) {}
};

#endif  // _IR_CUSTOM_EXCEPTIONS_H_
