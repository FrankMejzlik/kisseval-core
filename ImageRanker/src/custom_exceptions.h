
#ifndef _IR_CUSTOM_EXCEPTIONS_H_
#define _IR_CUSTOM_EXCEPTIONS_H_

/**
 * Custom exceptions.
 */

#include <exception>
#include <stdexcept>

namespace image_ranker
{
class UnableToCreateFileExcept : public std::runtime_error
{
 public:
  UnableToCreateFileExcept(const std::string& msg) : std::runtime_error(msg) {}
};

class NotSuportedModelOptionExcept : public std::runtime_error
{
 public:
  NotSuportedModelOptionExcept(const std::string& msg) : std::runtime_error(msg) {}
};

}  // namespace image_ranker

#endif  // _IR_CUSTOM_EXCEPTIONS_H_
