
#ifndef _IR_CUSTOM_EXCEPTIONS_H_
#define _IR_CUSTOM_EXCEPTIONS_H_

/**
 * \file custom_exceptions.h
 *
 * Custom exceptions for signaling errors.
 */

#include <exception>
#include <stdexcept>

namespace image_ranker
{
/**
 * Representing unsupported feature for the given reuqest.
 */
class NotSuportedModelOptionExcept : public std::runtime_error
{
 public:
  NotSuportedModelOptionExcept(const std::string& msg) : std::runtime_error(msg) {}
};

}  // namespace image_ranker

#endif  // _IR_CUSTOM_EXCEPTIONS_H_
