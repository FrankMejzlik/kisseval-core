#pragma once

#include <exception>
#include <stdexcept>

class UnableToCreateFileExcept :
  public std::runtime_error
{
public:
  UnableToCreateFileExcept(const std::string& msg):
    std::runtime_error(msg)
  { }
};
