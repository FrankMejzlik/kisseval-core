#pragma once

#include <vector>

#include "common.h"
#include "utility.h"

class BaseVectorTransform
{
  virtual Matrix<float> apply(const Matrix<float>& data, const std::string& options = "") const = 0;
};
