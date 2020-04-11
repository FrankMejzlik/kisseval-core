#pragma once

#include <vector>

#include "common.h"

namespace image_ranker
{
/**
 * Applies provided transformation on the given input data and returns new instance of transformed data.
 */
template <class TransformClass>
[[nodiscard]] Matrix<float> apply(const Matrix<float>& data, const std::string& options = "")
{
  return TransformClass::apply(data, options);
}

/**
 * Class representing transformed data matrix, but this serves mainly as base class and theferoe is "no_transform"
 * transformation.
 */
class BaseVectorTransform
{
 public:
  BaseVectorTransform(const Matrix<float>& data_mat) : _data_mat(data_mat) {}
  BaseVectorTransform(Matrix<float>&& data_mat) : _data_mat(std::move(data_mat)) {}

  [[nodiscard]] virtual const Matrix<float>& data() const { return _data_mat; }

 private:
  Matrix<float> _data_mat;
};
}  // namespace image_ranker