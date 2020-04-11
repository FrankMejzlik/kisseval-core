#pragma once

#include <vector>

#include "common.h"

namespace image_ranker
{
class KeywordsContainer;

/**
 * Applies provided transformation on the given input data and returns new instance of transformed data.
 */
template <class TransformClass>
[[nodiscard]] Matrix<float> apply(const Matrix<float>& data, const std::string& options = "")
{
  return TransformClass::apply(data, options);
}

enum class HyperAccumType
{
  SUM,
  MAX,
  _COUNT
};

[[nodiscard]] Matrix<float> accumulate_hypernyms(const KeywordsContainer& keywords, Matrix<float>&& data,
                                                 HyperAccumType type, bool normalize = true);

/**
 * Class representing transformed data matrix, but this serves mainly as base class and theferoe is "no_transform"
 * transformation.
 */
class BaseVectorTransform
{
 public:
  BaseVectorTransform(const Matrix<float>& data_sum_mat, const Matrix<float>& data_max_mat)
      : _data_sum_mat(data_sum_mat), _data_max_mat(data_max_mat)
  {
  }
  BaseVectorTransform(Matrix<float>&& data_sum_mat, Matrix<float>&& data_max_mat)
      : _data_sum_mat(std::move(data_sum_mat)), _data_max_mat(std::move(data_max_mat))
  {
  }

  [[nodiscard]] virtual const Matrix<float>& data_max() const { return _data_max_mat; }
  [[nodiscard]] virtual const Matrix<float>& data_sum() const { return _data_sum_mat; }
  
  [[nodiscard]] virtual size_t num_frames() const { return _data_sum_mat.size(); }
  [[nodiscard]] virtual size_t num_dims() const { return _data_sum_mat.at(0).size(); }

 private:
  /** Data with precomputed hypernyms as a sum of all hyponyms
   *  Used for models with SUM inner operation. */
  Matrix<float> _data_sum_mat;

  /** Data with precomputed hypernyms as a max of all hyponyms.
   *  Used for models with MAX inner operation. */
  Matrix<float> _data_max_mat;
};
}  // namespace image_ranker