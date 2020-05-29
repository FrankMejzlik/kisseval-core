#pragma once

#include "BaseVectorTransform.h"

#include "common.h"

namespace image_ranker
{
class KeywordsContainer;

/**
 * Row-wise transformation representing SOFTMAX transformation.
 */
class [[nodiscard]] TransformationSoftmax : public BaseVectorTransform
{
 public:
  struct Options
  {
    // None needed
  };

 public:
  [[nodiscard]] static Matrix<float> apply(const Matrix<float>& data, [[maybe_unused]] const std::string& options = "");
  [[nodiscard]] static Matrix<float> apply_real(const Matrix<float>& data, [[maybe_unused]] const std::string& options = "");

  TransformationSoftmax(const KeywordsContainer& keywords, Matrix<float>& data_mat, bool accumulate = true,
                        [[maybe_unused]] const std::string& options = "");
};

/**
 * Row-wise transformation representing SOFTMAX transformation for Google Vision AI.
 */
class [[nodiscard]] TransformationSoftmaxGoogleVision : public BaseVectorTransform
{
 public:
  struct Options
  {
    // None needed
  };

 public:
  [[nodiscard]] static Matrix<float> apply(const Matrix<float>& data, [[maybe_unused]] const std::string& options = "");
  [[nodiscard]] static Matrix<float> apply_real(const Matrix<float>& data, [[maybe_unused]] const std::string& options = "");

  TransformationSoftmaxGoogleVision(const KeywordsContainer& keywords, Matrix<float>& data_mat,
                        [[maybe_unused]] const std::string& options = "");
};

}  // namespace image_ranker