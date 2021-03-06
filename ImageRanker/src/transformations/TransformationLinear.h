#pragma once

#include "BaseVectorTransform.h"

namespace image_ranker
{
class KeywordsContainer;

/**
 * Row-wise transformation that moves values to 0 and then scales them down into [0, 1] range linearly.
 */
class [[nodiscard]] TransformationLinear01 : public BaseVectorTransform
{
 public:
  struct Options
  {
    // No options needed here
  };

 public:
  [[nodiscard]] static Matrix<float> apply(const Matrix<float>& data, const std::string& options = "");

  TransformationLinear01(const KeywordsContainer& keywords, const Matrix<float>& data_mat, bool accumulate = true,
                         const std::string& options = "");
};

/**
 * Row-wise transformation that moves values to 0 and then scales them down into [0, 1] range linearly for Google Vision
 * AI.
 */
class [[nodiscard]] TransformationLinear01GoogleVision : public BaseVectorTransform
{
 public:
  struct Options
  {
    // No options needed here
  };

 public:
  [[nodiscard]] static Matrix<float> apply(const Matrix<float>& data, const std::string& options = "");

  TransformationLinear01GoogleVision(const KeywordsContainer& keywords, const Matrix<float>& data_mat,
                                     const std::string& options = "");
};
}  // namespace image_ranker
