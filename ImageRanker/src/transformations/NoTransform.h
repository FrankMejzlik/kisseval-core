#pragma once

#include "BaseVectorTransform.h"

#include "common.h"

namespace image_ranker
{
class KeywordsContainer;

/**
 * Transformation that represents no transformation whatsoever.
 */
class [[nodiscard]] NoTransform : public BaseVectorTransform
{
 public:
  struct Options
  {
    // No options needed here
  };

 public:
  NoTransform(const KeywordsContainer& keywords, Matrix<float>& data_mat,
              [[maybe_unused]] const std::string& options = "");

  /** Applies this transformation on the data with the provided options. */
  [[nodiscard]] static Matrix<float> apply(const Matrix<float>& data, [[maybe_unused]] const std::string& options = "");
};

/**
 * Transformation that represents no transformation whatsoever for GoogleVision AI data.
 */
class [[nodiscard]] NoTransformGoogleVision : public BaseVectorTransform
{
 public:
  struct Options
  {
    // No options needed here
  };

 public:
  NoTransformGoogleVision(const KeywordsContainer& keywords, Matrix<float>& data_mat,
                          [[maybe_unused]] const std::string& options = "");

  /** Applies this transformation on the data with the provided options. */
  [[nodiscard]] static Matrix<float> apply(const Matrix<float>& data, [[maybe_unused]] const std::string& options = "");
};
}  // namespace image_ranker