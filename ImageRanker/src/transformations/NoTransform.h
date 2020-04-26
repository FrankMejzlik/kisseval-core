#pragma once

#include "BaseVectorTransform.h"

#include "common.h"

namespace image_ranker
{
class KeywordsContainer;

/**
 * Row-wise transformation that ONLY HOLDS Softmax transformed data.
 *
 * \remark Please note that this class assumes that already softmax-ed data will be passed into it's contructor.
 */
class NoTransform : public BaseVectorTransform
{
 public:
  struct Options
  {
  };

 public:
  [[nodiscard]] static Matrix<float> apply(const Matrix<float>& data, [[maybe_unused]] const std::string& options = "");

  NoTransform(const KeywordsContainer& keywords, Matrix<float>& data_mat,
                        [[maybe_unused]] const std::string& options = "");
};
}  // namespace image_ranker