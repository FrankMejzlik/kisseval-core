#pragma once

#include "BaseVectorTransform.h"

#include "common.h"

namespace image_ranker
{
/**
 * Row-wise transformation that ONLY HOLDS Softmax transformed data.
 *
 * \remark Please note that this class assumes that already softmax-ed data will be passed into it's contructor.
 */
class TransformationSoftmax : public BaseVectorTransform
{
 public:
  struct Options
  {
  };

 public:
  [[nodiscard]] static Matrix<float> apply(const Matrix<float>& data, const std::string& options = "")
  {
    LOG_WARN("No transformation has been applied.");
    return data;
  }

  TransformationSoftmax(Matrix<float>&& data_mat, const std::string& options = "")
      : BaseVectorTransform(std::move(data_mat))
  {
  }
};
}  // namespace image_ranker