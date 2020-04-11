#pragma once

#include "BaseVectorTransform.h"

namespace image_ranker
{
/**
 * Row-wise transformation that moves values to 0 and then scales them down into [0, 1] range linearly.
 */
class TransformationLinear01 : public BaseVectorTransform
{
 public:
  struct Options
  {
  };

 public:
  [[nodiscard]] static Matrix<float> apply(const Matrix<float>& data, const std::string& options = "")
  {
    Matrix<float> result_mat;

    for (auto&& row : data)
    {
      float sum{0.0F};
      float min{std::numeric_limits<float>::max()};
      float max{std::numeric_limits<float>::min()};

      Vector<float> new_row;

      // Compute statistics
      for (auto&& cell : row)
      {
        sum += cell;
        min = std::min(min, cell);
        max = std::max(max, cell);
      }

      // Transform the row
      for (auto&& cell : row)
      {
        new_row.emplace_back((cell - min) / max);
      }

      result_mat.emplace_back(std::move(new_row));
    }

    return result_mat;
  }

  TransformationLinear01(const Matrix<float>& data_mat, const std::string& options = "")
      : BaseVectorTransform(apply(data_mat))
  {
  }
};
}  // namespace image_ranker
