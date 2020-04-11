#pragma once

#include "BaseVectorTransform.h"

class TransformationLinear01 : public BaseVectorTransform
{
  struct Options
  {
  };

  [[nodiscard]] Matrix<float> apply(const Matrix<float>& data, const std::string& options = "") const override
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
};
