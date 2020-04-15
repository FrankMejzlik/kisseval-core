
#include "TransformationLinear.h"

#include "KeywordsContainer.h"

using namespace image_ranker;

Matrix<float> TransformationLinear01::apply(const Matrix<float>& data, const std::string& options)
{
  Matrix<float> result_mat;

  for (auto&& row : data)
  {
    float sum{0.0F};
    float min{std::numeric_limits<float>::max()};
    float max{-std::numeric_limits<float>::max()};

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

TransformationLinear01::TransformationLinear01(const KeywordsContainer& keywords, const Matrix<float>& data_mat,
                                               const std::string& options)
    : BaseVectorTransform(accumulate_hypernyms(keywords, apply(data_mat), HyperAccumType::SUM),
                          accumulate_hypernyms(keywords, apply(data_mat), HyperAccumType::MAX))
{
}