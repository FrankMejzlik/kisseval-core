
#include "NoTransform.h"

#include "KeywordsContainer.h"

using namespace image_ranker;

Matrix<float> NoTransform::apply(const Matrix<float>& data, [[maybe_unused]] const std::string& options)
{
  return data;
}

NoTransform::NoTransform(const KeywordsContainer& keywords, Matrix<float>& data_mat,
                                             [[maybe_unused]] const std::string& options)
    : BaseVectorTransform(accumulate_hypernyms(keywords, apply(data_mat), HyperAccumType::SUM),
                          accumulate_hypernyms(keywords, apply(data_mat), HyperAccumType::MAX))
{
}