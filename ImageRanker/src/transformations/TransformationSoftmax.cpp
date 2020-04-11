
#include "TransformationSoftmax.h"

#include "KeywordsContainer.h"

using namespace image_ranker;

Matrix<float> TransformationSoftmax::apply(const KeywordsContainer& keywords, const Matrix<float>& data,
                                           [[maybe_unused]] const std::string& options)
{
  // Apply hypernym accumulation
  LOG_WARN("Hypernym accumulation not yet applied!");

  return data;
}

TransformationSoftmax::TransformationSoftmax(const KeywordsContainer& keywords, Matrix<float>& data_mat,
                                             [[maybe_unused]] const std::string& options)
    : BaseVectorTransform(apply(keywords, data_mat))
{
}