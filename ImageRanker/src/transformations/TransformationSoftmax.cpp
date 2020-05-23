
#include "TransformationSoftmax.h"

#include "KeywordsContainer.h"

using namespace image_ranker;

Matrix<float> TransformationSoftmax::apply(const Matrix<float>& data, [[maybe_unused]] const std::string& options)
{
  return data;
}

Matrix<float> TransformationSoftmax::apply_real(const Matrix<float>& data, [[maybe_unused]] const std::string& options)
{
  Matrix<float> res_mat;
  res_mat.reserve(data.size());

  for (auto&& vec_z : data)
  {
    float exp_sum{ 0.0F };
    Vector<float> vec_res;
    vec_res.reserve(vec_z.size());

    // Exponentiate
    for (auto&& z_i : vec_z)
    {
      auto z_ii{ std::exp(z_i) };

      exp_sum += z_ii;
      vec_res.emplace_back(z_ii);
    }

    // Normalize
    std::transform(vec_res.begin(), vec_res.end(), vec_res.begin(), [exp_sum](const float& x) { return x / exp_sum; });

    res_mat.emplace_back(std::move(vec_res));
  }

  return res_mat;
}

TransformationSoftmax::TransformationSoftmax(const KeywordsContainer& keywords, Matrix<float>& data_mat,
                                             bool accumulate, [[maybe_unused]] const std::string& options)
    : BaseVectorTransform(accumulate_hypernyms(keywords, apply_real(data_mat), HyperAccumType::SUM, true, accumulate),
                          accumulate_hypernyms(keywords, apply_real(data_mat), HyperAccumType::MAX, true, accumulate),
                          calc_stats(keywords, apply_real(data_mat)))
{
}

Matrix<float> TransformationSoftmaxGoogleVision::apply(const Matrix<float>& data,
                                                       [[maybe_unused]] const std::string& options)
{
  return data;
}

Matrix<float> TransformationSoftmaxGoogleVision::apply_real(const Matrix<float>& data,
                                                            [[maybe_unused]] const std::string& options)
{
  Matrix<float> res_mat;
  res_mat.reserve(data.size());

  for (auto&& vec_z : data)
  {
    float exp_sum{ 0.0F };
    Vector<float> vec_res;
    vec_res.reserve(vec_z.size());

    // Exponentiate
    for (auto&& z_i : vec_z)
    {
      auto z_ii{ std::exp(z_i) };

      exp_sum += z_ii;
      vec_res.emplace_back(z_ii);
    }

    // Normalize
    std::transform(vec_res.begin(), vec_res.end(), vec_res.begin(), [exp_sum](const float& x) { return x / exp_sum; });

    res_mat.emplace_back(std::move(vec_res));
  }

  return res_mat;
}

TransformationSoftmaxGoogleVision::TransformationSoftmaxGoogleVision(const KeywordsContainer& keywords,
                                                                     Matrix<float>& data_mat,
                                                                     [[maybe_unused]] const std::string& options)
    : BaseVectorTransform(
          BaseVectorTransform(calc_stats(keywords, apply(data_mat), false), std::pair<Matrix<float>, DataInfo>()))
{
}