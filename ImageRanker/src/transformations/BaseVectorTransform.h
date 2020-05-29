#pragma once

#include <map>
#include <unordered_map>
#include <vector>

#include "common.h"

namespace image_ranker
{
class KeywordsContainer;

using TfidfCacheKey = std::tuple<eTermFrequency, eInvDocumentFrequency, float, float>;

/**
 * Applies provided transformation on the given input data and returns new instance of transformed data.
 */
template <class TransformClass>
[[nodiscard]] Matrix<float> apply(const Matrix<float>& data, const std::string& options = "")
{
  return TransformClass::apply(data, options);
}

enum class HyperAccumType
{
  SUM,
  MAX,
  _COUNT
};

[[nodiscard]] std::pair<Matrix<float>, DataInfo> accumulate_hypernyms(const KeywordsContainer& keywords,
                                                                      Matrix<float>&& data, HyperAccumType type,
                                                                      bool normalize = true, bool accumulate = true);

[[nodiscard]] std::pair<Matrix<float>, DataInfo> calc_stats(const KeywordsContainer& keywords, Matrix<float>&& data,
                                                            bool normalize = true);

/**
 * Class representing transformed data matrix, but this serves mainly as base class and theferoe is "no_transform"
 * transformation.
 */
class [[nodiscard]] BaseVectorTransform
{
 public:
  // -----------------------------------------
  // We need virtual dctor.
  BaseVectorTransform() = default;
  BaseVectorTransform(const BaseVectorTransform& other) = default;
  BaseVectorTransform(BaseVectorTransform&& other) = default;
  BaseVectorTransform& operator=(const BaseVectorTransform& other) = default;
  BaseVectorTransform& operator=(BaseVectorTransform&& other) = default;
  virtual ~BaseVectorTransform() noexcept = default;
  // -----------------------------------------

  BaseVectorTransform(std::pair<Matrix<float>, DataInfo>&& data_sum_mat,
    std::pair<Matrix<float>, DataInfo>&& data_max_mat, std::pair<Matrix<float>, DataInfo>&& data_lin_raw = std::pair<Matrix<float>, DataInfo>{})
      : _data_sum_mat(std::move(data_sum_mat.first)),
        _data_sum_mat_info(std::move(data_sum_mat.second)),
        _data_max_mat(std::move(data_max_mat.first)),
        _data_max_mat_info(std::move(data_max_mat.second)),
        _data_linear_raw(std::move(data_lin_raw.first)),
        _data_linear_raw_info(std::move(data_lin_raw.second))
  {
  }

  [[nodiscard]] virtual const Matrix<float>& data_linear_raw() const { return _data_linear_raw; }
  [[nodiscard]] virtual const DataInfo& data_linear_raw_info() const { return _data_linear_raw_info; }

  [[nodiscard]] virtual const Matrix<float>& data_max() const { return _data_max_mat; }
  [[nodiscard]] virtual const DataInfo& data_max_info() const { return _data_max_mat_info; }

  [[nodiscard]] virtual const Matrix<float>& data_sum() const { return _data_sum_mat; }
  [[nodiscard]] virtual const DataInfo& data_sum_info() const { return _data_sum_mat_info; }

  [[nodiscard]] virtual const Matrix<float>& data_sum_tfidf(eTermFrequency tf_ID, eInvDocumentFrequency idf_ID,
                                                            float true_t, float idf_coef) const;
  [[nodiscard]] virtual const Vector<float>& data_idfs(float true_threshold, float idf_coef) const;

  [[nodiscard]] virtual size_t num_frames() const { return _data_sum_mat.size(); }
  [[nodiscard]] virtual size_t num_dims() const { return _data_sum_mat.at(0).size(); }

 private:
  /** Data with precomputed hypernyms as a sum of all hyponyms
   *  Used for models with SUM inner operation. */
  Matrix<float> _data_sum_mat;
  DataInfo _data_sum_mat_info;

  /** Data with precomputed hypernyms as a max of all hyponyms.
   *  Used for models with MAX inner operation. */
  Matrix<float> _data_max_mat;
  DataInfo _data_max_mat_info;

  mutable std::map<std::pair<float, float>, Vector<float>> _data_idfs;
  mutable std::map<TfidfCacheKey, Matrix<float>> _transformed_data_cache;

  Matrix<float> _data_linear_raw;
  DataInfo _data_linear_raw_info;
};
}  // namespace image_ranker