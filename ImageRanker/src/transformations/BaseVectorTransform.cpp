
#include "BaseVectorTransform.h"

#include "KeywordsContainer.h"

#include <algorithm>
#include <cmath>

using namespace image_ranker;

namespace image_ranker
{
std::pair<Matrix<float>, DataInfo> accumulate_hypernyms(const KeywordsContainer& keywords, Matrix<float>&& data,
                                                        HyperAccumType type, bool normalize)
{
  Matrix<float> result_data;
  result_data.reserve(data.size());
  DataInfo di;

  for (auto&& row : data)
  {
    Vector<float> new_row;
    new_row.reserve(row.size());

    float row_sum{0.0F};
    float row_max{-99999999.0F};
    float row_min{std::numeric_limits<float>::max()};

    // Iterate over all bins in this vector
    for (auto&& [it, i]{std::tuple(row.begin(), size_t{0})}; it != row.end(); ++it, ++i)
    {
      auto& bin{*it};
      auto pKw{keywords.GetKeywordConstPtrByVectorIndex(i)};

      float new_cell_value{0.0F};

      // Iterate over all indices this keyword interjoins
      for (auto&& kwIndex : pKw->m_hyponymBinIndices)
      {
        new_cell_value += row[kwIndex];
      }

      row_sum += new_cell_value;
      row_max = std::max(row_max, new_cell_value);
      row_min = std::min(row_min, new_cell_value);

      new_row.emplace_back(new_cell_value);
    }

    if (normalize)
    {
      std::transform(new_row.begin(), new_row.end(), new_row.begin(), [row_sum](float x) { return x / row_sum; });

      row_max /= row_sum;
      row_min /= row_sum;
    }

    di.maxes.emplace_back(row_max);
    di.mins.emplace_back(row_min);

    result_data.emplace_back(std::move(new_row));
  }

  return std::pair(std::move(result_data), std::move(di));
}

const Vector<float>& BaseVectorTransform::data_idfs(float true_threshold) const
{
  // Look into the cache first
  const auto it{_data_idfs.find(true_threshold)};
  if (it != _data_idfs.end())
  {
    return it->second;
  }

  /*
   * Compute IDF values for all concepts
   */
  std::function<float(float)> idf_val_fn;

  // Choose correct idf evaluation function based on parameter
  if (true_threshold == 0.0F)
  {
    idf_val_fn = [](float score) { return score; };
  }
  else
  {
    idf_val_fn = [true_threshold](float score) { return (score < true_threshold ? 0.0F : 1.0F); };
  }

  Vector<float> idfs(num_dims(), 0.0F);

  // Iterate all frame feature vectors
  for (auto&& fea_vec : _data_sum_mat)
  {
    // Iterate over all concept indices
    for (size_t i{0_z}; i < num_dims(); ++i)
    {
      idfs[i] += idf_val_fn(fea_vec[i]);
    }
  }

  // Compute final value
  if (true_threshold == 0.0F)
  {
    // Just normalize with number of frames
    for (auto&& val : idfs)
    {
      val /= _data_sum_mat.size();
    }
  }
  else
  {
    // IDF formula
    for (auto&& val : idfs)
    {
      val = log(_data_sum_mat.size() / val);
    }
  }

  // Plase inside the cache
  _data_idfs.emplace(true_threshold, std::move(idfs));

  // Return it
  return data_idfs(true_threshold);
}

const Matrix<float>& BaseVectorTransform::data_sum_tfidf(eTermFrequency tf_ID, eInvDocumentFrequency idf_ID,
                                                         float true_t) const
{
  // Look into the cache first
  TfidfCacheKey key{TfidfCacheKey(tf_ID, idf_ID)};
  const auto it{_transformed_data_cache.find(key)};
  if (it != _transformed_data_cache.end())
  {
    return it->second;
  }

  auto term_tf_fn{pick_tf_scheme_fn(tf_ID)};

  float term_t = idf_ID == eInvDocumentFrequency::IDF ? true_t : 0.0F;

  // Get IDFs
  const Vector<float>& term_idfs_vector{data_idfs(term_t)};
  const Vector<float>& data_mat_maximums = data_sum_info().maxes;

  Matrix<float> new_data_mat;
  new_data_mat.reserve(_data_sum_mat.size());

  for (auto&& fea_vec : _data_sum_mat)
  {
    Vector<float> frame_vector;
    frame_vector.reserve(fea_vec.size());

    size_t i{0_z};
    for (auto&& val : fea_vec)
    {
      //float tf{term_tf_fn(val, data_mat_maximums[i])};
      float tf{val};
      float idf{term_idfs_vector[i]};

      frame_vector.emplace_back(tf * idf);
      ++i;
    }

    //frame_vector = normalize(frame_vector);
    new_data_mat.emplace_back(std::move(frame_vector));
  }

  // Insert into the cache
  _transformed_data_cache.emplace(key, std::move(new_data_mat));

  // Return it
  return data_sum_tfidf(tf_ID, idf_ID, true_t);
}

}  // namespace image_ranker
