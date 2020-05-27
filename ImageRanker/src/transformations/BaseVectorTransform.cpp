
#include "BaseVectorTransform.h"

#include "KeywordsContainer.h"

#include <algorithm>
#include <cmath>
#include <queue>

using namespace image_ranker;

namespace image_ranker
{
std::pair<Matrix<float>, DataInfo> accumulate_hypernyms(const KeywordsContainer& keywords, Matrix<float>&& data,
                                                        HyperAccumType type, bool normalize, bool accumulate)
{
  Matrix<float> result_data;
  result_data.reserve(data.size());
  DataInfo di;
  std::vector<std::vector<KeywordId>> top_classes;

  for (auto&& row : data)
  {
    Vector<float> new_row;
    new_row.reserve(row.size());

    std::vector<KeywordId> row_top_classes;
    row_top_classes.reserve(NUM_TOP_KWS_LOADED);

    float row_sum{ 0.0F };
    float row_max{ -std::numeric_limits<float>::max() };
    float row_min{ std::numeric_limits<float>::max() };

    using KwPair = std::pair<float, KeywordId>;
    // Comparator for the priority queue
    auto frame_pair_cmptor = [](const KwPair& left, const KwPair& right) { return left.first < right.first; };

    // Create inner container for the queue
    std::vector<KwPair> queue_cont;
    queue_cont.reserve(row.size());

    // Construct prioprity queue
    std::priority_queue<KwPair, std::vector<KwPair>, decltype(frame_pair_cmptor)> max_prio_queue(frame_pair_cmptor, std::move(queue_cont));

    // Iterate over all bins in this vector
    for (auto&& [it, i]{ std::tuple(row.begin(), size_t{ 0 }) }; it != row.end(); ++it, ++i)
    {
      auto& bin{ *it };
      auto pKw{ keywords.GetKeywordConstPtrByVectorIndex(i) };

      float new_cell_value{ 0.0F };

      // If hypernym accumulation is required
      if (accumulate)
      {
        // Iterate over all indices this keyword interjoins
        for (auto&& kwIndex : pKw->m_hyponymBinIndices)
        {
          if (type == HyperAccumType::MAX)
          {
            new_cell_value = std::max(new_cell_value, row[kwIndex]);
          }
          else
          {
            new_cell_value += row[kwIndex];
          }
        }
      }
      else
      {
        new_cell_value = bin;

        // Deaccumulate them 
        for (auto&& kwIndex : pKw->m_hyponymBinIndices)
        {
          if (type == HyperAccumType::MAX)
          {
            new_cell_value = std::max(new_cell_value, row[kwIndex]);
          }
          else
          {
            new_cell_value += row[kwIndex];
          }
        }
      }

      row_sum += new_cell_value;
      row_max = std::max(row_max, new_cell_value);
      row_min = std::min(row_min, new_cell_value);

      new_row.emplace_back(new_cell_value);
      max_prio_queue.emplace(std::pair(new_cell_value, i));
    }

    if (normalize)
    {
      std::transform(new_row.begin(), new_row.end(), new_row.begin(), [row_sum](float x) { return x / row_sum; });

      row_max /= row_sum;
      row_min /= row_sum;
    }

    // Ge top concepts
    for (size_t ii{ 0_z }; ii < NUM_TOP_KWS_LOADED; ++ii)
    {
      auto&& [score, idx]{ max_prio_queue.top() };

      if (score < ZERO_WEIGHT)
      {
        break;
      }

      auto ID{ keywords.GetKeywordConstPtrByVectorIndex(idx)->ID };

      row_top_classes.emplace_back(ID);
      max_prio_queue.pop();
    }

    di.maxes.emplace_back(row_max);
    di.mins.emplace_back(row_min);
    di.top_classes.emplace_back(std::move(row_top_classes));

    result_data.emplace_back(std::move(new_row));
  }

  return std::pair(std::move(result_data), std::move(di));
}

std::pair<Matrix<float>, DataInfo> calc_stats(const KeywordsContainer& keywords, Matrix<float>&& data, bool normalize)
{
  Matrix<float> result_data;
  result_data.reserve(data.size());
  DataInfo di;
  size_t iii{ 0 };
  for (auto&& row : data)
  {
    Vector<float> new_row;
    new_row.reserve(row.size());

    if (row.size() == 0)
    {
      std::cout << row.size() << std::endl;
      std::cout << iii << std::endl;
    }

    std::vector<KeywordId> row_top_classes;
    row_top_classes.reserve(NUM_TOP_KWS_LOADED);

    float row_sum{ 0.0F };
    float row_max{ -std::numeric_limits<float>::max() };
    float row_min{ std::numeric_limits<float>::max() };

    using KwPair = std::pair<float, KeywordId>;
    // Comparator for the priority queue
    auto frame_pair_cmptor = [](const KwPair& left, const KwPair& right) { return left.first < right.first; };

    // Create inner container for the queue
    std::vector<KwPair> queue_cont;
    queue_cont.reserve(row.size());

    // Construct prioprity queue
    std::priority_queue<KwPair, std::vector<KwPair>, decltype(frame_pair_cmptor)> max_prio_queue(frame_pair_cmptor,
                                                                                                 std::move(queue_cont));

    // Iterate over all bins in this vector
    for (auto&& [it, i]{ std::tuple(row.begin(), size_t{ 0 }) }; it != row.end(); ++it, ++i)
    {
      auto& bin{ *it };

      row_sum += bin;
      row_max = std::max(row_max, bin);
      row_min = std::min(row_min, bin);

      new_row.emplace_back(bin);
      max_prio_queue.emplace(std::pair(bin, i));
    }

    if (normalize)
    {
      std::transform(new_row.begin(), new_row.end(), new_row.begin(), [row_sum](float x) { return x / row_sum; });

      row_max /= row_sum;
      row_min /= row_sum;
    }

    if (max_prio_queue.empty())
    {
      std::cout << "xx" << std::endl;
    }

    // Ge top concepts
    for (size_t ii{ 0_z }; ii < NUM_TOP_KWS_LOADED; ++ii)
    {
      auto&& [score, idx]{ max_prio_queue.top() };

      auto ID{ keywords.GetKeywordConstPtrByVectorIndex(idx)->ID };

      row_top_classes.emplace_back(ID);
      max_prio_queue.pop();
    }

    di.maxes.emplace_back(row_max);
    di.mins.emplace_back(row_min);
    di.top_classes.emplace_back(std::move(row_top_classes));

    result_data.emplace_back(std::move(new_row));
    ++iii;
  }

  return std::pair(std::move(result_data), std::move(di));
}

const Vector<float>& BaseVectorTransform::data_idfs(float true_threshold, float idf_coef) const
{
  // Look into the cache first
  std::pair<float, float> key = std::pair(true_threshold, idf_coef);

  const auto it{ _data_idfs.find(key) };
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
    idf_val_fn = [idf_coef](float score) { return score; };
  }
  else
  {
    idf_val_fn = [true_threshold](float score) { return (score < true_threshold ? 0.0F : 1.0F); };
  }

  Vector<float> idfs(num_dims(), 0.0F);

  float max{ 0.0F };

  // Iterate all frame feature vectors
  for (auto&& fea_vec : _data_sum_mat)
  {
    // Iterate over all concept indices
    for (size_t i{ 0_z }; i < num_dims(); ++i)
    {
      auto v{ idf_val_fn(fea_vec[i]) };
      idfs[i] += v;
      max = std::max(max, idfs[i]);
    }
  }

  // Compute final value
  if (true_threshold == 0.0F)
  {
    // Just normalize with number of frames
    for (auto&& val : idfs)
    {
      auto v = (1 - (val / max));
      val = 2 * pow(v, idf_coef);
      // xoxo
      // val = log(_data_sum_mat.size() / val);
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
  _data_idfs.emplace(key, std::move(idfs));

  // Return it
  return data_idfs(true_threshold, idf_coef);
}

const Matrix<float>& BaseVectorTransform::data_sum_tfidf(eTermFrequency tf_ID, eInvDocumentFrequency idf_ID,
                                                         float true_t, float idf_coef) const
{
  // Look into the cache first
  TfidfCacheKey key{ TfidfCacheKey(tf_ID, idf_ID, true_t, idf_coef) };
  const auto it{ _transformed_data_cache.find(key) };
  if (it != _transformed_data_cache.end())
  {
    return it->second;
  }

  auto term_tf_fn{ pick_tf_scheme_fn(tf_ID) };

  float term_t = idf_ID == eInvDocumentFrequency::IDF ? true_t : 0.0F;

  // Get IDFs
  const Vector<float>& data_mat_maximums = data_sum_info().maxes;

  size_t num_classes{ _data_sum_mat.front().size() };

  const Vector<float>& term_idfs_vector{ term_t != 0.0F ? data_idfs(term_t, idf_coef)
                                                        : Vector<float>(num_classes, 1.0F) };

  Matrix<float> new_data_mat;
  new_data_mat.reserve(_data_sum_mat.size());

  FrameId frame_ID = 0;
  for (auto&& fea_vec : _data_sum_mat)
  {
    Vector<float> frame_vector;
    frame_vector.reserve(fea_vec.size());

    auto tf_max{data_mat_maximums[frame_ID]};
    
    size_t i{ 0_z };
    for (auto&& val : fea_vec)
    {
      float tf{ term_tf_fn(val, tf_max) };  
      float idf{ 1.0F };
      if (idf_ID == eInvDocumentFrequency::IDF)
      {
        idf = term_idfs_vector[i];
      }

      frame_vector.emplace_back(tf * idf);
      ++i;
    }

    // frame_vector = normalize(frame_vector);
    new_data_mat.emplace_back(std::move(frame_vector));

    ++frame_ID;
  }

  // Insert into the cache
  _transformed_data_cache.emplace(key, std::move(new_data_mat));

  // Return it
  return data_sum_tfidf(tf_ID, idf_ID, true_t, idf_coef);
}

}  // namespace image_ranker
