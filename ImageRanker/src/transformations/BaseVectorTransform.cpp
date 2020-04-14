
#include "BaseVectorTransform.h"

#include "KeywordsContainer.h"

#include <algorithm>

using namespace image_ranker;

namespace image_ranker
{
Matrix<float> accumulate_hypernyms(const KeywordsContainer& keywords, Matrix<float>&& data, HyperAccumType type,
                                   bool normalize)
{
  Matrix<float> result_data;
  result_data.reserve(data.size());
  
  for (auto&& row : data)
  {
    Vector<float> new_row;
    new_row.reserve(row.size());

    float row_sum{ 0.0F };

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
      new_row.emplace_back(new_cell_value);
    }

    if (normalize)
    {
      std::transform(new_row.begin(), new_row.end(), new_row.begin(),
        [row_sum](float x) {
        
          return x / row_sum;
        });
    }

    result_data.emplace_back(std::move(new_row));
  }

  return result_data;
}

}  // namespace image_ranker
