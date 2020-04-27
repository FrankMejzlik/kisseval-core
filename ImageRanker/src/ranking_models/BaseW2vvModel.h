#pragma once

#include <string>
using namespace std::string_literals;
#include <vector>

#include "common.h"

namespace image_ranker
{

template <typename ModelType>
auto parse_model_options(const std::vector<ModelKeyValOption>& option_key_val_pairs) -> decltype(ModelType::Options)
{
  return ModelType::parse_options(option_key_val_pairs);
}

class KeywordsContainer;
class BaseVectorTransform;

class BaseModel {

};

class BaseW2vvModel : public BaseModel
{
 public:
  /**
   * Returns sorted vector of ranked images based on provided data for the given query.
   *
   * Query in format: "1&3&4" where numbers are indices to scoring vector.
   */
  [[nodiscard]] virtual RankingResult rank_frames(
      const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
      const std::vector<CnfFormula>& user_query, size_t result_size,
      const std::vector<ModelKeyValOption>& options = std::vector<ModelKeyValOption>(),
      FrameId target_frame_ID = ERR_VAL<FrameId>()) const = 0;


  /**
   * Returns results of this model after running provided test queries .
   *
   * Query in format: "1&3&4" where numbers are indices to scoring vector.
   */
  [[nodiscard]] virtual ModelTestResult test_model(
      const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
      const std::vector<UserTestQuery>& test_user_queries,
      const std::vector<ModelKeyValOption>& options = std::vector<ModelKeyValOption>(),
      size_t result_points = NUM_MODEL_TEST_RESULT_POINTS) const = 0;
};
}  // namespace image_ranker