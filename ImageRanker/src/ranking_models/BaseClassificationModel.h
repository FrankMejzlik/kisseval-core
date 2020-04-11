#pragma once

#include <string>
using namespace std::string_literals;
#include <vector>

#include "common.h"

namespace image_ranker
{
class KeywordsContainer;
class BaseVectorTransform;

class BaseClassificationModel
{
 public:
  /**
   * Returns sorted vector of ranked images based on provided data for the given query.
   *
   * Query in format: "1&3&4" where numbers are indices to scoring vector.
   */
  [[nodiscard]] virtual std::vector<FrameId> rank_frames(const BaseVectorTransform& transformed_data,
                                                         const KeywordsContainer& keywords,
                                                         const std::vector<std::string>& user_query,
                                                         const std::string& options = ""s) const = 0;

  /**
   * Returns results of this model after running provided test queries .
   *
   * Query in format: "1&3&4" where numbers are indices to scoring vector.
   */
  [[nodiscard]] virtual std::vector<FrameId> run_test(
      const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
      const std::vector<std::pair<std::vector<std::string>, FrameId>>& test_user_queries,
      const std::string& options = ""s, size_t result_points = NUM_MODEL_TEST_RESULT_POINTS) const = 0;
};
}  // namespace image_ranker