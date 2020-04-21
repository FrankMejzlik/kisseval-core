#pragma once

#include "BaseClassificationModel.h"

#include <common.h>
#include <queue>

namespace image_ranker
{
class BooleanModel : public BaseClassificationModel
{
 public:
    struct Options
  {
    Options()
        : true_threshold(0.001F)
    {
    }

    /** Values greater or equal will be consideted `true`. */
    float true_threshold;
  };

 public:
  static Options ParseOptionsString(const std::vector<ModelKeyValOption>& option_key_val_pairs);

  /**
   * Returns sorted vector of ranked images based on provided data for the given query.
   *
   * \remark Options are provided unparsed as vector of key->value string pairs.
   *
   * Query in format: "1&3&4" where numbers are indices to scoring vector.
   */
  [[nodiscard]] virtual RankingResult rank_frames(
      const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
      const std::vector<CnfFormula>& user_query, size_t result_size,
      const std::vector<ModelKeyValOption>& options = std::vector<ModelKeyValOption>(),
      FrameId target_frame_ID = ERR_VAL<FrameId>()) const override;

  /**
   * Returns sorted vector of ranked images based on provided data for the given query.
   *
   * \remark Options are provded already parsed.
   *
   * Query in format: "1&3&4" where numbers are indices to scoring vector.
   */
  [[nodiscard]] RankingResult rank_frames(const BaseVectorTransform& transformed_data,
                                                      const KeywordsContainer& keywords,
                                                      const std::vector<CnfFormula>& user_query, size_t result_size,
                                                      const Options& opts, FrameId target_frame_ID) const;

  /**
   * Returns results of this model after running provided test queries .
   *
   * Query in format: "1&3&4" where numbers are indices to scoring vector.
   */
  [[nodiscard]] virtual ModelTestResult run_test(
      const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
      const std::vector<UserTestQuery>& test_user_queries,
      const std::vector<ModelKeyValOption>& options = std::vector<ModelKeyValOption>(),
      size_t result_points = NUM_MODEL_TEST_RESULT_POINTS) const override;

 private:
  /** Returns ranking for the provided frame data, query and options */
  [[nodiscard]] float rank_frame(const Vector<float>& frame_data, const CnfFormula& single_query,
                                 const Options& options) const;
};
}  // namespace image_ranker