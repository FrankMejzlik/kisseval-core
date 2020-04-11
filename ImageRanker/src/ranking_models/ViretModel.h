#pragma once

#include "BaseClassificationModel.h"

#include <common.h>
#include <queue>

namespace image_ranker
{
class ViretModel : public BaseClassificationModel
{
 public:
  /*!
   * List of possible settings what how to calculate rank
   */
  enum class eScoringOperations
  {
    cMultSum = 0,
    cMultMax = 1,
    cSumSum = 2,
    cSumMax = 3,
    cMaxMax = 4
  };

  struct Options
  {
    /** Values less then this threshold will be considered zero */
    float ignore_below_threshold;

    /** How ranking for each frame will be calculated */
    eScoringOperations scoring_operations;

    /** How aggregated siccesor ranking will be combined with ranking of the main frame */
    eMainTempRankingAggregation main_temp_aggregation;

    /** How rankings of successor frames will be aggregated. */
    eSuccesorAggregation succ_aggregation;
  };

 public:
  static Options ParseOptionsString(const std::vector<ModelKeyValOption>& options_string);

  /**
   * Returns sorted vector of ranked images based on provided data for the given query.
   *
   * Query in format: "1&3&4" where numbers are indices to scoring vector.
   */
  [[nodiscard]] virtual RankingResult rank_frames(
      const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
      const std::vector<CnfFormula>& user_query, size_t result_size,
      const std::vector<ModelKeyValOption>& options = std::vector<ModelKeyValOption>(),
      FrameId target_frame_ID = ERR_VAL<FrameId>()) const override;

  /**
   * Returns results of this model after running provided test queries .
   *
   * Query in format: "1&3&4" where numbers are indices to scoring vector.
   */
  [[nodiscard]] virtual std::vector<FrameId> run_test(
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