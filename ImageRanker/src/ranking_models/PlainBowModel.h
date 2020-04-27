#pragma once

#include "BaseW2vvModel.h"

#include <common.h>
#include <queue>

namespace image_ranker
{
class PlainBowModel : public BaseW2vvModel
{
 public:
    struct Options
  {
    Options()
        : sub_PCA_mean(true)
    {
    }

    bool sub_PCA_mean;
  };

 public:
  static Options parse_options(const std::vector<ModelKeyValOption>& option_key_val_pairs);

  /**
   * Returns sorted vector of ranked images based on provided data for the given query.
   */
  [[nodiscard]] virtual RankingResult rank_frames(
      const Matrix<float>& transformed_data, const KeywordsContainer& keywords,
      const std::vector<CnfFormula>& user_query, size_t result_size,
      const std::vector<ModelKeyValOption>& options = std::vector<ModelKeyValOption>(),
      FrameId target_frame_ID = ERR_VAL<FrameId>()) const override;

  [[nodiscard]] virtual ModelTestResult test_model(
      const Matrix<float>& transformed_data, const KeywordsContainer& keywords,
      const std::vector<UserTestQuery>& test_user_queries,
      const std::vector<ModelKeyValOption>& options = std::vector<ModelKeyValOption>(),
      size_t result_points = NUM_MODEL_TEST_RESULT_POINTS) const override;

  /**
   * Returns sorted vector of ranked images based on provided data for the given query.
   */
  [[nodiscard]] RankingResult rank_frames(const Matrix<float>& transformed_data,
                                                      const KeywordsContainer& keywords,
                                                      const std::vector<CnfFormula>& user_query, size_t result_size,
                                                      const Options& opts, FrameId target_frame_ID) const;


 private:
  /** Returns ranking for the provided frame data, query and options */
  [[nodiscard]] float rank_frame(const Vector<float>& frame_data, const CnfFormula& single_query,
                                 const Options& options) const;
};
}  // namespace image_ranker