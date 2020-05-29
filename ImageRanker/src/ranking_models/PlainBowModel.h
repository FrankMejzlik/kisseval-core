#pragma once

#include "BaseW2vvModel.h"

#include <common.h>
#include <queue>

namespace image_ranker
{
class [[nodiscard]] PlainBowModel : public BaseW2vvModel
{
 public:
  struct Options
  {
    Options() : sub_PCA_mean(true), do_PCA(false), dist_fn(eDistFunction::COSINE_NONORM) {}

    bool sub_PCA_mean;
    bool do_PCA;
    eDistFunction dist_fn;
  };

 public:
  static Options parse_options(const std::vector<ModelKeyValOption>& option_key_val_pairs);

  /**
   * Returns sorted vector of ranked images based on provided data for the given query.
   */
  [[nodiscard]] virtual RankingResult rank_frames(
      const Matrix<float>& transformed_data, const Matrix<float>& kw_features, const Vector<float>& kw_bias_vec,
      const Matrix<float>& kw_PCA_mat, const Vector<float>& kw_PCA_mean_vec, const KeywordsContainer& keywords,
      const std::vector<CnfFormula>& user_query, size_t result_size,
      const std::vector<ModelKeyValOption>& options = std::vector<ModelKeyValOption>(),
      FrameId target_frame_ID = ERR_VAL<FrameId>()) const override;

  [[nodiscard]] virtual ModelTestResult test_model(
      const Matrix<float>& transformed_data, const Matrix<float>& kw_features, const Vector<float>& kw_bias_vec,
      const Matrix<float>& kw_PCA_mat, const Vector<float>& kw_PCA_mean_vec, const KeywordsContainer& keywords,
      const std::vector<UserTestQuery>& test_user_queries,
      const std::vector<ModelKeyValOption>& options = std::vector<ModelKeyValOption>(),
      size_t result_points = NUM_MODEL_TEST_RESULT_POINTS) const override;

  /**
   * Returns sorted vector of ranked images based on provided data for the given query.
   */
  [[nodiscard]] RankingResult rank_frames(const Matrix<float>& transformed_data, const Matrix<float>& kw_features,
                                          const Vector<float>& kw_bias_vec, const Matrix<float>& kw_PCA_mat,
                                          const Vector<float>& kw_PCA_mean_vec, const KeywordsContainer& keywords,
                                          const std::vector<CnfFormula>& user_query, size_t result_size,
                                          const Options& opts, FrameId target_frame_ID) const;

 private:
  [[nodiscard]] Vector<float> embedd_native_user_query(const Matrix<float>& kw_features,
                                                       const Vector<float>& kw_bias_vec,
                                                       const Matrix<float>& kw_PCA_mat,
                                                       const Vector<float>& kw_PCA_mean_vec, const CnfFormula& query,
                                                       const Options& opts) const;
};
}  // namespace image_ranker