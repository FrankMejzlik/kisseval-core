#pragma once

#include <queue>

#include "common.h"

#include "BaseClassificationModel.h"

namespace image_ranker
{
class VectorSpaceModel : public BaseClassificationModel
{
 public:
  struct Options
  {
    Options()
        : true_threshold(0.0F), idf_coef(2.0F), dist_fn(eDistFunction::MANHATTAN),
          term_tf(eTermFrequency::NATURAL),
          term_idf(eInvDocumentFrequency::IDF),
          query_tf(eTermFrequency::AUGMENTED),
          query_idf(eInvDocumentFrequency::IDF)
    {
    }

    float true_threshold;
    float idf_coef;
    eDistFunction dist_fn;

    eTermFrequency term_tf;
    eInvDocumentFrequency term_idf;

    eTermFrequency query_tf;
    eInvDocumentFrequency query_idf;
  };

 public:
  static Options parse_options(const std::vector<ModelKeyValOption>& option_key_val_pairs);

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

  [[nodiscard]] virtual ModelTestResult test_model(
      const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
      const std::vector<UserTestQuery>& test_user_queries,
      const std::vector<ModelKeyValOption>& options = std::vector<ModelKeyValOption>(),
      size_t result_points = NUM_MODEL_TEST_RESULT_POINTS) const override;

  /**
   * Returns sorted vector of ranked images based on provided data for the given query.
   *
   * \remark Options are provded already parsed.
   *
   * Query in format: "1&3&4" where numbers are indices to scoring vector.
   */
  [[nodiscard]] RankingResult rank_frames(const BaseVectorTransform& transformed_data,
                                          const KeywordsContainer& keywords, const std::vector<CnfFormula>& user_query,
                                          size_t result_size, const Options& opts, FrameId target_frame_ID) const;

 private:
  [[nodiscard]] Vector<float> create_user_query_vector(const CnfFormula& single_query, size_t vec_dim) const;
  [[nodiscard]] Vector<float> create_user_query_vector(const CnfFormula& single_query, size_t vec_dim, const Vector<float>& idfs, const Options& options) const;
};
}  // namespace image_ranker