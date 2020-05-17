#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "KeywordsContainer.h"

#include "data_packs/BaseDataPack.h"

#include "BaseClassificationModel.h"
#include "BaseVectorTransform.h"
#include "SimUser.h"

namespace image_ranker
{
class GoogleVisionDataPack : public BaseDataPack
{
 public:
  GoogleVisionDataPack(const BaseImageset* p_is, const StringId& ID, const StringId& target_imageset_ID, const std::string& model_options,
                const std::string& description, const DataPackStats& stats, const GoogleDataPackRef::VocabData& vocab_data_refs,
                std::vector<std::vector<float>>&& presoft);

  [[nodiscard]] virtual RankingResult rank_frames(const std::vector<CnfFormula>& user_queries,
                                                  const std::string& model_options, size_t result_size,
                                                  FrameId target_image_ID = ERR_VAL<FrameId>()) const override;

  [[nodiscard]] virtual ModelTestResult test_model(const std::vector<UserTestQuery>& test_queries,
                                                   const std::string& model_options, size_t num_points) const override;

  [[nodiscard]] void cache_up_example_images(const std::vector<const Keyword*>& kws,
                                            const std::string& model_commands) const;

  [[nodiscard]] virtual std::vector<const Keyword*> get_frame_top_classes(
      FrameId frame_ID, std::vector<ModelKeyValOption> opt_key_vals, bool accumulated) const override;

  [[nodiscard]] virtual HistogramChartData<size_t, float> get_histogram_used_labels(const std::vector<UserTestQuery>& test_queries,
                                                   const std::string& model_options,size_t num_queries,  size_t num_points, bool accumulated) const override;

  [[nodiscard]] virtual const std::string& get_vocab_ID() const override;
  [[nodiscard]] virtual const std::string& get_vocab_description() const override;

  [[nodiscard]] virtual std::string humanize_and_query(const std::string& and_query) const override;
  [[nodiscard]] virtual std::vector<Keyword*> top_frame_keywords(FrameId frame_ID, const std::string& model_commands, size_t count) const override;

  [[nodiscard]] virtual AutocompleteInputResult get_autocomplete_results(const std::string& query_prefix,
                                                                         size_t result_size,
                                                                         bool with_example_images, const std::string& model_commands) const override;

  [[nodiscard]] virtual DataPackInfo get_info() const override;

 private:
  /** Converts CNF query using keyword IDs to the one using only valid vector indices */
  CnfFormula keyword_IDs_to_vector_indices(CnfFormula ID_query) const;

 private:
  mutable KeywordsContainer _keywords;

  std::vector<std::vector<float>> _presoftmax_data_raw;

  /** Models for this data pack - only classification ones */
  std::unordered_map<std::string, std::unique_ptr<BaseClassificationModel>> _models;

  /** Transformations for this data pack - only classification ones */
  std::unordered_map<std::string, std::unique_ptr<BaseVectorTransform>> _transforms;

  std::unordered_map<std::string, std::unique_ptr<BaseSimUser>> _sim_users;
};
}  // namespace image_ranker