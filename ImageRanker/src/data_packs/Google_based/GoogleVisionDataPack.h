#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "BaseClassificationModel.h"
#include "BaseVectorTransform.h"
#include "KeywordsContainer.h"
#include "SimUser.h"

#include "data_packs/BaseDataPack.h"

namespace image_ranker
{
class [[nodiscard]] GoogleVisionDataPack : public BaseDataPack
{
  /*
   * Methods
   */
 public:
  GoogleVisionDataPack(const BaseImageset* p_is, const StringId& ID, const StringId& target_imageset_ID,
                       const std::string& model_options, const std::string& description, const DataPackStats& stats,
                       const GoogleDataPackRef::VocabData& vocab_data_refs, std::vector<std::vector<float>>&& presoft);

  [[nodiscard]] virtual RankingResult rank_frames(const std::vector<CnfFormula>& user_queries,
                                                  const std::string& model_options, size_t result_size,
                                                  FrameId target_image_ID = ERR_VAL<FrameId>()) const override;

  [[nodiscard]] virtual ModelTestResult test_model(const std::vector<UserTestQuery>& test_queries,
                                                   const std::string& model_options, size_t num_points) const override;

  void cache_up_example_images(const std::vector<const Keyword*>& kws, const std::string& model_commands)
      const;

  [[nodiscard]] virtual std::vector<const Keyword*> get_frame_top_classes(
      FrameId frame_ID, const std::vector<ModelKeyValOption>& opt_key_vals, bool accumulated) const override;

  [[nodiscard]] virtual AutocompleteInputResult get_autocomplete_results(
      const std::string& query_prefix, size_t result_size, bool with_example_images, const std::string& model_commands)
      const override;

  [[nodiscard]] virtual const std::string& get_vocab_ID() const override;
  [[nodiscard]] virtual const std::string& get_vocab_description() const override;
  [[nodiscard]] virtual DataPackInfo get_info() const override;

 private:
  /** Converts CNF query using keyword IDs to the one using only valid vector indices */
  [[nodiscard]] static CnfFormula keyword_IDs_to_vector_indices(CnfFormula ID_query);

  /*
   * Member variables
   */
 private:
  /** Data pack's supported vocabulary. */
  mutable KeywordsContainer _keywords;

  /** Raw DCNN network data. */
  std::vector<std::vector<float>> _presoftmax_data_raw;

  /** Models for this data pack - only classification ones */
  std::unordered_map<std::string, std::unique_ptr<BaseClassificationModel>> _models;

  /** Transformations for this data pack - only classification ones */
  std::unordered_map<std::string, std::unique_ptr<BaseVectorTransform>> _transforms;

  /** Loaded models for simulated users. */
  std::unordered_map<std::string, std::unique_ptr<BaseSimUser>> _sim_users;
};
}  // namespace image_ranker