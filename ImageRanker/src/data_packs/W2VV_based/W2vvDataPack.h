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
class W2vvDataPack : public BaseDataPack
{
 public:
  W2vvDataPack(const StringId& ID, const StringId& target_imageset_ID, const std::string& model_options,
                const std::string& description, const W2vvDataPackRef::VocabData& vocab_data_refs,
                std::vector<std::vector<float>>&& presoft);

  [[nodiscard]] virtual RankingResult rank_frames(const std::vector<CnfFormula>& user_queries,
                                                  PackModelCommands model_commands, size_t result_size,
                                                  FrameId target_image_ID = ERR_VAL<FrameId>()) const override;

  [[nodiscard]] virtual ModelTestResult test_model(const std::vector<UserTestQuery>& test_queries,
                                                   PackModelCommands model_commands, size_t num_points) const override;

  [[nodiscard]] std::vector<UserTestQuery> process_sim_user(const BaseVectorTransform& transformed_data,
                                                            const KeywordsContainer& keywords,
                                                            const std::vector<UserTestQuery>& test_user_queries) const;

  [[nodiscard]] virtual const std::string& get_vocab_ID() const override;
  [[nodiscard]] virtual const std::string& get_vocab_description() const override;

  [[nodiscard]] virtual std::string humanize_and_query(const std::string& and_query) const override;
  [[nodiscard]] virtual std::vector<Keyword*> top_frame_keywords(FrameId frame_ID) const override;

  [[nodiscard]] virtual AutocompleteInputResult get_autocomplete_results(const std::string& query_prefix,
                                                                         size_t result_size,
                                                                         bool with_example_images) const override;

  [[nodiscard]] virtual DataPackInfo get_info() const override;

 private:
  /** Converts CNF query using keyword IDs to the one using only valid vector indices */
  CnfFormula keyword_IDs_to_vector_indices(CnfFormula ID_query) const;

 private:
  KeywordsContainer _keywords;

  std::vector<std::vector<float>> _presoftmax_data_raw;

  /** Models for this data pack - only classification ones */
  std::unordered_map<std::string, std::unique_ptr<BaseClassificationModel>> _models;

  /** Transformations for this data pack - only classification ones */
  std::unordered_map<std::string, std::unique_ptr<BaseVectorTransform>> _transforms;

  std::unordered_map<std::string, std::unique_ptr<BaseSimUser>> _sim_users;
};
}  // namespace image_ranker