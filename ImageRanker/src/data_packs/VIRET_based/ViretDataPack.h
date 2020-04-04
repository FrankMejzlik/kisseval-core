#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "KeywordsContainer.h"

#include "data_packs/BaseDataPack.h"

#include "BaseClassificationModel.h"
#include "BaseClassificationTransformation.h"

class ViretDataPack : public BaseDataPack
{
 public:
  ViretDataPack(const StringId& ID, const StringId& target_imageset_ID, const std::string& description,
                const ViretDataPackRef::VocabData& vocab_data_refs, std::vector<std::vector<float>>&& presoft,
                std::vector<std::vector<float>>&& softmax_data, std::vector<std::vector<float>>&& feas_data);

  [[nodiscard]] RankingResult rank_frames(const std::vector<std::string>& user_queries,
                                          PackModelCommands model_commands, size_t result_size,
                                          FrameId target_image_ID = ERR_VAL<FrameId>()) const override;

  [[nodiscard]] const std::string& get_vocab_ID() const override;
  [[nodiscard]] const std::string& get_vocab_description() const override;

  [[nodiscard]] std::string humanize_and_query(const std::string& and_query) const override;
  [[nodiscard]] std::vector<Keyword*> top_frame_keywords(FrameId frame_ID) const override;

  [[nodiscard]] AutocompleteInputResult get_autocomplete_results(const std::string& query_prefix, size_t result_size,
                                                              bool with_example_images) const override;

  [[nodiscard]] DataPackInfo get_info() const override;

 private:
  KeywordsContainer _keywords;

  std::vector<std::vector<float>> _presoftmax_data;
  std::vector<std::vector<float>> _softmax_data;
  std::vector<std::vector<float>> _feas_data;

  /** Models for this data pack - only classification ones */
  std::unordered_map<StringId, std::unique_ptr<BaseClassificationModel>> _models;

  /** Transformations for this data pack - only classification ones */
  std::unordered_map<StringId, std::unique_ptr<BaseClassificationTransformation>> _transforms;

  std::unordered_map<StringId, VecMat> _transformed_data;
};
