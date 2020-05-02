#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "KeywordsContainer.h"

#include "data_packs/BaseDataPack.h"

#include "BaseW2vvModel.h"
#include "SimUser.h"

namespace image_ranker
{
class W2vvDataPack : public BaseDataPack
{
 public:
  W2vvDataPack(const BaseImageset* p_is, const StringId& ID, const StringId& target_imageset_ID, const std::string& model_options,
               const std::string& description, const W2vvDataPackRef::VocabData& vocab_data_refs,
               std::vector<std::vector<float>>&& frame_features, Matrix<float>&& kw_features, Vector<float>&& kw_bias_vec,
               Matrix<float>&& kw_PCA_mat, Vector<float>&& kw_PCA_mean_vec);

  [[nodiscard]] virtual RankingResult rank_frames(const std::vector<CnfFormula>& user_queries,
                                                  PackModelCommands model_commands, size_t result_size,
                                                  FrameId target_image_ID = ERR_VAL<FrameId>()) const override;
  [[nodiscard]] virtual RankingResult rank_frames(const std::vector<std::string>& user_native_queries,
                                                  PackModelCommands model_commands, size_t result_size,
                                                  FrameId target_image_ID = ERR_VAL<FrameId>()) const override;

  [[nodiscard]] virtual ModelTestResult test_model(const std::vector<UserTestQuery>& test_queries,
                                                   PackModelCommands model_commands, size_t num_points) const override;
  [[nodiscard]] virtual ModelTestResult test_model(const std::vector<UserTestNativeQuery>& test_native_queries,
                                                   PackModelCommands model_commands, size_t num_points) const override;

  [[nodiscard]] virtual const std::string& get_vocab_ID() const override;
  [[nodiscard]] virtual const std::string& get_vocab_description() const override;

  [[nodiscard]] virtual std::string humanize_and_query(const std::string& and_query) const override;
  [[nodiscard]] virtual std::vector<Keyword*> top_frame_keywords(FrameId frame_ID, PackModelCommands model_commands, size_t count) const override;

  [[nodiscard]] virtual AutocompleteInputResult get_autocomplete_results(const std::string& query_prefix,
                                                                         size_t result_size,
                                                                         bool with_example_images, const std::string& model_commands) const override;

  [[nodiscard]] virtual DataPackInfo get_info() const override;

 private:
  CnfFormula native_query_to_CNF_formula(const std::string& native_query) const;

 private:
  KeywordsContainer _keywords;

  Matrix<float> _kw_features;
  Vector<float> _kw_bias_vec;
  Matrix<float> _kw_PCA_mat;
  Vector<float> _kw_PCA_mean_vec;

  Matrix<float> _features_of_frames;

  /** Models for this data pack - only classification ones */
  std::unordered_map<std::string, std::unique_ptr<BaseW2vvModel>> _models;

  std::unordered_map<std::string, std::unique_ptr<BaseSimUser>> _sim_users;
};
}  // namespace image_ranker