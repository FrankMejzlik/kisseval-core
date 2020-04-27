#pragma once

#include <string>
#include <vector>

#include "common.h"

namespace image_ranker
{
class BaseDataPack
{
 public:
  BaseDataPack(const StringId& ID, const StringId& target_imageset_ID, const std::string& model_options,
               const std::string& description)
      : _ID(ID), _description(description), _model_options(model_options), _target_imageset_ID(target_imageset_ID)
  {
    LOGI("Loaded data pack: " << ID << std::endl << "\tdescription:" << description << std::endl << "\tdefault_options: " << model_options << std::endl);
  }

  [[nodiscard]] virtual RankingResult rank_frames(const std::vector<CnfFormula>& user_queries,
                                                  PackModelCommands model_commands, size_t result_size,
                                                  FrameId target_image_ID = ERR_VAL<FrameId>()) const = 0;

  [[nodiscard]] virtual ModelTestResult test_model(const std::vector<UserTestQuery>& test_queries,
                                                  PackModelCommands model_commands, size_t num_points) const = 0;

  [[nodiscard]] virtual const std::string& get_vocab_ID() const = 0;
  [[nodiscard]] virtual const std::string& get_vocab_description() const = 0;

  [[nodiscard]] virtual std::string humanize_and_query(const std::string& and_query) const = 0;
  [[nodiscard]] virtual std::vector<Keyword*> top_frame_keywords(FrameId frame_ID) const = 0;

  [[nodiscard]] virtual AutocompleteInputResult get_autocomplete_results(const std::string& query_prefix,
                                                                         size_t result_size,
                                                                         bool with_example_images) const = 0;

  [[nodiscard]] virtual DataPackInfo get_info() const = 0;

  [[nodiscard]] virtual const std::string& get_ID() const { return _ID; };
  [[nodiscard]] virtual const std::string& get_description() const { return _description; };
  [[nodiscard]] virtual const std::string& get_model_options() const { return _model_options; };
  [[nodiscard]] virtual const std::string& target_imageset_ID() const { return _target_imageset_ID; };

 private:
  StringId _ID;
  std::string _description;
  std::string _model_options;
  std::string _target_imageset_ID;
};
}  // namespace image_ranker