#pragma once

#include <string>
#include <vector>

#include "common.h"

namespace image_ranker
{
class BaseImageset;

class BaseDataPack
{
 public:
  BaseDataPack(const BaseImageset* p_is, const StringId& ID, const StringId& target_imageset_ID,
               const std::string& model_options, const std::string& description, const DataPackStats& stats)
      : _stats(stats),
        _p_is(p_is),
        _ID(ID),
        _description(description),
        _model_options(model_options),
        _target_imageset_ID(target_imageset_ID)
  {
    LOGI("Loading data pack: " << ID << std::endl
                               << "\tdescription:" << description << std::endl
                               << "\tdefault_options: " << model_options << std::endl);
  }

  [[nodiscard]] virtual RankingResult rank_frames(
      [[maybe_unused]] const std::vector<CnfFormula>& user_queries, [[maybe_unused]] const std::string& model_options,
      [[maybe_unused]] size_t result_size, [[maybe_unused]] FrameId target_image_ID = ERR_VAL<FrameId>()) const = 0;
  [[nodiscard]] virtual RankingResult rank_frames([[maybe_unused]] const std::vector<std::string>& user_native_queries,
                                                  [[maybe_unused]] const std::string& model_options,
                                                  [[maybe_unused]] size_t result_size,
                                                  [[maybe_unused]] FrameId target_image_ID = ERR_VAL<FrameId>()) const
  {
    // Datapacks do not have to support this
    LOGE("Unsupported data pack feature requested."s);
    PROD_THROW_NOT_SUPP("This data pack does not support this."s);
  };

  [[nodiscard]] virtual std::vector<const Keyword*> get_frame_top_classes(
      [[maybe_unused]] FrameId frame_ID, [[maybe_unused]] const std::vector<ModelKeyValOption>& opt_key_vals,
      [[maybe_unused]] bool accumulated) const
  {
    // Datapacks do not have to support this
    LOGE("Unsupported data pack feature requested."s);
    PROD_THROW_NOT_SUPP("This data pack does not support this."s);
  };

  [[nodiscard]] virtual ModelTestResult test_model([[maybe_unused]] const std::vector<UserTestQuery>& test_queries,
                                                   [[maybe_unused]] const std::string& model_commands,
                                                   [[maybe_unused]] size_t num_points) const = 0;
  [[nodiscard]] virtual ModelTestResult test_model(
      [[maybe_unused]] const std::vector<UserTestNativeQuery>& test_native_queries,
      [[maybe_unused]] const std::string& model_commands, [[maybe_unused]] size_t num_points) const
  {
    // Datapacks do not have to support this
    LOGE("Unsupported data pack feature requested."s);
    PROD_THROW_NOT_SUPP("This data pack does not support this."s);
  };

  [[nodiscard]] virtual HistogramChartData<size_t, float> get_histogram_used_labels(
      [[maybe_unused]] const std::vector<UserTestQuery>& test_queries,
      [[maybe_unused]] const std::string& model_commands, [[maybe_unused]] size_t num_queries,
      [[maybe_unused]] size_t num_points, [[maybe_unused]] bool accumulated) const
  {
    // Datapacks do not have to support this
    LOGE("Unsupported data pack feature requested.");
    PROD_THROW_NOT_SUPP("Unsupported data pack feature requested.");
  };

  [[nodiscard]] virtual const std::string& get_vocab_ID() const = 0;
  [[nodiscard]] virtual const std::string& get_vocab_description() const = 0;

  [[nodiscard]] virtual const DataPackStats& get_stats() const { return _stats; };



  [[nodiscard]] virtual AutocompleteInputResult get_autocomplete_results(const std::string& query_prefix,
                                                                         size_t result_size, bool with_example_images,
                                                                         const std::string& model_commands) const = 0;

  virtual void cache_up_example_images([[maybe_unused]] const std::vector<const Keyword*>& kws,
                                       [[maybe_unused]] const std::string& model_commands) const
  {
    LOGE("Unsupported data pack feature requested.");
    PROD_THROW_NOT_SUPP("This data pack does not support this.");
  };

  [[nodiscard]] virtual DataPackInfo get_info() const = 0;

  [[nodiscard]] virtual const std::string& get_ID() const { return _ID; };
  [[nodiscard]] virtual const BaseImageset* get_imageset_ptr() const { return _p_is; };
  [[nodiscard]] virtual const std::string& get_description() const { return _description; };
  [[nodiscard]] virtual const std::string& get_model_options() const { return _model_options; };
  [[nodiscard]] virtual const std::string& target_imageset_ID() const { return _target_imageset_ID; };

 protected:
  DataPackStats _stats;

 private:
  const BaseImageset* _p_is;
  StringId _ID;
  std::string _description;
  std::string _model_options;
  std::string _target_imageset_ID;
};
}  // namespace image_ranker