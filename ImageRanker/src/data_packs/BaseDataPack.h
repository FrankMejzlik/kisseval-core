#pragma once

#include <string>
#include <vector>

#include "common.h"

namespace image_ranker
{
class BaseImageset;

class [[nodiscard]] BaseDataPack
{
 public:
  // -----------------------------------------
  // We need virtual dctor.
  BaseDataPack() = default;
  BaseDataPack(const BaseDataPack& other) = default;
  BaseDataPack(BaseDataPack && other) = default;
  BaseDataPack& operator=(const BaseDataPack& other) = default;
  BaseDataPack& operator=(BaseDataPack&& other) = default;
  virtual ~BaseDataPack() noexcept = default;
  // -----------------------------------------

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

  /**
   * Ranks frames and returns sorted ranking result using already parsed CNF-like queries.
   *
   * \see image_ranker::ViretDataPack
   * \see image_ranker::GoogleVisionDataPack
   *
   * \param user_queries
   * \param model_options
   * \param result_size
   * \param target_image_ID
   * \return Sorted frames with meta-data.
   */
  [[nodiscard]] virtual RankingResult rank_frames(
      [[maybe_unused]] const std::vector<CnfFormula>& user_queries, [[maybe_unused]] const std::string& model_options,
      [[maybe_unused]] size_t result_size, [[maybe_unused]] FrameId target_image_ID = ERR_VAL<FrameId>()) const = 0;

  /**
   * Ranks frames and returns sorted ranking with native language queries.
   *
   * \see image_ranker::W2vvDataPack
   *
   * \param user_native_queries
   * \param model_options
   * \param result_size
   * \param target_image_ID
   * \return Sorted frames with meta-data.
   */
  [[nodiscard]] virtual RankingResult rank_frames([[maybe_unused]] const std::vector<std::string>& user_native_queries,
                                                  [[maybe_unused]] const std::string& model_options,
                                                  [[maybe_unused]] size_t result_size,
                                                  [[maybe_unused]] FrameId target_image_ID = ERR_VAL<FrameId>()) const
  {
    // Datapacks do not have to support this
    LOGE("Unsupported data pack feature requested."s);
    PROD_THROW_NOT_SUPP("This data pack does not support this."s);
  };

  /**
   * Returns top keywords for the given frame.
   *
   * \param frame_ID
   * \param opt_key_vals
   * \param accumulated
   * \return Sequence of top keyword pointers.
   */
  [[nodiscard]] virtual std::vector<const Keyword*> get_frame_top_classes(
      [[maybe_unused]] FrameId frame_ID, [[maybe_unused]] const std::vector<ModelKeyValOption>& opt_key_vals,
      [[maybe_unused]] bool accumulated) const
  {
    // Datapacks do not have to support this
    LOGE("Unsupported data pack feature requested."s);
    PROD_THROW_NOT_SUPP("This data pack does not support this."s);
  };

  /**
   * Runs test on provided CNF-like test queries.
   *
   * \see image_ranker::ViretDataPack
   * \see image_ranker::GoogleVisionDataPack
   *
   * \param test_queries
   * \param model_commands
   * \param num_points
   * \return Chart data representing test results.
   */
  [[nodiscard]] virtual ModelTestResult test_model([[maybe_unused]] const std::vector<UserTestQuery>& test_queries,
                                                   [[maybe_unused]] const std::string& model_commands,
                                                   [[maybe_unused]] size_t num_points) const = 0;

  /**
   * Runs test on provided native language querise.
   *
   * \see image_ranker::W2vvDataPack
   *
   * \param test_queries
   * \param model_commands
   * \param num_points
   * \return Chart data representing test results.
   */
  [[nodiscard]] virtual ModelTestResult test_model(
      [[maybe_unused]] const std::vector<UserTestNativeQuery>& test_native_queries,
      [[maybe_unused]] const std::string& model_commands, [[maybe_unused]] size_t num_points) const
  {
    // Datapacks do not have to support this
    LOGE("Unsupported data pack feature requested."s);
    PROD_THROW_NOT_SUPP("This data pack does not support this."s);
  };

  /**
   * Returns data about real users using top-K labels same as network.
   *
   * \see image_ranker::W2vvDataPack
   *
   * \param test_queries
   * \param model_commands
   * \param num_queries
   * \param num_points
   * \param accumulated
   * \return Chart data representing test results.
   */
  [[nodiscard]] virtual HistogramChartData<size_t, float> get_histogram_used_labels(
      [[maybe_unused]] const std::vector<UserTestQuery>& test_queries,
      [[maybe_unused]] const std::string& model_commands, [[maybe_unused]] size_t num_queries,
      [[maybe_unused]] size_t num_points, [[maybe_unused]] bool accumulated) const
  {
    // Datapacks do not have to support this
    LOGE("Unsupported data pack feature requested.");
    PROD_THROW_NOT_SUPP("Unsupported data pack feature requested.");
  };

  /**
   * Returns the neares keywords for the provided string prefix.
   *
   * \param query_prefix
   * \param result_size
   * \param with_example_images
   * \param model_commands
   * \return Structure containing resulting keywords.
   */
  [[nodiscard]] virtual AutocompleteInputResult get_autocomplete_results(
      const std::string& query_prefix, size_t result_size, bool with_example_images, const std::string& model_commands)
      const = 0;
  /**
   * Caches up example images for the given keywords.
   *
   * \param kws
   * \param model_commands
   */
  virtual void cache_up_example_images([[maybe_unused]] const std::vector<const Keyword*>& kws,
                                       [[maybe_unused]] const std::string& model_commands) const
  {
    LOGE("Unsupported data pack feature requested.");
    PROD_THROW_NOT_SUPP("This data pack does not support this.");
  };

  [[nodiscard]] virtual const std::string& get_vocab_ID() const = 0;
  [[nodiscard]] virtual const std::string& get_vocab_description() const = 0;
  [[nodiscard]] virtual const DataPackStats& get_stats() const { return _stats; };
  [[nodiscard]] virtual DataPackInfo get_info() const = 0;
  [[nodiscard]] virtual const std::string& get_ID() const { return _ID; };
  [[nodiscard]] virtual const BaseImageset* get_imageset_ptr() const { return _p_is; };
  [[nodiscard]] virtual const std::string& get_description() const { return _description; };
  [[nodiscard]] virtual const std::string& get_model_options() const { return _model_options; };
  [[nodiscard]] virtual const std::string& target_imageset_ID() const { return _target_imageset_ID; };

 protected:
  /** Statistics about this data pack. */
  DataPackStats _stats;

 private:
  const BaseImageset* _p_is;
  StringId _ID;
  std::string _description;
  std::string _model_options;
  std::string _target_imageset_ID;
};
}  // namespace image_ranker