#pragma once

#include <string>
#include <unordered_map>
#include <vector>
using namespace std::string_literals;

#include "common.h"
#include "config.h"
#include "utility.h"

#include "DataManager.h"
#include "Database.h"
#include "FileParser.h"

#include "data_packs/BaseDataPack.h"
#include "datasets/SelFramesDataset.h"

namespace image_ranker
{
/**
 * Main public API of this library.
 */
class ImageRanker
{
  /****************************
   * Subtypes
   ****************************/
 public:
  /**
   * ImageRanker modes of operation.
   */
  enum class eMode
  {
    cFullAnalytical = 0,
    cCollector = 1,
    cSearchTool = 2
  };

  /**
   * Config structure needed for instantiation of ImageRanker.
   */
  struct Config
  {
    eMode mode;

    std::vector<DatasetDataPackRef> dataset_packs;

    std::vector<ViretDataPackRef> VIRET_packs;
    std::vector<GoogleDataPackRef> Google_packs;
    std::vector<W2vvDataPackRef> W2VV_packs;
  };

  struct Settings
  {
    Settings(const Config& cfg) : config(cfg) {}
    Config config;
  };

  /****************************
   * Methods
   ****************************/
 public:
  static Config parse_data_config_file(eMode mode, const std::string& cfg_filepath, const std::string& data_dir);

  ImageRanker() = delete;

  ImageRanker(const ImageRanker::Config& cfg);

  [[nodiscard]] RankingResultWithFilenames rank_frames(const std::vector<std::string>& user_queries,
                                                       const DataPackId& data_pack_ID,
                                                       const PackModelCommands& model_commands, size_t result_size,
                                                       bool native_lang_queries = false,
                                                       FrameId target_image_ID = ERR_VAL<FrameId>()) const;

  [[nodiscard]] virtual ModelTestResult run_model_test(eUserQueryOrigin queries_origin, const DataPackId& data_pack_ID,
                                                       const PackModelCommands& model_commands,
                                                       bool native_lang_queries = false,
                                                       size_t num_points = NUM_MODEL_TEST_RESULT_POINTS,
                                                       bool normalize_y = true) const;

  /** Only for the given number of generated queries */
  [[nodiscard]] virtual ModelTestResult run_model_test(size_t test_queries_count, const DataPackId& data_pack_ID,
                                                       const PackModelCommands& model_commands,
                                                       bool native_lang_queries = false,
                                                       size_t num_points = NUM_MODEL_TEST_RESULT_POINTS,
                                                       bool normalize_y = true) const;

  /**
   * This processes input queries that come from users, generates results and sends them back.
   */
  std::vector<GameSessionQueryResult> submit_annotator_user_queries(
      const StringId& data_pack_ID, const ::std::string& model_options, size_t user_level, bool with_example_images,
      const std::vector<AnnotatorUserQuery>& user_queries);

  /**
   * Processes and saves provided search session into the database.
   */
  bool submit_search_session(const DataPackId& data_pack_ID, const PackModelCommands& model_commands, size_t user_level,
                             bool with_example_images, FrameId target_frame_ID, eSearchSessionEndStatus end_status,
                             size_t duration, const std::string& sessionId,
                             const std::vector<InteractiveSearchAction>& actions);

  [[nodiscard]] FrameDetailData get_frame_detail_data(FrameId frame_ID, const std::string& data_pack_ID,
                                                      const std::string& model_commands, bool with_example_frames,
                                                      bool accumulated = false);

  [[nodiscard]] std::vector<const SelFrame*> get_random_frame_sequence(const std::string& imageset_ID,
                                                                       size_t seq_len) const;
  [[nodiscard]] const SelFrame* get_random_frame(const std::string& imageset_ID) const;

  [[nodiscard]] AutocompleteInputResult get_autocomplete_results(const std::string& data_pack_ID,
                                                                 const std::string& query_prefix, size_t result_size,
                                                                 bool with_example_images,
                                                                 const std::string& model_options) const;

  [[nodiscard]] std::vector<const SelFrame*> frame_successors(const std::string& imageset_ID, FrameId ID,
                                                              size_t num_succs) const;

  /**
   * Returns struct containing chart data showing Q1, Q2, Q3 for progress of ranks for currently collected search
   * sessions.
   *
   * \param   data_pack_ID    Collected data over this `data_pack_ID` will be used for the chart generation.
   * \param   model_options   Additional specification on what data will be used as source data.
   * \param   max_user_level  Maximum user level records that will be used for generating the data.
   * \return                  Struct with data needed for plotting the chart.
   */
  [[nodiscard]] SearchSessRankChartData get_search_sessions_rank_progress_chart_data(
      const std::string& data_pack_ID, const std::string& model_options = ""s, size_t max_user_level = 9_z,
      size_t min_samples = 50_z, bool normalize = true) const;

  [[nodiscard]] HistogramChartData<size_t, float> get_histogram_used_labels(const std::string& data_pack_ID,
                                                                            const std::string& model_options,
                                                                            size_t num_points, bool accumulated = false,
                                                                            size_t max_user_level = 9_z) const;

  [[nodiscard]] LoadedImagesetsInfo get_loaded_imagesets_info() const;
  [[nodiscard]] LoadedDataPacksInfo get_loaded_data_packs_info() const;

  [[nodiscard]] const std::string& get_frame_filename(const std::string& imageset_ID, size_t imageId) const;
  [[nodiscard]] const SelFrame& get_frame(const std::string& imageset_ID, size_t imageId) const;

 private:
  [[nodiscard]] const BaseImageset& imageset(const std::string& imageset_ID) const
  {
    return *_imagesets.at(imageset_ID);
  }

  [[nodiscard]] const BaseDataPack& data_pack(const std::string& data_pack_ID) const
  {
    return *_data_packs.at(data_pack_ID);
  }

 private:
  Settings _settings;
  FileParser _fileParser;

  /** Manages database data collection, processing and retrieval */
  DataManager _data_manager;

  eMode _mode;

  std::unordered_map<std::string, std::unique_ptr<BaseImageset>> _imagesets;
  std::unordered_map<DataPackId, std::unique_ptr<BaseDataPack>> _data_packs;
};
}  // namespace image_ranker