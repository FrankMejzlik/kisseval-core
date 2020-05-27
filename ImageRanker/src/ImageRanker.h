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
 * Public API for evaluation library used for keyword-based models.
 *
 * On instantiation it loads all cofigured data packs and preprocesses
 * them for later usage. When instantiated it handles SEQUENTIAL requests
 * and provides results back. It also accepts submit request and stores them
 * inside the database.
 */
class [[nodiscard]] ImageRanker
{
  friend class Tester;

  /*
   * Subtypes
   */
 public:
  /**
   * ImageRanker modes of operation.
   *
   * Currently not used - only fully analytical version is used.
   */
  enum class eMode
  {
    cFullAnalytical = 0,
    cCollector = 1,
    cSearchTool = 2
  };

  /**
   * Config structure needed used for instantiation of ImageRanker.
   */
  struct Config
  {
    /** Mode of operation application will be initialized in. */
    eMode mode;

    /** Imagesets to be loaded. */
    std::vector<DatasetDataPackRef> dataset_packs;

    /** VIRET type data packs to be loaded. */
    std::vector<ViretDataPackRef> VIRET_packs;

    /** Google type data packs to be loeaded. */
    std::vector<GoogleDataPackRef> Google_packs;

    /** W2VV type data packs to be loeaded. */
    std::vector<W2vvDataPackRef> W2VV_packs;
  };

  /**
   * Container for current ImageRanker settings.
   */
  struct Settings
  {
    Settings(const Config& cfg) : config(cfg) {}

    /** Config used while initialization. */
    Config config;
  };

  /*
   * Methods
   */
 public:
  // --------------------------
  // We don't allow anything else than main ctor.
  ImageRanker() = delete;
  ImageRanker(const ImageRanker& other) = delete;
  ImageRanker(ImageRanker && other) = delete;
  ImageRanker& operator=(const ImageRanker& other) = delete;
  ImageRanker& operator=(ImageRanker&& other) = delete;
  ~ImageRanker() noexcept = default;
  // --------------------------

  /**
   * The only constructor for ImageRanker.
   *
   * \see image_ranker::Config
   *
   * \param cfg Parsed \ref Config instance.
   */
  ImageRanker(const ImageRanker::Config& cfg);

  /**
   * Parses config JSON file and returns Config instance that is provided to ImageRanker ctor.
   *
   * \param mode          Mode of operation for ImageRanker to be initialized in.
   * \param cfg_filepath  Filepath of config JSON file to be parsed.
   * \param data_dit      Root directory of data files.
   * \return              Returns copy of parsed Config for ImageRanker construction.
   */
  [[nodiscard]] static Config parse_data_config_file(eMode mode, const std::string& cfg_filepath,
                                                     const std::string& data_dir);

  /**
   * Ranks frames against provided query and with provided model configuration.
   *
   * \param user_queries  User query string (or multiple if temporal).
   *                      EXAMPLE QUERY: ("&-20+--1+-3++-55+-333+", "&-2+--12+-3++-424+-33+")
   * \param data_pack_ID        Data pack ID we want to rank frames from.
   * \param model_options       Model options string (e.g. "model=w2vv_bow_plain;").
   * \param result_size         Number of frames in result.
   * \param native_lang_queries If provided queries are in natural language (e.g. "Cat in a house.").
   * \param target_image_ID     Frame that is the target.
   * \return  Ranking result containing ordered frames.
   */
  [[nodiscard]] RankingResultWithFilenames rank_frames(
      const std::vector<std::string>& user_queries, const std::string& data_pack_ID, const std::string& model_commands,
      size_t result_size, bool native_lang_queries = false, FrameId target_image_ID = ERR_VAL<FrameId>()) const;

  /**
   * Evaluates collected user queries and returns cummulative chart data showing it's effectivity.
   *
   * \param queries_origin      User query string (or multiple if temporal).
   * \param data_pack_ID        Data pack ID we want to rank frames from.
   * \param model_options       Model options string (e.g. "model=w2vv_bow_plain;").
   * \param native_lang_queries If provided queries are in natural language (e.g. "Cat in a house.").
   * \param num_points          Number of x points in result chart data.
   * \param normalize_y         If y values should be normalized into percentage.
   * \return              Returns data needed for visualisation of effectivity chart.
   */
  [[nodiscard]] virtual ModelTestResult run_model_test(
      eUserQueryOrigin queries_origin, const std::string& data_pack_ID, const std::string& model_options,
      bool native_lang_queries = false, size_t num_points = NUM_MODEL_TEST_RESULT_POINTS, bool normalize_y = true)
      const;

  /**
   * This processes input queries that come from users, generates results and sends them back.
   *
   * \param data_pack_ID        Data pack ID we want to rank frames from.
   * \param model_options       Model options string (e.g. "model=w2vv_bow_plain;").
   * \param user_level          What level has the user that provied tese queries.
   * \param with_example_images If example images were presented during annotation.
   * \param user_queries        Vector of user queries
   * \return  Returns results of a "game" that shows what keywords are closest.
   */
  std::vector<GameSessionQueryResult> submit_annotator_user_queries(
      const StringId& data_pack_ID, const ::std::string& model_options, size_t user_level, bool with_example_images,
      const std::vector<AnnotatorUserQuery>& user_queries);

  /**
   * Processes and saves provided search session into the database.
   *
   * \param data_pack_ID        Data pack ID we want to rank frames from.
   * \param model_options       Model options string (e.g. "model=w2vv_bow_plain;").
   * \param user_level          What level has the user that provied tese queries.
   * \param with_example_images If example images were presented during annotation.
   * \param target_image_ID
   * \param end_status
   * \param duration
   * \param sessionId
   * \param actions
   * \return  Returns `true` on success.
   */
  void submit_search_session(const DataPackId& data_pack_ID, const PackModelCommands& model_options, size_t user_level,
                             bool with_example_images, FrameId target_frame_ID, eSearchSessionEndStatus end_status,
                             size_t duration, const std::string& sessionId,
                             const std::vector<InteractiveSearchAction>& actions);

  [[nodiscard]] FrameDetailData get_frame_detail_data(FrameId frame_ID, const std::string& data_pack_ID,
                                                      const std::string& model_commands, bool with_example_frames,
                                                      bool accumulated = false);

  [[nodiscard]] std::vector<const SelFrame*> get_random_frame_sequence(const std::string& imageset_ID, size_t seq_len)
      const;
  [[nodiscard]] const SelFrame* get_random_frame(const std::string& imageset_ID) const;

  [[nodiscard]] AutocompleteInputResult get_autocomplete_results(
      const std::string& data_pack_ID, const std::string& query_prefix, size_t result_size, bool with_example_images,
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
      const std::string& data_pack_ID, const std::string& model_options = ""s, size_t max_user_level = 9,
      size_t min_samples = 50, bool normalize = true) const;

  [[nodiscard]] HistogramChartData<size_t, float> get_histogram_used_labels(
      const std::string& data_pack_ID, const std::string& model_options, size_t num_points, bool accumulated = false,
      size_t max_user_level = 9) const;

  [[nodiscard]] LoadedImagesetsInfo get_loaded_imagesets_info() const;
  [[nodiscard]] LoadedDataPacksInfo get_loaded_data_packs_info() const;

  [[nodiscard]] const std::string& get_frame_filename(const std::string& imageset_ID, size_t imageId) const;
  [[nodiscard]] const SelFrame& get_frame(const std::string& imageset_ID, size_t imageId) const;

 private:
  /**
   * Returns reference to loaded imageset with provided imageset_ID.
   *
   * \param  imageset_ID Imageset ID we want reference to.
   * \return Reference to desired imageset. Throws if not found.
   */
  [[nodiscard]] const BaseImageset& imageset(const std::string& imageset_ID) const
  {
    return *_imagesets.at(imageset_ID);
  }

  /**
   * Returns reference to loaded data pack with provided data_pack_ID.
   *
   * \param  data_pack_ID Data pack ID we want reference to.
   * \return Reference to desired data pack. Throws if not found.
   */
  [[nodiscard]] const BaseDataPack& data_pack(const std::string& data_pack_ID) const
  {
    return *_data_packs.at(data_pack_ID);
  }

  /*
   * Member variables
   */
 private:
  /** Instance holding ImageRanker settings. */
  Settings _settings;

  /** File parser used for data parsing. */
  FileParser _fileParser;

  /** Manages database data collection, processing and retrieval */
  DataManager _data_manager;

  /** Mode of operation. */
  eMode _mode;

  /** Loaded imagesets. */
  std::unordered_map<std::string, std::unique_ptr<BaseImageset>> _imagesets;

  /** Loaded data packs. */
  std::unordered_map<DataPackId, std::unique_ptr<BaseDataPack>> _data_packs;
};
}  // namespace image_ranker