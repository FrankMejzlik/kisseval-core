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

  FrameDetailData get_frame_detail_data(FrameId frame_ID, const std::string& data_pack_ID,
                                        const std::string& model_commands, bool with_example_frames,
                                        bool accumulated = false);

  std::vector<const SelFrame*> get_random_frame_sequence(const std::string& imageset_ID, size_t seq_len) const;
  const SelFrame* get_random_frame(const std::string& imageset_ID) const;

  AutocompleteInputResult get_autocomplete_results(const std::string& data_pack_ID, const std::string& query_prefix,
                                                   size_t result_size, bool with_example_images,
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
  [[nodiscard]] QuantileLineChartData get_search_sessions_rank_progress_chart_data(
      const std::string& data_pack_ID, const std::string& model_options = ""s, size_t max_user_level = 9_z) const;

  LoadedImagesetsInfo get_loaded_imagesets_info() const;
  LoadedDataPacksInfo get_loaded_data_packs_info() const;

  const std::string& get_frame_filename(const std::string& imageset_ID, size_t imageId) const;
  const SelFrame& get_frame(const std::string& imageset_ID, size_t imageId) const;

 private:
  const BaseImageset& imageset(const std::string& imageset_ID) const { return *_imagesets.at(imageset_ID); }

  const BaseDataPack& data_pack(const std::string& data_pack_ID) const { return *_data_packs.at(data_pack_ID); }

 private:
  Settings _settings;
  FileParser _fileParser;

  /** Manages database data collection, processing and retrieval */
  DataManager _data_manager;

  eMode _mode;

  std::unordered_map<std::string, std::unique_ptr<BaseImageset>> _imagesets;

  std::unordered_map<DataPackId, std::unique_ptr<BaseDataPack>> _data_packs;

  /****************************
   * Member variables
   ****************************/
 private:
  // =====================================
  //  NOT REFACTORED CODE BELOW
  // =====================================

#if 0
  public:
   
    

    std::tuple<KeywordsGeneralStatsTuple, ScoringsGeneralStatsTuple, AnnotatorDataGeneralStatsTuple,
      RankerDataGeneralStatsTuple>
      GetGeneralStatistics(DataId data_ID, UserDataSourceId dataSourceType) const;

    KeywordsGeneralStatsTuple GetGeneralKeywordsStatistics(DataId data_ID,
      UserDataSourceId dataSourceType) const;
    ScoringsGeneralStatsTuple GetGeneralScoringStatistics(DataId data_ID,
      UserDataSourceId dataSourceType) const;
    AnnotatorDataGeneralStatsTuple GetGeneralAnnotatorDataStatistics(DataId data_ID,
      UserDataSourceId dataSourceType) const;
    RankerDataGeneralStatsTuple GetGeneralRankerDataStatistics(DataId data_ID,
      UserDataSourceId dataSourceType) const;

    std::string ExportDataFile(DataId data_ID, eExportFileTypeId fileType, const std::string& outputFilepath,
      bool native) const;

    bool ExportUserAnnotatorData(DataId data_ID, UserDataSourceId dataSource,
      const std::string& outputFilepath, bool native) const;
    bool ExportNormalizedScores(DataId data_ID, const std::string& outputFilepath) const;
    bool ExportUserAnnotatorNumHits(DataId data_ID, UserDataSourceId dataSource,
      const std::string& outputFilepath) const;

    size_t MapIdToVectorIndex(size_t id) const;
    KeywordsContainer* GetCorrectKwContainerPtr(DataId data_ID) const;

    std::vector<std::pair<TestSettings, ChartData>> RunGridTest(const std::vector<TestSettings>& testSettings);

    void RecalculateHypernymsInVectorUsingSum(std::vector<float>& binVectorRef);
    void RecalculateHypernymsInVectorUsingMax(std::vector<float>& binVectorRef);
    void LowMem_RecalculateHypernymsInVectorUsingSum(std::vector<float>& binVectorRef);
    void LowMem_RecalculateHypernymsInVectorUsingMax(std::vector<float>& binVectorRef);
    /*!
     * Gets all data about image with provided ID
     *
     * \param imageId
     * \return
     */
    const Image* GetImageDataById(size_t imageId) const;
    Image* GetImageDataById(size_t imageId);

    const Keyword* GetKeywordConstPtr(eVocabularyId kwDataType, size_t keywordId) const;
    Keyword* GetKeywordPtr(eVocabularyId kwDataType, size_t keywordId);



    void SubmitUserDataNativeQueries(std::vector<std::tuple<size_t, std::string, std::string>>& queries);

    const Image* GetRandomImage() const;
    std::tuple<const Image*, bool, size_t> GetCouplingImage() const;
    std::tuple<const Image*, bool, size_t> GetCoupledImagesNative() const;




    Keyword* GetKeywordByVectorIndex(DataId data_ID, size_t index);



    std::pair<uint8_t, uint8_t> GetGridTestProgress() const;

    std::vector<ChartData> RunModelSimulatedQueries(std::string run_name, DataId data_ID,
      InputDataTransformId aggId, RankingModelId modelId,
      UserDataSourceId dataSource,
      const SimulatedUserSettings& simulatedUserSettings,
      const RankingModelSettings& aggModelSettings,
      const InputDataTransformSettings& netDataTransformSettings,
      size_t expansionSettings) const;

    ChartData RunModelTestWrapper(DataId data_ID, InputDataTransformId aggId, RankingModelId modelId,
      UserDataSourceId dataSource, const SimulatedUserSettings& simulatedUserSettings,
      const RankingModelSettings& aggModelSettings,
      const InputDataTransformSettings& netDataTransformSettings,
      size_t expansionSettings) const;

    std::vector<std::vector<UserImgQuery>> DoQueryAndExpansion(DataId data_ID,
      const std::vector<std::vector<UserImgQuery>>& origQuery,
      size_t setting) const;
    std::vector<std::vector<UserImgQuery>> DoQueryOrExpansion(DataId data_ID,
      const std::vector<std::vector<UserImgQuery>>& origQuery,
      size_t setting) const;

    std::tuple<UserAccuracyChartData, UserAccuracyChartData> GetStatisticsUserKeywordAccuracy(
      UserDataSourceId queriesSource = UserDataSourceId::cAll) const;

    std::string GetKeywordDescriptionByWordnetId(DataId data_ID, size_t wordnetId)
    {
      KeywordsContainer* pkws{ nullptr };

      switch (std::get<0>(data_ID))
      {
      case eVocabularyId::VIRET_1200_WORDNET_2019:
        pkws = _pViretKws;
        break;

      case eVocabularyId::GOOGLE_AI_20K_2019:
        pkws = _pGoogleKws;
        break;

      default:
        LOG_ERROR("Invalid keyword data type.");
      }

      return pkws->GetKeywordDescriptionByWordnetId(wordnetId);
    }

#if TRECVID_MAPPING

    //! return: <elapsed time, [<video ID, shot ID>]>
    std::tuple<float, std::vector<std::pair<size_t, size_t>>> TrecvidGetRelevantShots(
      DataId data_ID, const std::vector<std::string>& queriesEncodedPlaintext, size_t numResults,
      InputDataTransformId aggId, RankingModelId modelId, const RankingModelSettings& modelSettings,
      const InputDataTransformSettings& aggSettings, float elapsedTime, size_t imageId = ERR_VAL<size_t>());

#endif
    std::vector<std::vector<UserImgQuery>> GetSimulatedQueries(DataId data_ID, UserDataSourceId dataSource,
      const SimulatedUser& simUser) const;

    std::vector<std::vector<UserImgQuery>> GetSimulatedQueries(DataId data_ID, size_t num_quries,
      bool sample_targets, const SimulatedUser& simUser) const;
    // ^^^^^^^^^^^^^^^^^^^^^^^
    //////////////////////////
    //    API Methods
    //////////////////////////

  private:
    SimulatedUser GetSimUserSettings(const SimulatedUserSettings& settings) const;

    void ComputeApproxDocFrequency(size_t aggregationGuid, float treshold);

    size_t GetRandomImageId() const;

    std::string GetKeywordByWordnetId(DataId data_ID, size_t wordnetId) const
    {
      KeywordsContainer* pkws{ nullptr };

      // Save shortcuts
      switch (std::get<0>(data_ID))
      {
      case eVocabularyId::VIRET_1200_WORDNET_2019:
        pkws = _pViretKws;
        break;

      case eVocabularyId::GOOGLE_AI_20K_2019:
        pkws = _pGoogleKws;
        break;
      }
      return pkws->GetKeywordByWordnetId(wordnetId);
    }

    std::string GetImageFilenameById(size_t imageId) const;

    void RunGridTestsFromTo(std::vector<std::pair<TestSettings, ChartData>>* pDest, size_t fromIndex, size_t toIndex);

    bool LoadKeywordsFromDatabase(Database::Type type);
    bool LoadImagesFromDatabase(Database::Type type);
    std::vector<std::pair<std::string, float>> GetHighestProbKeywords(DataId data_ID, size_t imageId,
      size_t N) const;

    std::vector<std::string> TokenizeAndQuery(std::string_view query) const;
    std::vector<std::string> StringenizeAndQuery(DataId data_ID, const std::string& query) const;

    size_t GetNumImages() const { return _images.size(); };

    void InitializeGridTests();

    std::vector<UserImgQueryRaw>& GetCachedQueriesRaw(UserDataSourceId dataSource) const;
    std::vector<UserDataNativeQuery>& GetUserAnnotationNativeQueriesCached() const;

    std::vector<std::vector<UserImgQuery>>& GetCachedQueries(DataId data_ID,
      UserDataSourceId dataSource) const;

    std::vector<std::vector<UserImgQuery>> GetExtendedRealQueries(DataId data_ID, UserDataSourceId dataSource,
      const SimulatedUser& simUser) const;

    UserImgQuery GetSimulatedQueryForImage(size_t imageId, const SimulatedUser& simUser) const;

    std::string EncodeAndQuery(const std::string& query) const;



    bool LoadRepresentativeImages(DataId data_ID, Keyword* pKw);

    void GenerateBestHypernymsForImages();
    void PrintIntActionsCsv() const;

#endif
};
}  // namespace image_ranker