#pragma once

#include <string>
#include <unordered_map>
#include <vector>
using namespace std::string_literals;

#include "common.h"
#include "config.h"
#include "utility.h"

#include "Database.h"
#include "FileParser.h"

#include "datasets/SelFramesDataset.h"
#include "data_packs/BaseDataPack.h"


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
    std::vector<BowDataPackRef> BoW_packs;
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
  ImageRanker() = delete;

  ImageRanker(const ImageRanker::Config& cfg);

  bool Initialize();
  bool InitializeFullMode();

  [[nodiscard]]
  RankingResult rank_frames(const std::vector<std::string>& user_queries, eDataPackType data_pack_type,
                                DataPackId data_pack_ID, PackModelId pack_model_ID, PackModelCommands model_commands,
                                size_t result_size, FrameId target_image_ID = ERR_VAL<FrameId>()) const;
  
  [[nodiscard]]
  const FileParser* get_file_parser() const { return &_fileParser; }

 private:
  Settings _settings;
  Database _db;
  FileParser _fileParser;

  eMode _mode;

  std::unordered_map<DataPackId, std::unique_ptr<BaseDataset>> _datasets;

  std::unordered_map<DataPackId, std::unique_ptr<BaseDataPack>> _VIRET_packs;
  /*std::unordered_map<DataPackId, std::unique_ptr<GoogleModelPackRef>> _Google_packs;
  std::unordered_map<DataPackId, std::unique_ptr<BoWModelPackRef>> _BoW_packs;*/

  /****************************
   * Member variables
   ****************************/
 private:
  // =====================================
  //  NOT REFACTORED CODE BELOW
  // =====================================

#if 0
public:
  //! Constructor with data from files with presoftmax file
  ImageRanker(const std::string& imagesPath, const std::vector<KeywordsFileRef>& keywordsFileRefs,
              const std::vector<DataFileSrc>& imageScoringFileRefs,
              const std::vector<DataFileSrc>& imageSoftmaxScoringFileRefs = std::vector<DataFileSrc>(),
              const std::vector<DataFileSrc>& deepFeaturesFileRefs = std::vector<DataFileSrc>(),
              const std::string& imageToIdMapFilepath = ""s, size_t idOffset = 1ULL, eMode mode = DEFAULT_MODE,
              const std::vector<KeywordsFileRef>& imageToIdMapFilepaths = std::vector<KeywordsFileRef>());

  ~ImageRanker() noexcept = default;

  Keyword* GetKeywordPtr(eVocabularyId kwType, const std::string& wordString);

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

  std::tuple<std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>>,
             std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>>,
             std::vector<std::pair<size_t, std::string>>>
  GetImageKeywordsForInteractiveSearch(size_t imageId, size_t numResults, DataId data_ID,
                                       bool withExampleImages);

  void SubmitInteractiveSearchSubmit(DataId data_ID, InteractiveSearchOrigin originType, size_t imageId,
                                     RankingModelId modelId, InputDataTransformId transformId,
                                     std::vector<std::string> modelSettings, std::vector<std::string> transformSettings,
                                     std::string sessionId, size_t searchSessionIndex, int endStatus,
                                     size_t sessionDuration, std::vector<InteractiveSearchAction> actions,
                                     size_t userId = 0_z);

 
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

  /*!
   * This processes input queries that come from users, generates results and sends them back
   */
  std::vector<GameSessionQueryResult> SubmitUserQueriesWithResults(DataId data_ID,
                                                                   std::vector<GameSessionInputQuery> inputQueries,
                                                                   UserDataSourceId origin = UserDataSourceId::cPublic);

  void SubmitUserDataNativeQueries(std::vector<std::tuple<size_t, std::string, std::string>>& queries);

  const Image* GetRandomImage() const;
  std::tuple<const Image*, bool, size_t> GetCouplingImage() const;
  std::tuple<const Image*, bool, size_t> GetCoupledImagesNative() const;

  std::vector<const Image*> GetRandomImageSequence(size_t seqLength) const;

  NearKeywordsResponse GetNearKeywords(DataId data_ID, const std::string& prefix, size_t numResults,
                                       bool withExampleImages);
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
    KeywordsContainer* pkws{nullptr};

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
      const InputDataTransformSettings& aggSettings, float elapsedTime, size_t imageId = SIZE_T_ERROR_VALUE);

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
    KeywordsContainer* pkws{nullptr};

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
