#pragma once

#include <assert.h>

#include <limits>
#include <string>
using namespace std::string_literals;

#include <cmath>
#include <fstream>
#include <iostream>
#include <stack>
#include <stdexcept>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include <chrono>
#include <functional>
#include <queue>
#include <random>
#include <sstream>

#include <unordered_map>

#include <atomic>
#include <chrono>
#include <iomanip>
#include <locale>
#include <map>
#include <set>
#include <thread>

#include "config.h"

#include "Database.h"
#include "FileParser.h"
#include "Image.hpp"
#include "KeywordsContainer.h"
#include "common.h"
#include "utility.h"

#include "GridTest.h"

#include "ranking_models.h"
#include "transformations.h"

class ImageRanker
{
  /****************************
   * Subtypes
   ****************************/
 public:
  /**
   * Config structure needed for instantiation of ImageRanker.
   */
  struct Config
  {
    std::vector<ViretDataPack> dataset_packs;

    std::vector<ViretDataPack> VIRET_packs;
    std::vector<GoogleDataPack> Google_packs;
    std::vector<BowDataPack> BoW_packs;
  };

  /**
   * ImageRanker modes of operation.
   */
  enum class eMode
  {
    cFullAnalytical = 0,
    cCollector = 1,
    cSearchTool = 2
  };
  
  /****************************
   * Methods
   ****************************/
 public:
  ImageRanker() = delete;

  bool Initialize();

  const FileParser* GetFileParser() const
  {
    return &_fileParser;
  }

  // =====================================
  //  NOT REFACTORED CODE BELOW
  // =====================================

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

  RelevantImagesResponse GetRelevantImages(DataId data_ID,
                                           const std::vector<std::string>& queriesEncodedPlaintext, size_t numResults,
                                           InputDataTransformId aggId, RankingModelId modelId,
                                           const RankingModelSettings& modelSettings,
                                           const InputDataTransformSettings& aggSettings,
                                           size_t imageId = SIZE_T_ERROR_VALUE, bool withOccuranceValue = false) const;

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

  /*!
   * Initializes ImageRanker for working in Collector app
   *
   * \return True on success
   */
  bool InitializeCollectorMode();

  /*!
   * Initializes ImageRanker for working as search tool with reduced memory usage
   *
   * \return True on success
   */
  bool InitializeSearchToolMode();

  /*!
   * Initializes ImageRanker for working in IRApp
   *
   * \return True on success
   */
  bool InitializeFullMode();

  /*!
   *  Parses binary file containing raw ranking data
   *
   * \return  Hash map with all Image instances loaded
   */
  std::map<size_t, std::unique_ptr<Image>> ParseRawNetRankingBinFile();
  std::map<size_t, std::unique_ptr<Image>> LowMem_ParseRawNetRankingBinFile();

  /*!
   * Parses Softmax binary file into all \ref Image instances we're now holding
   *
   * \return
   */
  bool ParseSoftmaxBinFile();

  /*!
   * Parses Little Endian integer from provided buffer starting at specified index
   *
   * \return  Correct integer representation.
   */
  int32_t ParseIntegerLE(const std::byte* pFirstByte) const;

  /*!
   * Parses Little Endian float from provided buffer starting at specified index
   *
   * \return  Correct float representation.
   */
  float ParseFloatLE(const std::byte* pFirstByte) const;

  /*!
   * Returns true if no specific image filename mapping file provided
   *
   * \return True if default order of images in directory should be used
   */
  bool UseDefaultImageFilenames() const { return (_imageToIdMapFilepath.empty() ? true : false); };

  /*!
   * Gets aggregation instance if found
   *
   * \param id
   * \return
   */
  TransformationFunctionBase* GetAggregationById(InputDataTransformId id) const;

  /*!
   * Gets ranking model instance if found
   *
   * \param id
   * \return
   */
  RankingModelBase* GetRankingModelById(RankingModelId id) const;

  /*!
   * Gets list of image filenames we're working with
   *
   * \return
   */
  std::vector<std::string> GetImageFilenames() const;
  std::vector<std::string> GetImageFilenamesTrecvid() const;

  bool LoadRepresentativeImages(DataId data_ID, Keyword* pKw);

  void GenerateBestHypernymsForImages();
  void PrintIntActionsCsv() const;

#if TRECVID_MAPPING

  std::pair<size_t, size_t> ConvertToTrecvidShotId(size_t ourFrameId);
  std::vector<std::vector<std::pair<std::pair<unsigned int, unsigned int>, bool>>>
  ParseTrecvidShotReferencesFromDirectory(const std::string& path) const;
  std::vector<std::pair<size_t, size_t>> ParseTrecvidDroppedShotsFile(const std::string& filepath) const;
  void ResetTrecvidShotMap();

#endif

  // Attributes
 private:
  FileParser _fileParser;
  Database _primaryDb;
  Database _secondaryDb;

  eMode _mode;
  size_t _imageIdStride;
  std::string _imageToIdMapFilepath;
  std::string _imagesPath;
  std::map<DataId, std::string> _imageScoringFileRefs;
  std::map<DataId, std::string> _imageSoftmaxScoringFileRefs;
  std::map<DataId, std::string> _deepFeaturesFileRefs;

  std::map<eVocabularyId, KeywordsContainer> _keywordContainers;
  KeywordsContainer* _pViretKws;
  KeywordsContainer* _pGoogleKws;

  std::unordered_map<InputDataTransformId, std::unique_ptr<TransformationFunctionBase>> _transformations;
  std::unordered_map<RankingModelId, std::unique_ptr<RankingModelBase>> _models;

  std::vector<std::unique_ptr<Image>> _images;

  std::vector<float> _indexKwFrequency;

  mutable std::map<DataId, size_t> _stat_minLabels;
  mutable std::map<DataId, size_t> _stat_maxLabels;
  mutable std::map<DataId, float> _stat_avgLabels;
  mutable std::map<DataId, float> _stat_medianLabels;
  mutable std::map<DataId, float> _stat_labelHit;

#if TRECVID_MAPPING
  std::vector<std::vector<std::pair<std::pair<unsigned int, unsigned int>, bool>>> _trecvidShotReferenceMap;
  std::vector<std::pair<size_t, size_t>> _tvDroppedShots;
#endif
};
