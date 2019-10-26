#pragma once

#include <assert.h>

#include <limits>
#include <string>
using namespace std::string_literals;

#include <stack>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <iostream>

#include <fstream>
#include <vector>
#include <cstdint>
#include <array>
#include <filesystem>

#include <chrono>
#include <unordered_map>
#include <sstream>
#include <random>
#include <queue>
#include <functional>

#include <map>
#include <set>
#include <locale>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>

#include "config.h"

#include "common.h"
#include "utility.h"
#include "Database.h"
#include "FileParser.h"
#include "Image.hpp"
#include "KeywordsContainer.h"

#include "GridTest.h"

#include "transformations.h"
#include "ranking_models.h"




class ImageRanker
{
  // Structures
public:
  //! ImageRanker modes
  enum class eMode
  {
    cFullAnalytical = 0,
    cCollector = 1,
    cSearchTool = 2
  };

  
  // Methods
public:
  ImageRanker() = delete;

  //! Constructor with data from files with presoftmax file
  ImageRanker(
    const std::string& imagesPath,
    const std::vector<KeywordsFileRef>& keywordsFileRefs,
    const std::vector<ScoringDataFileRef>& imageScoringFileRefs,
    const std::vector<ScoringDataFileRef>& imageSoftmaxScoringFileRefs = std::vector<ScoringDataFileRef>(),
    const std::vector<ScoringDataFileRef>& deepFeaturesFileRefs = std::vector<ScoringDataFileRef>(),
    const std::string& imageToIdMapFilepath = ""s,
    size_t idOffset = 1ULL,
    eMode mode = DEFAULT_MODE,
    const std::vector<KeywordsFileRef>& imageToIdMapFilepaths = std::vector<KeywordsFileRef>()
  );

  ~ImageRanker() noexcept = default;
  
  Keyword* GetKeywordPtr(eKeywordsDataType kwType, const std::string& wordString);

  /*!
   * Initializes IR with current settings
   * 
   * \return True on success
   */
  bool Initialize();
  bool Reinitialize();
  
  eMode GetMode() const;
  void SetMode(eMode value);
  size_t GetIdOffset() const;
  void SetIdOffset(size_t value);
  const std::string& GetImagesPath() const;
  void SetImagesPath(const std::string& path);
  const std::string& GetScoreDataFilepath() const;
  void SetScoreDataFilepath(const std::string& path);
  const std::string& GetKeywordsFilepath() const;
  void SetKeywordsFilepath(const std::string& path);
  const std::string& GetSoftmaxDataFilepath() const;
  void SetSoftmaxDataFilepath(const std::string& path);
  const std::string& GetDeepFeaturesFilepath() const;
  void SetDeepFeaturesFilepath(const std::string& path);
  const std::string& GetImageToIdMapFilepath() const;
  void SetImageToIdMapFilepath(const std::string& path);

  const FileParser* GetFileParser() const;
  FileParser* GetFileParser();
  
  void ClearData();

  std::tuple<
    KeywordsGeneralStatsTuple, 
    ScoringsGeneralStatsTuple, 
    AnnotatorDataGeneralStatsTuple,
    RankerDataGeneralStatsTuple
  >
  GetGeneralStatistics(KwScoringDataId kwScDataId, DataSourceTypeId dataSourceType) const;

  KeywordsGeneralStatsTuple GetGeneralKeywordsStatistics(KwScoringDataId kwScDataId, DataSourceTypeId dataSourceType) const;
  ScoringsGeneralStatsTuple GetGeneralScoringStatistics(KwScoringDataId kwScDataId, DataSourceTypeId dataSourceType) const;
  AnnotatorDataGeneralStatsTuple GetGeneralAnnotatorDataStatistics(KwScoringDataId kwScDataId, DataSourceTypeId dataSourceType) const;
  RankerDataGeneralStatsTuple GetGeneralRankerDataStatistics(KwScoringDataId kwScDataId, DataSourceTypeId dataSourceType) const;


  std::string ExportDataFile(KwScoringDataId kwScDataId, eExportFileTypeId fileType, const std::string& outputFilepath) const;

  bool ExportUserAnnotatorData(KwScoringDataId kwScDataId, DataSourceTypeId dataSource, const std::string& outputFilepath) const;
  bool ExportNormalizedScores(KwScoringDataId kwScDataId, const std::string& outputFilepath) const;
  bool ExportUserAnnotatorNumHits(KwScoringDataId kwScDataId, DataSourceTypeId dataSource, const std::string& outputFilepath) const;


  size_t MapIdToVectorIndex(size_t id) const;
  KeywordsContainer* GetCorrectKwContainerPtr(KwScoringDataId kwScDataId) const;

  std::tuple<
    std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>>,
    std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>>,
    std::vector<std::pair<size_t, std::string>>
  >
  GetImageKeywordsForInteractiveSearch(
    size_t imageId, size_t numResults, KwScoringDataId kwScDataId,
    bool withExampleImages
  );

  void SubmitInteractiveSearchSubmit(
    KwScoringDataId kwScDataId,
    InteractiveSearchOrigin originType, size_t imageId, RankingModelId modelId, InputDataTransformId transformId,
    std::vector<std::string> modelSettings, std::vector<std::string> transformSettings,
    std::string sessionId, size_t searchSessionIndex, int endStatus, size_t sessionDuration,
    std::vector<InteractiveSearchAction> actions,
    size_t userId = 0_z
  );


  /*!
   * Set how ranker will rank by default
   * 
   * \param aggFn
   * \param rankingModel
   * \param dataSource
   * \param settings
   */
  void SetMainSettings(InputDataTransformId agg, RankingModelId rankingModel, RankingModelSettings settings);

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

  const Keyword* GetKeywordConstPtr(eKeywordsDataType kwDataType, size_t keywordId) const;
  Keyword* GetKeywordPtr(eKeywordsDataType kwDataType, size_t keywordId);

  /*!
   * This processes input queries that come from users, generates results and sends them back
   */
  std::vector<GameSessionQueryResult> SubmitUserQueriesWithResults(
    KwScoringDataId kwScDataId,
    std::vector<GameSessionInputQuery> inputQueries, 
    DataSourceTypeId origin = DataSourceTypeId::cPublic
  );

  void SubmitUserDataNativeQueries(
    std::vector<
      std::tuple<size_t, std::string, std::string>>& queries);


  const Image* GetRandomImage() const;
  std::tuple<const Image*, bool, size_t> GetCouplingImage() const;
  std::tuple<const Image*, bool, size_t> GetCoupledImagesNative() const;
  
  std::vector<const Image*> GetRandomImageSequence(size_t seqLength) const;


  NearKeywordsResponse GetNearKeywords(KwScoringDataId kwScDataId, const std::string& prefix, size_t numResults, bool withExampleImages);
  Keyword* GetKeywordByVectorIndex(KwScoringDataId kwScDataId, size_t index);

  RelevantImagesResponse GetRelevantImages(
    KwScoringDataId kwScDataId,
    const std::vector < std::string>& queriesEncodedPlaintext, size_t numResults,
    InputDataTransformId aggId, RankingModelId modelId,
    const RankingModelSettings& modelSettings, const InputDataTransformSettings& aggSettings,
    size_t imageId = SIZE_T_ERROR_VALUE,
    bool withOccuranceValue = false
  ) const;

  std::pair<uint8_t, uint8_t> GetGridTestProgress() const;



  ChartData RunModelTestWrapper(
    KwScoringDataId kwScDataId,
    InputDataTransformId aggId, RankingModelId modelId, DataSourceTypeId dataSource,
    const SimulatedUserSettings& simulatedUserSettings, const RankingModelSettings& aggModelSettings, 
    const InputDataTransformSettings& netDataTransformSettings,
    size_t expansionSettings
  ) const;


  std::vector<std::vector<UserImgQuery>> DoQueryAndExpansion(KwScoringDataId kwScDataId, const std::vector<std::vector<UserImgQuery>>& origQuery, size_t setting) const;
  std::vector<std::vector<UserImgQuery>> DoQueryOrExpansion(KwScoringDataId kwScDataId, const std::vector<std::vector<UserImgQuery>>& origQuery, size_t setting) const;
  

  std::tuple<UserAccuracyChartData, UserAccuracyChartData> GetStatisticsUserKeywordAccuracy(DataSourceTypeId queriesSource = DataSourceTypeId::cAll) const;


  std::string GetKeywordDescriptionByWordnetId(KwScoringDataId kwScDataId, size_t wordnetId)
  {
    KeywordsContainer* pkws{ nullptr };

    switch (std::get<0>(kwScDataId))
    {
    case eKeywordsDataType::cViret1:
      pkws = _pViretKws;
      break;

    case eKeywordsDataType::cGoogleAI:
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
    KwScoringDataId kwScDataId,
    const std::vector < std::string>& queriesEncodedPlaintext, size_t numResults,
    InputDataTransformId aggId, RankingModelId modelId,
    const RankingModelSettings& modelSettings, const InputDataTransformSettings& aggSettings,
    float elapsedTime,
    size_t imageId = SIZE_T_ERROR_VALUE
  );

#endif

  // ^^^^^^^^^^^^^^^^^^^^^^^
  //////////////////////////
  //    API Methods
  //////////////////////////
  
private:

  SimulatedUser GetSimUserSettings(const SimulatedUserSettings& settings) const;

  void ComputeApproxDocFrequency(size_t aggregationGuid, float treshold);

  size_t GetRandomImageId() const;
 

  std::string GetKeywordByWordnetId(KwScoringDataId kwScDataId, size_t wordnetId) const
  {
    KeywordsContainer* pkws{ nullptr };

    // Save shortcuts
    switch (std::get<0>(kwScDataId))
    {
    case eKeywordsDataType::cViret1:
      pkws = _pViretKws;
      break;

    case eKeywordsDataType::cGoogleAI:
      pkws = _pGoogleKws;
      break;
    }
    return pkws->GetKeywordByWordnetId(wordnetId);
  }

  std::string GetImageFilenameById(size_t imageId) const;
  
  void RunGridTestsFromTo(std::vector<std::pair<TestSettings, ChartData>>* pDest, size_t fromIndex, size_t toIndex);
  
  bool LoadKeywordsFromDatabase(Database::Type type);
  bool LoadImagesFromDatabase(Database::Type type);
  std::vector<std::pair<std::string, float>> GetHighestProbKeywords(KwScoringDataId kwScDataId, size_t imageId, size_t N) const;

  std::vector<std::string> TokenizeAndQuery(std::string_view query) const;
  std::vector<std::string> StringenizeAndQuery(KwScoringDataId kwScDataId, const std::string& query) const;

  size_t GetNumImages() const { return _images.size(); };


  void InitializeGridTests();

  std::vector<UserImgQueryRaw>& GetCachedQueriesRaw(DataSourceTypeId dataSource) const;
  std::vector<UserDataNativeQuery>& GetUserAnnotationNativeQueriesCached() const;
  
  std::vector< std::vector<UserImgQuery>>& GetCachedQueries(KwScoringDataId kwScDataId, DataSourceTypeId dataSource) const;


  std::vector< std::vector<UserImgQuery>> GetSimulatedQueries(KwScoringDataId kwScDataId, DataSourceTypeId dataSource, const SimulatedUser& pSimUser) const;
  std::vector< std::vector<UserImgQuery>> GetSimulatedQueries(size_t count, const SimulatedUser& pSimUser) const;
  std::vector< std::vector<UserImgQuery>> GetExtendedRealQueries(KwScoringDataId kwScDataId, DataSourceTypeId dataSource, const SimulatedUser& simUser) const;

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

  bool LoadRepresentativeImages(KwScoringDataId kwScDataId, Keyword* pKw);

  void GenerateBestHypernymsForImages();
  void PrintIntActionsCsv() const;


#if TRECVID_MAPPING
  
  std::pair<size_t, size_t> ConvertToTrecvidShotId(size_t ourFrameId);
  std::vector<std::vector<std::pair<std::pair<unsigned int, unsigned int>, bool>>> ParseTrecvidShotReferencesFromDirectory(const std::string& path) const;
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
  std::map<KwScoringDataId, std::string> _imageScoringFileRefs;
  std::map<KwScoringDataId, std::string> _imageSoftmaxScoringFileRefs;
  std::map<KwScoringDataId, std::string> _deepFeaturesFileRefs;
    
  std::map<eKeywordsDataType, KeywordsContainer> _keywordContainers;
  KeywordsContainer* _pViretKws;
  KeywordsContainer* _pGoogleKws;

  std::unordered_map<InputDataTransformId, std::unique_ptr<TransformationFunctionBase>> _transformations;
  std::unordered_map<RankingModelId, std::unique_ptr<RankingModelBase>> _models;

  std::vector<std::unique_ptr<Image>> _images;

  std::vector<float> _indexKwFrequency;

  mutable std::map<KwScoringDataId, size_t> _stat_minLabels;
  mutable std::map<KwScoringDataId, size_t> _stat_maxLabels;
  mutable std::map<KwScoringDataId, float> _stat_avgLabels;
  mutable std::map<KwScoringDataId, float> _stat_medianLabels;
  mutable std::map<KwScoringDataId, float> _stat_labelHit;
  


#if TRECVID_MAPPING
  std::vector<std::vector<std::pair<std::pair<unsigned int, unsigned int>, bool>>> _trecvidShotReferenceMap;
  std::vector<std::pair<size_t, size_t>> _tvDroppedShots;
#endif

};
