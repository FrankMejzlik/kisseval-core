#pragma once

#include <assert.h>

#include <limits>
#include <string>
using namespace std::string_literals;

#include <fstream>
#include <cmath>
#include <stdexcept>
#include <iostream>

#include <fstream>
#include <vector>
#include <cstdint>
#include <array>

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

#include "config.h"

#include "common.h"
#include "utility.h"
#include "Database.h"
#include "Image.hpp"
#include "KeywordsContainer.h"

#include <iomanip>

#include "GridTest.h"

#include "transformations.h"
#include "aggregation_models.h"


class ImageRanker
{
  // Structures
public:
  //! ImageRanker modes
  enum class Mode
  {
    cFull,
    cCollector
  };

  
  // Methods
public:
  ImageRanker() = delete;

  //! Constructor with data from files with presoftmax file
  ImageRanker(
    const std::string& imagesPath,
    const std::string& rawNetRankingFilepath,
    const std::string& keywordClassesFilepath,
    const std::string& softmaxFilepath = ""s,
    const std::string& deepFeaturesFilepath = ""s,
    const std::string& imageToIdMapFilepath = ""s,
    size_t idOffset = 1ULL,
    Mode mode = DEFAULT_MODE
  );

  ~ImageRanker() noexcept = default;


  //////////////////////////
  //    API Methods
  //////////////////////////
  // vvvvvvvvvvvvvvvvvvvvvvv

  /*!
   * Initializes IR with current settings
   * 
   * \return True on success
   */
  bool Initialize();
  bool Reinitialize();
  void Clear();
  void SetMode(Mode value);

  

  std::pair<std::vector<std::tuple<size_t, std::string, float>>, std::vector<std::tuple<size_t, std::string, float>>> GetImageKeywordsForInteractiveSearch(size_t imageId, size_t numResults);


  std::pair<
    std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>>, 
    std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>>
  > GetImageKeywordsForInteractiveSearchWithExampleImages(size_t imageId, size_t numResults);

  void SubmitInteractiveSearchSubmit(
    InteractiveSearchOrigin originType, size_t imageId, RankingModelId modelId, NetDataTransformation transformId,
    std::vector<std::string> modelSettings, std::vector<std::string> transformSettings,
    std::string sessionId, size_t searchSessionIndex, int endStatus, size_t sessionDuration,
    std::vector<InteractiveSearchAction> actions,
    size_t userId = 0_z
  );

  // So front end can display options dynamically
  // \todo Implement.
  void GetActiveAggregations() const;
  void GetActiveRankingModels() const;

  /*!
   * Set how ranker will rank by default
   * 
   * \param aggFn
   * \param rankingModel
   * \param dataSource
   * \param settings
   */
  void SetMainSettings(NetDataTransformation agg, RankingModelId rankingModel, AggModelSettings settings);

  std::vector<std::pair<TestSettings, ChartData>> RunGridTest(const std::vector<TestSettings>& testSettings);


  void RecalculateHypernymsInVectorUsingSum(AggregationVector& binVectorRef);
  void RecalculateHypernymsInVectorUsingMax(AggregationVector& binVectorRef);
  /*!
   * Gets all data about image with provided ID
   *
   * \param imageId
   * \return
   */
  const Image* GetImageDataById(size_t imageId) const;


  /*!
   * This processes input queries that come from users, generates results and sends them back
   */
  std::vector<GameSessionQueryResult> SubmitUserQueriesWithResults(std::vector<GameSessionInputQuery> inputQueries, QueryOriginId origin = QueryOriginId::cPublic);


  ImageReference GetRandomImage() const;

  KeywordReferences GetNearKeywords(const std::string& prefix);
  std::vector<Keyword*> GetNearKeywordsWithImages(const std::string& prefix);
  KeywordData GetKeywordByVectorIndex(size_t index);


  std::pair<std::vector<ImageReference>, QueryResult> GetRelevantImagesWrapper(
    const std::string& queryEncodedPlaintext, size_t numResults,
    NetDataTransformation aggId, RankingModelId modelId,
    const AggModelSettings& modelSettings, const NetDataTransformSettings& aggSettings,
    size_t imageId = SIZE_T_ERROR_VALUE
  ) const;

  std::tuple<std::vector<ImageReference>, std::vector<std::tuple<size_t, std::string, float>>, QueryResult> GetRelevantImagesWithSuggestedWrapper(
    const std::string& queryEncodedPlaintext, size_t numResults,
    NetDataTransformation aggId, RankingModelId modelId,
    const AggModelSettings& modelSettings, const NetDataTransformSettings& aggSettings,
    size_t imageId
  ) const;

  std::pair<uint8_t, uint8_t> GetGridTestProgress() const;



  ChartData RunModelTestWrapper(
    NetDataTransformation aggId, RankingModelId modelId, QueryOriginId dataSource,
    const SimulatedUserSettings& simulatedUserSettings, const AggModelSettings& aggModelSettings, const NetDataTransformSettings& netDataTransformSettings
  ) const;


  

  std::tuple<UserAccuracyChartData, UserAccuracyChartData> GetStatisticsUserKeywordAccuracy(QueryOriginId queriesSource = QueryOriginId::cAll) const;


  std::string GetKeywordDescriptionByWordnetId(size_t wordnetId)
  {
    return _keywords.GetKeywordDescriptionByWordnetId(wordnetId);
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^
  //////////////////////////
  //    API Methods
  //////////////////////////
  
private:

  SimulatedUser GetSimUserSettings(const SimulatedUserSettings& settings) const;

  void ComputeApproxDocFrequency(size_t aggregationGuid, float treshold);

  size_t GetRandomImageId() const;
 

  std::string GetKeywordByWordnetId(size_t wordnetId) const
  {
    return _keywords.GetKeywordByWordnetId(wordnetId);
  }



  std::string GetImageFilenameById(size_t imageId) const;
  
  void RunGridTestsFromTo(std::vector<std::pair<TestSettings, ChartData>>* pDest, size_t fromIndex, size_t toIndex);
  
  bool LoadKeywordsFromDatabase(Database::Type type);
  bool LoadImagesFromDatabase(Database::Type type);
  std::vector<std::pair<std::string, float>> GetHighestProbKeywords(size_t imageId, size_t N) const;

  std::vector<std::string> TokenizeAndQuery(std::string_view query) const;
  std::vector<std::string> StringenizeAndQuery(const std::string& query) const;

  std::unordered_map<size_t, std::pair<size_t, std::string> > ParseKeywordClassesTextFile(std::string_view filepath) const;

  std::unordered_map<size_t, std::pair<size_t, std::string> > ParseHypernymKeywordClassesTextFile(std::string_view filepath) const;

  size_t GetNumImages() const { return _images.size(); };

  const std::vector<float>& GetMainRankingVector(const Image& image) const;
  std::vector<float>& GetMainRankingVector(Image& image);

  void InitializeGridTests();




  

  std::vector<UserImgQueryRaw>& GetCachedQueriesRaw(QueryOriginId dataSource) const;
  
  std::vector<UserImgQuery>& GetCachedQueries(QueryOriginId dataSource) const;


  std::vector<UserImgQuery> GetSimulatedQueries(QueryOriginId dataSource, const SimulatedUser& pSimUser) const;
  std::vector<UserImgQuery> GetSimulatedQueries(size_t count, const SimulatedUser& pSimUser) const;

  UserImgQuery GetSimulatedQueryForImage(size_t imageId, const SimulatedUser& simUser) const;

  std::string EncodeAndQuery(const std::string& query) const;

  /*!
   * Initializes ImageRanker for working in Collector app
   * 
   * \return True on success
   */
  bool InitializeCollectorMode();

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
  std::unordered_map<size_t, std::unique_ptr<Image>> ParseRawNetRankingBinFile();

  /*!
   * Parses Softmax binary file into all \ref Image instances we're now holding
   * 
   * \return 
   */
  bool ParseSoftmaxBinFile();
  
  /*!
  * Loads bytes from specified file into buffer
  *
  * \param filepath  Path to file to load.
  * \return New vector byte buffer.
  */
  std::vector<std::byte> LoadFileToBuffer(std::string_view filepath) const;

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
  bool UseDefaultImageFilenames() const { return (_imageToIdMap.empty() ? true : false); };

  /*!
   * Gets aggregation instance if found
   * 
   * \param id
   * \return 
   */
  TransformationFunctionBase* GetAggregationById(NetDataTransformation id) const;
  
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

  /*!
   * Gets list of image filenames we're working with from dir structure
   *
   * \return
   */
  std::vector<std::string> GetImageFilenamesFromDirectoryStructure() const;

  bool LoadRepresentativeImages(Keyword* pKw) const;

  void GenerateBestHypernymsForImages();
  void PrintIntActionsCsv() const;

#if PUSH_DATA_TO_DB

  bool PushDataToDatabase();
  bool PushKeywordsToDatabase();
  bool PushImagesToDatabase();

#endif // PUSH_DATA_TO_DB


  // Attributes
private:
  Database _primaryDb;
  Database _secondaryDb;

  std::mt19937_64 _generator;
  std::uniform_real_distribution<double> _uniformRealDistribution;
 

  bool _isReinitNeeded;
  Mode _mode;

  //! Aggregation that will be used mainly
  NetDataTransformation _mainAggregation; 

  //! Ranking model that will be used mainly
  RankingModelId _mainRankingModel;

  //! Model settings that will be used mainly
  AggModelSettings _mainSettings;

  size_t _imageIdStride;

  std::string _imagesPath;
  std::string _rawNetRankingFilepath;

  std::string _softmaxFilepath;
  std::string _deepFeaturesFilepath;

  std::string _imageToIdMap;

  KeywordsContainer _keywords;
  std::unordered_map<size_t, std::unique_ptr<Image>> _images;

  std::vector<float> m_indexKwFrequency;


  std::unordered_map<NetDataTransformation, std::unique_ptr<TransformationFunctionBase>> _aggregations;
  std::unordered_map<RankingModelId, std::unique_ptr<RankingModelBase>> _models;
};


