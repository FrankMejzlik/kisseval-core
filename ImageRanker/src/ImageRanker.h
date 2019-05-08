#pragma once

#include <assert.h>

#include <limits>
#include <string>
using namespace std::string_literals;

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

#include "GridTest.h"

#include "aggregations.h"
#include "ranking_models.h"


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

  // So front end can display options dynamically
  // \todo Implement.
  void GetAggregations() const;
  void GetRankingModels() const;

  /*!
   * Set how ranker will rank by default
   * 
   * \param aggFn
   * \param rankingModel
   * \param dataSource
   * \param settings
   */
  void SetMainSettings(AggregationId agg, RankingModelId rankingModel, ModelSettings settings);

  // const chartData = [
  //   { index: 0, value: 10 },
  //   { index: 1, value: 20 },
  //   { index: 2, value: 30 },
  //   { index: 3, value: 40 },
  //   { index: 4, value: 40.32 },
  //   { index: 5, value: 50.3 },
  //   { index: 6, value: 60.4 }
  // ];
  ChartData RunModelTest(
    AggregationId aggFn, RankingModelId rankingModel, QueryOriginId dataSource, const ModelSettings& settings, const AggregationSettings& aggSettings
  ) const;

  std::vector<ChartData> RunModelTests(const std::vector<TestSettings>& testSettings) const;

  std::vector<std::pair<TestSettings, ChartData>> RunGridTest(const std::vector<TestSettings>& testSettings);


  std::string GetKeywordByVectorIndex(size_t index) const
  {
    return _keywords.GetKeywordByVectorIndex(index);
  }

  /*!
   * Gets all data about image with provided ID
   *
   * \param imageId
   * \return
   */
    Image GetImageDataById(size_t imageId) const;


  /*!
   * This processes input queries that come from users, generates results and sends them back
   */
  std::vector<GameSessionQueryResult> SubmitUserQueriesWithResults(std::vector<GameSessionInputQuery> inputQueries, QueryOriginId origin = QueryOriginId::cPublic);


  ImageReference GetRandomImage() const;
  KeywordReferences GetNearKeywords(const std::string& prefix);

  std::pair<std::vector<ImageReference>, QueryResult> GetRelevantImages(
    const std::string& query, size_t numResults, 
    AggregationId aggFn, RankingModelId rankingModel, const ModelSettings& settings, const AggregationSettings& aggSettings,
    size_t imageId = SIZE_T_ERROR_VALUE  
  ) const;


  std::pair<std::vector<ImageReference>, QueryResult> GetRelevantImagesWrapper(
    const std::string& queryEncodedPlaintext, size_t numResults,
    AggregationId aggId, RankingModelId modelId, 
    const ModelSettings& modelSettings, const AggregationSettings& aggSettings,
    size_t imageId = SIZE_T_ERROR_VALUE
  ) const;


  std::pair<std::vector<ImageReference>, QueryResult> GetRelevantImagesPlainQuery(
    const std::string& query, size_t numResults,
    AggregationId aggFn, RankingModelId rankingModel, 
    const ModelSettings& settings, const AggregationSettings& aggSettings,
    size_t imageId = SIZE_T_ERROR_VALUE
  ) const
  {
    return GetRelevantImages(EncodeAndQuery(query), numResults, aggFn, rankingModel, settings, aggSettings, imageId);
  }


  std::pair<uint8_t, uint8_t> GetGridTestProgress() const;

  // ^^^^^^^^^^^^^^^^^^^^^^^
  //////////////////////////
  //    API Methods
  //////////////////////////
  
private:
#if PUSH_DATA_TO_DB
  bool PushDataToDatabase();
  bool PushKeywordsToDatabase();
  bool PushImagesToDatabase();
#endif

  ChartData RunBooleanCustomModelTest(AggregationId aggFn, QueryOriginId dataSource, const ModelSettings& settings, const AggregationSettings& aggSettings) const;
  ChartData RunViretBaseModelTest(AggregationId aggFn, QueryOriginId dataSource, const ModelSettings& settings, const AggregationSettings& aggSettings) const;
  


  size_t GetRandomImageId() const;
 

  std::pair<std::vector<ImageReference>, QueryResult> GetImageRankingBooleanCustomModel(
    const std::string& query, size_t numResults, 
    size_t targetImageId,
    AggregationId aggFn , const ModelSettings& settings, const AggregationSettings& aggSettings
  ) const;
  std::pair<std::vector<ImageReference>, QueryResult> GetImageRankingViretBaseModel(
    const std::string& query, size_t numResults, 
    size_t targetImageId,
    AggregationId aggFn, const ModelSettings& settings, const AggregationSettings& aggSettings
  ) const;


  std::string GetKeywordByWordnetId(size_t wordnetId) const
  {
    return _keywords.GetKeywordByWordnetId(wordnetId);
  }

  std::string GetKeywordDescriptionByWordnetId(size_t wordnetId)
  {
    return _keywords.GetKeywordDescriptionByWordnetId(wordnetId);
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

  std::pair< size_t, std::vector< std::vector<std::string>>>& GetCachedQueries(QueryOriginId dataSource)  const;

  std::string EncodeAndQuery(const std::string& query) const;


  size_t GetNumImages() const { return _images.size(); };


  const std::vector<float>& GetMainRankingVector(const Image& image) const;
  std::vector<float>& GetMainRankingVector(Image& image);






  void InitializeGridTests();

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
  std::unordered_map<size_t, Image> ParseRawNetRankingBinFile();

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
  AggregationFunctionBase* GetAggregationById(size_t id) const;

  /*!
   * Gets ranking model instance if found
   * 
   * \param id
   * \return 
   */
  RankingModelBase* GetRankingModelById(size_t id) const;

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

private:
  Database _primaryDb;
  Database _secondaryDb;

  bool _isReinitNeeded;
  Mode _mode;

  //! Aggregation that will be used mainly
  AggregationId _mainAggregation; 

  //! Ranking model that will be used mainly
  RankingModelId _mainRankingModel;

  //! Model settings that will be used mainly
  ModelSettings _mainSettings;

  size_t _imageIdStride;

  std::string _imagesPath;
  std::string _rawNetRankingFilepath;

  std::string _softmaxFilepath;
  std::string _deepFeaturesFilepath;

  std::string _imageToIdMap;

  KeywordsContainer _keywords;
  std::unordered_map<size_t, Image> _images;


  std::unordered_map<AggregationId, std::unique_ptr<AggregationFunctionBase>> _aggregations;
  std::unordered_map<RankingModelId, std::unique_ptr<RankingModelBase>> _models;
};


