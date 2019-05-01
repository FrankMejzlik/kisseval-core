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
#include <sstream>
#include <queue>
#include <functional>

#include <map>
#include <set>
#include <locale>
#include <thread>
#include <atomic>
#include <chrono>

#include "config.h"

#include "utility.h"
#include "Database.h"
#include "KeywordsContainer.h"


class ImageRanker;
class GridTest;

class RankingModel
{
public:

};

class BooleanBucketModel
{
public:
  static float m_trueTresholdFrom;
  static float m_trueTresholdTo;
  static float m_trueTresholdStep;
  static std::vector<float> m_trueTresholds;

  static std::vector<uint8_t> m_inBucketOrders;
};

class BooleanViretModel
{
public:
  static float m_trueTresholdFrom;
  static float m_trueTresholdTo;
  static float m_trueTresholdStep;
  static std::vector<float> m_trueTresholds;

  static std::vector<uint8_t> m_queryOperations;
};



struct Image
{
  Image() = default;

  Image(
    size_t id, 
    std::string&& filename, 
    std::vector<float>&& rawNetRanking,
    float min, float max,
    float mean, float variance
  ) :
    m_imageId(id),
    m_filename(std::move(filename)),
    m_rawNetRanking(std::move(rawNetRanking)),
    m_rawNetRankingSorted(),
    m_min(min), m_max(max),
    m_mean(mean), m_variance(variance)
  {

    // Create sorted array
    size_t i{0ULL};
    for (auto&& img : m_rawNetRanking)
    {
      m_rawNetRankingSorted.emplace_back(std::pair(static_cast<uint32_t>(i), img));

      ++i;
    }
    

    // Sort it
    std::sort(
      m_rawNetRankingSorted.begin(), m_rawNetRankingSorted.end(),
      [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) -> bool
      {
        return a.second > b.second;
      }
    );


  }

  size_t m_imageId;
  std::string m_filename;

  //! Raw vector as it came out of neural network
  std::vector<float> m_rawNetRanking;

  float m_min;
  float m_max;
  float m_mean;
  float m_variance;

  //! Raw vector as it came out of neural network but SORTED
  std::vector<std::pair<uint32_t, float>> m_rawNetRankingSorted;
   

  //! Softmax probability ranking
  std::vector<float> m_softmaxVector;

  //! Softmax probability vector
  std::vector<float> m_softmaxProbAmplified1;
  std::vector<float> m_softmaxProbAmplified2;

  //! Probability vector from custom MinMax Clamp method
  std::vector<float> m_minMaxLinearVector;

  //! Probability vector from custom Boolean Aggregation with treshold
  std::vector<float> m_amplifyProbVector1;
  std::vector<float> m_amplifyProbVector2;
  std::vector<float> m_amplifyProbVector3;

};

class ImageRanker
{
  // Structures
public:
  enum RankingModel
  {
    cBoolean = 0,
    cBooleanBucket = 1,
    cBooleanExtended = 2,
    cViretBase = 3,
    cFuzzyLogic = 4
  };

  enum QueryOrigin
  {
    cDeveloper = 0,
    cPublic = 1,
    cManaged = 2
  };

  enum Aggregation
  {
    cSoftmax = 1,
      cAmplifiedSoftmax1 = 100,
      cAmplifiedSoftmax2 = 101,
      
    cMinMaxLinear = 2,
      cAmplified1 = 200,
      cAmplified2 = 201,
      cAmplified3 = 202,
  };

  enum Mode
  {
    cFull,
    cCollector
  };



  struct QueryResult 
  {
    QueryResult() :
      m_targetImageRank(0ULL)
    {}

    size_t m_targetImageRank;
  };

  using Buffer = std::vector<std::byte>;

  //! This is returned to front end app when some quesries are submited
  //! <SessionID, image filename, user keywords, net <keyword, probability> >
  using GameSessionQueryResult = std::tuple<std::string, std::string, std::vector<std::string>, std::vector<std::pair<std::string, float>>>;
  
  //! Array of those is submited from front-end app game
  using GameSessionInputQuery = std::tuple<std::string, size_t, std::string>;

  using ImageReference = std::pair<size_t, std::string>;

  /*! <wordnetID, keyword, description> */
  using KeywordReferences = std::vector<std::tuple<size_t, std::string, std::string>>;

  /*!
   * Output data for drawing chart
   */
  using ChartData = std::vector <std::pair<uint32_t, uint32_t>>;

  using TestSettings = std::tuple<Aggregation, RankingModel, QueryOrigin, std::vector<std::string>>;

  /*!
   * FORMAT:
   *  0: Boolean:
   *  1: BooleanBucket:
   *    0 => trueTreshold
   *    1 => inBucketSorting
   *  2: BooleanExtended:
   *  3: ViretBase:
   *    0 => ignoreTreshold
   *    1 => rankCalcMethod
   *      0: Multiply & (Add |)
   *      1: Add only
   *  4: FuzzyLogic:
   */
  using ModelSettings = std::vector<std::string>;

  
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
    size_t idOffset = 1ULL
  );

  //! Constructor with data from database
  ImageRanker(
    const std::string& imagesPath
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

  /*!
   * Set how ranker will rank by default
   * 
   * \param aggFn
   * \param rankingModel
   * \param dataSource
   * \param settings
   */
  void SetMainSettings(Aggregation agg, RankingModel rankingModel, ModelSettings settings);

  // const chartData = [
  //   { index: 0, value: 10 },
  //   { index: 1, value: 20 },
  //   { index: 2, value: 30 },
  //   { index: 3, value: 40 },
  //   { index: 4, value: 40.32 },
  //   { index: 5, value: 50.3 },
  //   { index: 6, value: 60.4 }
  // ];
  ImageRanker::ChartData RunModelTest(
    Aggregation aggFn, RankingModel rankingModel, QueryOrigin dataSource, const ModelSettings& settings
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
  std::vector<GameSessionQueryResult> SubmitUserQueriesWithResults(std::vector<GameSessionInputQuery> inputQueries, QueryOrigin origin = QueryOrigin::cPublic);


  ImageReference GetRandomImage() const;
  KeywordReferences GetNearKeywords(const std::string& prefix);

  std::pair<std::vector<ImageReference>, QueryResult> GetRelevantImages(
    const std::string& query, size_t numResults, 
    Aggregation aggFn, RankingModel rankingModel, const ModelSettings& settings,
    size_t imageId = SIZE_T_ERROR_VALUE  
  ) const;


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

  ImageRanker::ChartData RunBooleanCustomModelTest(Aggregation aggFn, QueryOrigin dataSource, const ImageRanker::ModelSettings& settings) const;
  ImageRanker::ChartData RunViretBaseModelTest(Aggregation aggFn, QueryOrigin dataSource, const ImageRanker::ModelSettings& settings) const;
  


  size_t GetRandomImageId() const;
 

  std::pair<std::vector<ImageReference>, QueryResult> GetImageRankingBooleanModel(
const std::string& query, size_t numResults, 
    size_t targetImageId,
    Aggregation aggFn, const ModelSettings& settings
  ) const;
  std::pair<std::vector<ImageReference>, QueryResult> GetImageRankingBooleanCustomModel(
    const std::string& query, size_t numResults, 
    size_t targetImageId,
    Aggregation aggFn , const ModelSettings& settings
  ) const;
  std::pair<std::vector<ImageReference>, QueryResult> GetImageRankingViretBaseModel(
    const std::string& query, size_t numResults, 
    size_t targetImageId,
    Aggregation aggFn, const ModelSettings& settings
  ) const;
  std::pair<std::vector<ImageReference>, QueryResult> GetImageRankingFuzzyLogicModel(
    const std::string& query, size_t numResults, 
    size_t targetImageId,
    Aggregation aggFn, const ModelSettings& settings
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


  void RunGridTestsFromTo(std::vector<std::pair<ImageRanker::TestSettings, ImageRanker::ChartData>>* pDest, size_t fromIndex, size_t toIndex);
  
  bool LoadKeywordsFromDatabase(Database::Type type);
  bool LoadImagesFromDatabase(Database::Type type);
  std::vector<std::pair<std::string, float>> GetHighestProbKeywords(size_t imageId, size_t N) const;

  std::vector<std::string> TokenizeAndQuery(std::string_view query) const;
  std::vector<std::string> StringenizeAndQuery(const std::string& query) const;

  std::unordered_map<size_t, std::pair<size_t, std::string> > ParseKeywordClassesTextFile(std::string_view filepath) const;

  std::unordered_map<size_t, std::pair<size_t, std::string> > ParseHypernymKeywordClassesTextFile(std::string_view filepath) const;

  std::pair< size_t, std::vector< std::vector<std::string>>>& GetCachedQueries(ImageRanker::QueryOrigin dataSource)  const;

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
  std::map<size_t, Image> ParseRawNetRankingBinFile();
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


  std::vector<std::string> GetImageFilenames() const;
  std::vector<std::string> GetImageFilenamesFromDirectoryStructure() const;

  //////////////////////////
  // uncertain stuff
  void CalculateMinMaxClampAgg();

  // uncertain stuff
  //////////////////////////

private:
  Database _primaryDb;
  Database _secondaryDb;

  bool _isReinitNeeded;
  Mode _mode;

  //! Aggregation that will be used mainly
  Aggregation _mainAggregation; 

  //! Ranking model that will be used mainly
  RankingModel _mainRankingModel;

  //! Model settings that will be used mainly
  ModelSettings _mainSettings;

  size_t _imageIdStride;

  std::string _imagesPath;
  std::string _rawNetRankingFilepath;

  std::string _softmaxFilepath;
  std::string _deepFeaturesFilepath;

  std::string _imageToIdMap;

  KeywordsContainer _keywords;
  std::map<size_t, Image> _images;


};


class GridTest
{
public:
  static std::atomic<size_t> numCompletedTests;

  static void ProgressCallback()
  {
    GridTest::numCompletedTests.operator++();
  }

  static std::pair<uint8_t, uint8_t> GetGridTestProgress()
  {
    return std::pair((uint8_t)GridTest::numCompletedTests, (uint8_t)m_testSettings.size());
  }

  static void ReportTestProgress()
  {
    while (true)
    {
      if (numCompletedTests >= m_testSettings.size())
      {
        break;
      }

      LOG("Test progress is "s + std::to_string(GridTest::numCompletedTests) + "/"s + std::to_string(m_testSettings.size()));
      
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
    
  }

  

  static std::vector<ImageRanker::Aggregation> m_aggregations;
  static std::vector<ImageRanker::QueryOrigin> m_queryOrigins;
  static std::vector<ImageRanker::RankingModel> m_rankingModels;


  static std::vector<ImageRanker::TestSettings> m_testSettings;
};