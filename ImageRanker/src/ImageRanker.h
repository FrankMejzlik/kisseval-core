#pragma once

#include <stdexcept>
#include <iostream>
#include <assert.h>
#include <fstream>
#include <vector>
#include <cstdint>
#include <array>
#include <string>
#include <chrono>
#include <unordered_map>
#include <sstream>
#include <random>
#include <sstream>
#include <queue>

#include <map>
#include <set>
#include <locale>

#include "config.h"

#include "utility.h"
#include "Database.h"
#include "KeywordsContainer.h"

using namespace std::string_literals;

struct Image
{
  Image(): 
    m_imageId(SIZE_T_ERROR_VALUE),
    m_filename(""s)
  {};
  Image(size_t id, std::string&& filename, std::vector<std::pair<size_t, float>>&& probVector):
    m_imageId(id),
    m_filename(std::move(filename)),
    m_probabilityVector(std::move(probVector))
  {}

  Image(size_t id, std::string&& filename, std::vector<std::pair<size_t, float>>&& probVector, std::vector<std::pair<size_t, uint8_t>>&& boolVector) :
    m_imageId(id),
    m_filename(std::move(filename)),
    m_probabilityVector(std::move(probVector)),
    m_booleanProbVector(boolVector)
  {
  }

  size_t m_imageId;
  std::string m_filename;
  std::vector<std::pair<size_t, float>> m_probabilityVector;
  std::vector<std::pair<size_t, uint8_t>> m_booleanProbVector;

  //! Raw vector as it came out of neural network
  std::vector<float> m_rawProbabilityVector;

  //! Probability vector from custom MinMax Clamp method
  std::vector<float> m_minMaxClampAggProbVector;

  //! Probability vector from custom Boolean Aggregation with treshold
  std::vector<float> m_boolAggProbVector;
};


struct ImageData 
{

};

class ImageRanker
{
  // Structures
public:
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
  using ImageData = Image;


  //!
  /*! <wordnetID, keyword, description> */
  using KeywordReference = std::vector<std::tuple<size_t, std::string, std::string>>;

  using ChartData = std::vector <std::pair<uint32_t, uint32_t>>;

  enum RankingModel 
  {
    cBoolean,
    cBooleanCustom,
    cBooleanExtended,
    cViretBase,
    cFuzzyLogic
  };

  enum QueryOrigin
  {
    cDeveloper,
    cPublic
  };

  enum AggregationFunction
  {
    cSoftmax,
    cMinMaxClamp
  };

  // Methods
public:
  ImageRanker() = delete;

  //! Constructor with data from files
  ImageRanker(
    std::string_view imagesPath,
    std::string_view probabilityVectorFilepath,
    std::string_view deepFeaturesFilepath,
    std::string_view keywordClassesFilepath,
    std::string_view imagesListFilepath,
    size_t columnIndexFilename,
    size_t imageListFileLineLength,
    size_t numRows,
    size_t idOffset
  );

  //! Constructor with data from files with presoftmax file
  ImageRanker(
    std::string_view imagesPath,
    std::string_view probabilityVectorFilepath,
    std::string_view rawProbabilityVectorFilepath,
    std::string_view deepFeaturesFilepath,
    std::string_view keywordClassesFilepath,
    std::string_view imagesListFilepath,
    size_t columnIndexFilename,
    size_t imageListFileLineLength,
    size_t numRows,
    size_t idOffset
  );

  //! Constructor with data from database
  ImageRanker(
    std::string_view imagesPath
  );

  ~ImageRanker() noexcept = default;

  // Testing
# if 1 

  void TEST_GetVectorKeywords(size_t wordnetId)
  {
    auto keywords = _keywords.GetVectorKeywords(wordnetId);

    std::cout << "===========" << std::endl;
    std::cout << "KEYWORD: " << GetKeywordByWordnetId(wordnetId) << std::endl;

    for (auto&& keywordId : keywords)
    {
      std::cout << GetKeywordByWordnetId(keywordId) << std::endl;
    }
  }


  void TEST_GetCanonicalQuery(const std::string& query)
  {
    CnfFormula fml = _keywords.GetCanonicalQuery(query);

    std::cout << "===========" << std::endl;
    std::cout << "QUERY: " << query << std::endl;
    std::cout << _keywords.StringifyCnfFormula(fml) << std::endl;

  }

#endif


  //////////////////////////
  //    API Methods
  //////////////////////////
  // vvvvvvvvvvvvvvvvvvvvvvv

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
    AggregationFunction aggFn, RankingModel rankingModel, QueryOrigin dataSource,
    std::vector<std::string> settings
  );


  /*!
   * Gets all data about image with provided ID
   * 
   * \param imageId
   * \return 
   */
  ImageData GetImageDataById(size_t imageId) const;
  std::string GetKeywordByVectorIndex(size_t index) const
  {
    return _keywords.GetKeywordByVectorIndex(index);
  }

  /*!
   * This processes input queries that come from users, generates results and sends them back
   */
  std::vector<GameSessionQueryResult> SubmitUserQueriesWithResults(std::vector<GameSessionInputQuery> inputQueries, QueryOrigin origin = QueryOrigin::cPublic);


  ImageReference GetRandomImage() const;
  KeywordReference GetNearKeywords(const std::string& prefix);

  std::pair<std::vector<ImageReference>, QueryResult> GetRelevantImages(
    const std::string& query, size_t numResults = NUM_IMAGES_PER_PAGE, 
    AggregationFunction aggFn = DEFAULT_AGG_FUNCTION, RankingModel rankingModel = DEFAULT_RANKING_MODEL, std::vector<std::string> settings = DEFAULT_MODEL_SETTINGS,
    size_t imageId = SIZE_T_ERROR_VALUE  
  ) const;


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

  ImageRanker::ChartData RunBooleanCustomModelTest(AggregationFunction aggFn, QueryOrigin dataSource, std::vector<std::string> settings);
  ImageRanker::ChartData RunViretBaseModelTest(AggregationFunction aggFn, QueryOrigin dataSource, std::vector<std::string> settings);
  


  size_t GetRandomImageId() const;
  

  size_t GetNumImages() const
  {
    return _images.size();
  }

  std::pair<std::vector<ImageReference>, QueryResult> GetImageRankingBooleanModel(
const std::string& query, size_t numResults = 0ULL, 
    size_t targetImageId = SIZE_T_ERROR_VALUE,
    AggregationFunction aggFn = DEFAULT_AGG_FUNCTION, std::vector<std::string> settings = DEFAULT_MODEL_SETTINGS
  ) const;
  std::pair<std::vector<ImageReference>, QueryResult> GetImageRankingBooleanCustomModel(
    const std::string& query, size_t numResults = 0ULL, 
    size_t targetImageId = SIZE_T_ERROR_VALUE,
    AggregationFunction aggFn = DEFAULT_AGG_FUNCTION, std::vector<std::string> settings = DEFAULT_MODEL_SETTINGS
  ) const;
  std::pair<std::vector<ImageReference>, QueryResult> GetImageRankingViretBaseModel(
    const std::string& query, size_t numResults = 0ULL, 
    size_t targetImageId = SIZE_T_ERROR_VALUE,
    AggregationFunction aggFn = DEFAULT_AGG_FUNCTION, std::vector<std::string> settings = DEFAULT_MODEL_SETTINGS
  ) const;
  std::pair<std::vector<ImageReference>, QueryResult> GetImageRankingFuzzyLogicModel(
    const std::string& query, size_t numResults = 0ULL, 
    size_t targetImageId = SIZE_T_ERROR_VALUE,
    AggregationFunction aggFn = DEFAULT_AGG_FUNCTION, std::vector<std::string> settings = DEFAULT_MODEL_SETTINGS
  ) const;

  std::string GetKeywordByWordnetId(size_t wordnetId)
  {
    return _keywords.GetKeywordByWordnetId(wordnetId);
  }

  std::string GetKeywordDescriptionByWordnetId(size_t wordnetId)
  {
    return _keywords.GetKeywordDescriptionByWordnetId(wordnetId);
  }

  std::string GetImageFilenameById(size_t imageId) const;

  std::string GetImageFilepathByIndex(size_t imgIndex, bool relativePaths = false) const;

  int GetRandomInteger(int from, int to) const;
  bool LoadKeywordsFromDatabase(Database::Type type);
  bool LoadImagesFromDatabase(Database::Type type);
  std::vector<std::pair<std::string, float>> GetHighestProbKeywords(size_t imageId, size_t N) const;

  std::vector<std::string> TokenizeAndQuery(std::string_view query) const;

  std::map<size_t, Image> ParseSoftmaxBinFile(std::string_view filepath, std::string_view imageFilesFilepath) const;
  bool ParseRawProbabilityBinFile(std::string_view filepath, std::string_view imageFilesFilepath);

  std::vector< std::pair< size_t, std::unordered_map<size_t, float> > > ParseSoftmaxBinFileFiltered(std::string_view filepath, float minProbabilty) const;

  std::unordered_map<size_t, std::pair<size_t, std::string> > ParseKeywordClassesTextFile(std::string_view filepath) const;

  std::unordered_map<size_t, std::pair<size_t, std::string> > ParseHypernymKeywordClassesTextFile(std::string_view filepath) const;

  bool CalculateMinMaxClampAgg(Image* pImage, float min, float max, float avg);

  /*!
  * Loads bytes from specified file into buffer
  *
  * \param filepath  Path to file to load.
  * \return New vector byte buffer.
  */
  std::vector<std::byte> LoadFileToBuffer(std::string_view filepath) const;

  /*!
  * Parses Little Endian integer from provided buffer starting at specified index.
  *
  * \param buffer  Reference to source buffer.
  * \param startIndex  Index where integer starts.
  * \return  Correct integer representation.
  */
  int32_t ParseIntegerLE(const Buffer& buffer, size_t startIndex) const;
  /*!
  * Parses Little Endian float from provided buffer starting at specified index.
  *
  * \param buffer  Reference to source buffer.
  * \param startIndex  Index where float starts.
  * \return  Correct float representation.
  */
  float ParseFloatLE(const Buffer& buffer, size_t startIndex) const;

  std::vector<std::string> ParseImageFilenamesTextFile(std::string_view filepath) const;

private:
  Database _primaryDb;

  Database _secondaryDb;

  KeywordsContainer _keywords;
  std::map<size_t, Image> _images;
  

  std::string _softmaxFilepath;
  std::string _preSoftmaxFilepath;

  std::string _imagesListFilepath;
  std::string _imagesPath;
  size_t _columnIndexFilename;
  size_t _imageListFileLineLength;

  size_t _numRows;
  size_t _idOffset;

  float _currBoolModelProbTreshold;

};