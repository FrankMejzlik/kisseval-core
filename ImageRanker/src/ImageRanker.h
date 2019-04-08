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


};


struct ImageData 
{

};

class ImageRanker
{
  // Structures
public:
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

  enum RankingModel 
  {
    cBoolean,
    cFuzzyLogic
  };

  enum QueryOrigin
  {
    cDeveloper,
    cPublic
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

  std::vector<std::pair<size_t, size_t>> RunModelTest(QueryOrigin queryOrigin, RankingModel rankingModel = DEFAULT_RANKING_MODEL) const;

  std::vector<ImageReference> GetRelevantImages(const std::string& query, RankingModel rankingModel = DEFAULT_RANKING_MODEL) const;


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

  
  size_t GetRandomImageId() const;
  

  std::vector<ImageReference> GetRelevantImagesBooleanModel(const std::string& query) const;
  std::vector<ImageReference> GetRelevantImagesFuzzyLogicModel(const std::string& query) const;

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

  std::vector< std::pair< size_t, std::unordered_map<size_t, float> > > ParseSoftmaxBinFileFiltered(std::string_view filepath, float minProbabilty) const;

  std::unordered_map<size_t, std::pair<size_t, std::string> > ParseKeywordClassesTextFile(std::string_view filepath) const;

  std::unordered_map<size_t, std::pair<size_t, std::string> > ParseHypernymKeywordClassesTextFile(std::string_view filepath) const;

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
  

  std::string _imagesListFilepath;
  std::string _imagesPath;
  size_t _columnIndexFilename;
  size_t _imageListFileLineLength;

  size_t _numRows;
  size_t _idOffset;



};