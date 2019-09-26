#pragma once

#include <vector>
#include <cstddef>

#include "config.h"
#include "custom_exceptions.h"

using size_t = std::size_t;

// Forward decls
class Keyword;
class Image;

/*!
 * Enum assigning IDs to types
 */
enum class eKeywordsDataType
{
  cViret1 = 0,
  cGoogleAI = 100
};

inline std::string ToString(eKeywordsDataType id)
{
  std::string resultString;

  switch (id)
  {
  case eKeywordsDataType::cViret1:
    resultString += "eKeywordsDataType::cViret1";
    break;

  case eKeywordsDataType::cGoogleAI:
    resultString += "eKeywordsDataType::cGoogleAI";
    break;
  }

  return resultString;
}

/*!
 * Type representing reference to file containing keyword descriptions
 * 
 * FORMAT:
 *  (type ID, filepath)
 * 
 *  Type ID - unique ID determining what format file is in (e.g. Viret, Google AI Vision)
 *    \see enum class eKeywordsDataType
 * 
 *  filepath - String containing filepath (relative of absolute) to file
 * 
 */
using KeywordsFileRef = std::tuple<eKeywordsDataType, std::string>;



/*!
 * Enum assigning IDs to scoring data types
 */
enum class eImageScoringDataType
{
  cNasNet = 0,
  cGoogLeNet = 1,
  cGoogleAI = 100
};

inline std::string ToString(eImageScoringDataType id)
{
  std::string resultString;

  switch (id)
  {
  case eImageScoringDataType::cNasNet:
    resultString += "cNasNet";
    break;

  case eImageScoringDataType::cGoogLeNet:
    resultString += "cGoogLeNet";
    break;

  case eImageScoringDataType::cGoogleAI:
    resultString += "cGoogleAI";
    break;
  }

  return resultString;
}

/*!
 * Type representing reference to file containing input image scoring data
 *
 * FORMAT:
 *  (KW type ID, scoring type ID, filepath)
 *
 *  KW type ID - unique ID determining what format file is in (e.g. Viret, Google AI Vision)
 *    \see enum class eKeywordsDataType
 *  scoring type ID - unique ID determining what how scoring data has been generated
 *    \see enum class eImageScoringDataType
 *  filepath - String containing filepath (relative of absolute) to file
 *
 */
using ScoringDataFileRef = std::tuple<eKeywordsDataType, eImageScoringDataType, std::string>;


/*!
 * Structure holding data about occurance rate of one keyword
 * 
 * FORMAT:
 *  ( synsetId, synsetWord, totalValue )
 * 
 * synsetId - ID of this synset
 * synsetWord - string representing this synset
 * totalValue - accumulated total value of this synset in given resultset
 */
using KeywordOccurance = std::tuple<size_t, std::string, float>;

/*!
 * Structure for returnig relevant images based on input query
 * 
 * FORMAT:
 *  ( sortedRelevantImages,  kwOccurances, targetImagePosition )
 * 
 * sortedRelevantImages - images sorted based on ranking for given query
 * kwOccurances - keyword occurances for given number of results
 *  \see KeywordOccurance
 * targetImagePosition - position of image that was set as target
 */
using RelevantImagesResponse = std::tuple<std::vector<const Image*>, std::vector<KeywordOccurance>, size_t>;

//! Identifier for ( kwTypeId, scoringTypeId )
using KwScoringDataId = std::tuple<eKeywordsDataType, eImageScoringDataType>;

//! Unique ID for type of final data transformation
using TransformFullId = size_t;

/*!
 * Structure representing ptr to keyword and its scores
 * 
 * FORMAT:
 *  [ ( keywordPtr, keywordScore ), (...)  ]
 */ 
using KeywordPtrScoringPair = std::vector<std::tuple<Keyword*, float>>;

using ImageIdFilenameTuple = std::tuple<size_t, std::string>;

/*!
 * 
 * (num of distincts keywords)
 */
using KeywordsGeneralStatsTuple = std::tuple<size_t>;

/*!
 *
 * (..)
 */
using ScoringsGeneralStatsTuple = std::tuple<size_t>;

/*!
 *
 * (min number of labels, max # labels, avg labels, median labels, prob of hit)
 */
using AnnotatorDataGeneralStatsTuple = std::tuple<size_t, size_t, float, float, float>;

/*!
 *
 * ()
 */
using RankerDataGeneralStatsTuple = std::tuple<size_t>;


/*!
 *      0 - eUserAnnotatorQueries
 *      1 -  eNetNormalizedScores
 *      2 - eQueryNumHits
 * 
 */
enum class eExportFileTypeId {
  cUserAnnotatorQueries = 0,
  cNetNormalizedScores = 1,
  cQueryNumHits = 2
};


// ------------------------------------------------
enum class eTempQueryOpOutter
{
  cSum,
  cProduct
};

enum class eTempQueryOpInner
{
  cSum,
  cProduct,
  cMax
};

/*
 * User defined literals
 */
constexpr size_t operator ""_z(unsigned long long int x)
{
  return static_cast<size_t>(x);
}

using Clause = std::vector<std::pair<bool, size_t>>;
using CnfFormula = std::vector<Clause>;
using InteractiveSearchAction = std::tuple<size_t, size_t, size_t>;

struct StatPerKwSc
{
  StatPerKwSc() :
    m_labelHitProb{0.0f}
  {}
  float m_labelHitProb;
};

enum class InteractiveSearchOrigin
{
  cDeveloper,
  cPublic
};

enum class RankingModelId
{
  cBooleanBucket = 1,
  cBooleanExtended = 2,
  cViretBase = 3
};

enum class InputDataTransformId
{
  cSoftmax = 100,
  cXToTheP = 200,
  cSine = 300,
  cNoTransform = NO_TRANSFORM_ID
};

enum class DataSourceTypeId
{
  cDeveloper = 0,
  cPublic = 1,
  cManaged = 2,

  cAll = 999,

  // Simulated variants
  cDeveloperSimulated = 10000,
  cPublicSimulated = 10001,
  cManagedSimulated = 10002,
  cAllSimulated = 10999,

  // Extended variants
  cDeveloperExtended = 20000,
  cPublicExtended = 20001,
  cManagedExtended = 20002,
  cAllExtended = 20999

};



/*!
*  FORMAT:
*  ========================================================
* (x) cBooleanBucket
* --- 
* 
*    0 => keyword frequency handling
*    1 => trueTreshold
*    2 => inBucketSorting
*    3 => outter temporal query operation (\ref eTempQueryOpOutter)
*       0: sum
*       1: product
*    4 => inner temporal query operation  (\ref eTempQueryOpInner)
*       0: sum
*       1: product
*       2: max
* 
*  ========================================================
*  (x) cBooleanExtended:
*  ---
* 
*  ========================================================
*  (x) cViretBase
*  ----
*    0 => keyword frequency handling
*    1 => ignoreTreshold
*    2 => rankCalcMethod
*       cMultSum = 0,
*       cMultMax = 1,
*       cSumSum = 2,
*       cSumMax = 3,
*       cMaxMax= 4
*    3 => outter temporal query operation (\ref eTempQueryOpOutter)
*       0: sum
*       1: product
*    4 => inner temporal query operation (\ref eTempQueryOpInner)
*       0: sum
*       1: product
*       2: max
*/
using RankingModelSettings = std::vector<std::string>;


/*!
* FORMAT:
*  1: cSoftmax
*  2: cXToTheP:
*    0 => Exponent fot x^p postscale transformation
*    1 => 
*       0: SUM accumulated precoputed hypernyms
*       1: MAX accumulated precoputed hypernyms
*/
using InputDataTransformSettings = std::vector<std::string>;


/*!
* FORMAT:
*  0 => Simulated user exponent
*/
using SimulatedUserSettings = std::vector<std::string>;






using TestSettings = std::tuple<InputDataTransformId, RankingModelId, DataSourceTypeId, RankingModelSettings, InputDataTransformSettings>;

using Buffer = std::vector<std::byte>;

//! This is returned to front end app when some quesries are submited
//! <SessionID, image filename, user keywords, net <keyword, probability> >
using GameSessionQueryResult = std::tuple<std::string, std::string, std::vector<std::string>, std::vector<std::pair<std::string, float>>>;

//! Array of those is submited from front-end app game
using GameSessionInputQuery = std::tuple<std::string, size_t, std::string>;

/*!
 * Keywords that are possible for given prefix
 * 
 * FORMAT:
 *  [ nearKeywords ]
 * 
 */
using NearKeywordsResponse = std::vector<Keyword*>;


/*! <wordnetID, keyword, description> */
using KeywordData = std::tuple<size_t, std::string, std::string>;

/*!
 * Output data for drawing chart
 */
using ChartData = std::vector <std::pair<uint32_t, uint32_t>>;

using UserImgQuery = std::tuple<size_t, CnfFormula, bool>;
using UserImgQueryRaw = std::tuple<size_t, std::vector<size_t>>;


//! (datasourceID, percentageofAll)
using UserAccuracyChartDataMisc = std::tuple<size_t, float>;

using UserAccuracyChartData = std::pair<UserAccuracyChartDataMisc, ChartData>;


class SimulatedUser
{
public:
  SimulatedUser():
    m_exponent(1)
  { }

public:
  int m_exponent;
};


