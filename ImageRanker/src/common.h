/**
 * Common defines, typedefs and types.
 */

#ifndef _IR_COMMON_H_
#define _IR_COMMON_H_

#include "config.h"
#include "custom_exceptions.h"

/**
 * Unique IDs of all possible frames datasets used.
 *
 * EXAMPLES: V3C1 20k subset, V3C1
 */
enum class eDatasetId
{
  V3C1_20K_SUBSET_2019,
  V3C1_2019,
  V3C1_VBS2020
};

/**
 * Unique IDs of all possible vocabularies used.
 *
 * EXAMPLES: Viret synsets, Google AI set, BoW vocabulary
 */
enum class eVocabularyId
{
  VIRET_1200_WORDNET_2019 = 0,
  GOOGLE_AI_20K_2019 = 100,
  BOW_2020 = 1000
};

/**
 * Unique IDs of all possible vocabularies used.
 *
 * EXAMPLES: NasNet 2019 by TS, GoogLeNet 2019 by TS, Google AI from 2019, BoW variant by Xirong
 */
enum class eScoringsId
{
  NASNET_2019 = 0,
  GOOG_LE_NET_2019 = 1,
  GOOGLE_AI_2019 = 100,
  RESNEXT_RESNET_BOW_2020 = 1000
};

/**
 * Type representing reference to file containing input image scoring data.
 *
 * FORMAT: ( <vocabulary_type>, <scoring_type>, <filepath> )
 *
 *  <vocabulary_type>:
      Unique ID determining what format file is in (e.g. Viret, Google AI Vision).
 *    - \see enum class eKeywordsDataType
 *
 *  <scoring_type>:
 *    Unique ID determining what how scoring data has been generated.
 *    - \see enum class eImageScoringDataType
 *
 *  <filepath>:
*     String containing filepath (relative of absolute) to the data file.
 *
 */
using DataFileSrc = std::tuple<eVocabularyId, eScoringsId, std::string>;

/**
 * Base class for all usable data packs.
 */
struct BaseDataPack
{
  eDatasetId target_dataset;
};

/**
 * Represents one input dataset containg selected frames we have scorings for.
 */
struct DatasetDataPack : public BaseDataPack
{
  std::string images_dir;
  std::string imgage_to_ID_fpth;
};

/**
 * Represents input data for 'Viret' based models.
 */
struct ViretDataPack : public BaseDataPack
{
  /*
   * Substructures
   */
  struct VocabData
  {
    DataFileSrc keyword_synsets_fpth;
  };

  struct ScoreData
  {
    DataFileSrc presoftmax_scorings_fpth;
    DataFileSrc softmax_scorings_fpth;
    DataFileSrc deep_features_fpth;
  };

  /*
   * Member variables
   */
  VocabData vocabulary_data;
  ScoreData score_data;
};

/**
 * Represents input data for 'Google AI' based models.
 */
struct GoogleDataPack : public BaseDataPack
{
  struct VocabData
  {
    DataFileSrc keyword_synsets_fpth;
  };

  struct ScoreData
  {
    DataFileSrc presoftmax_scorings_fpth;
    DataFileSrc softmax_scorings_fpth;
    DataFileSrc deep_features_fpth;
  };

  VocabData vocabulary_data;
  ScoreData score_data;
};

/**
 * Represents input data for 'BoW/W2V++' based models.
 */
struct BowDataPack : public BaseDataPack
{
  struct VocabData
  {
    DataFileSrc word_to_idx_fpth;
    DataFileSrc kw_features_fpth;
    DataFileSrc kw_bias_vec_fpth;
    DataFileSrc kw_PCA_mat_fpth;
  };

  struct ScoreData
  {
    DataFileSrc img_features_fpth;
  };

  VocabData vocabulary_data;
  ScoreData score_data;
};

// =====================================
//  NOT REFACTORED CODE BELOW
// =====================================

#include <cstddef>
#include <vector>

// Forward decls
class Keyword;
class Image;

inline std::string ToString(eVocabularyId id)
{
  std::string resultString;

  switch (id)
  {
    case eVocabularyId::VIRET_1200_WORDNET_2019:
      resultString += "eKeywordsDataType::cViret1";
      break;

    case eVocabularyId::GOOGLE_AI_20K_2019:
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
using KeywordsFileRef = std::tuple<eVocabularyId, std::string>;

inline std::string ToString(eScoringsId id)
{
  std::string resultString;

  switch (id)
  {
    case eScoringsId::NASNET_2019:
      resultString += "cNasNet";
      break;

    case eScoringsId::GOOG_LE_NET_2019:
      resultString += "cGoogLeNet";
      break;

    case eScoringsId::GOOGLE_AI_2019:
      resultString += "cGoogleAI";
      break;
  }

  return resultString;
}

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
using DataId = std::tuple<eVocabularyId, eScoringsId>;

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
enum class eExportFileTypeId
{
  cUserAnnotatorQueries = 0,
  cNetNormalizedScores = 1,
  cQueryNumHits = 2,
  cNativeQueries = 2
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
constexpr size_t operator""_z(unsigned long long int x) { return static_cast<size_t>(x); }
// std::vector<std::vector<std::pair<bool, size_t>>>
using Clause = std::vector<std::pair<bool, size_t>>;
using CnfFormula = std::vector<Clause>;
using InteractiveSearchAction = std::tuple<size_t, size_t, size_t>;

struct StatPerKwSc
{
  StatPerKwSc() : m_labelHitProb{0.0f} {}
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

enum class UserDataSourceId
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

using TestSettings = std::tuple<InputDataTransformId, RankingModelId, UserDataSourceId, RankingModelSettings,
                                InputDataTransformSettings>;

using Buffer = std::vector<std::byte>;

//! This is returned to front end app when some quesries are submited
//! <SessionID, image filename, user keywords, net <keyword, probability> >
using GameSessionQueryResult =
    std::tuple<std::string, std::string, std::vector<std::string>, std::vector<std::pair<std::string, float>>>;

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
using ChartData = std::vector<std::pair<uint32_t, uint32_t>>;

using UserImgQuery = std::tuple<size_t, CnfFormula, bool>;
using UserImgQueryRaw = std::tuple<size_t, std::vector<size_t>>;

//! FORMAT: (imageId, userNativeQuery, timestamp, sessionId, isManuallyValidated)
using UserDataNativeQuery = std::tuple<size_t, std::string, size_t, std::string, bool>;

//! (datasourceID, percentageofAll)
using UserAccuracyChartDataMisc = std::tuple<size_t, float>;

using UserAccuracyChartData = std::pair<UserAccuracyChartDataMisc, ChartData>;

class SimulatedUser
{
 public:
  SimulatedUser() : m_exponent(1) {}

 public:
  int m_exponent;
};

#endif  // _IR_COMMON_H_
