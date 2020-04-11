/**
 * Common defines, typedefs and types.
 */

#ifndef _IR_COMMON_H_
#define _IR_COMMON_H_

#include <array>
#include <string>
#include <vector>
using namespace std::string_literals;

#include "config.h"
#include "custom_exceptions.h"
#include "log.h"

namespace image_ranker
{
class Keyword;

/**********************************************
 * Name definitions
 ***********************************************/
enum class eImagesetId
{
  V3C1_20K,
  V3C1_2019,
  V3C1_JAN_2020,
  LSC_APR_2020,
  _COUNT
};

const std::array<std::pair<std::string, std::string>, size_t(eImagesetId::_COUNT)> eImagesetId_labels = {{
    std::pair("V3C1_20k", "Subset of V31C dataset, every 50th selected frame."),
    std::pair("V3C1_2019", "Selected frames from V31C dataset by Tomas Soucek in 2019."),
    std::pair("V3C1_Jan_2020", "Selected frames from V31C dataset by Tomas Soucek for VBS 2020."),
    std::pair("LSC_Apr_2020", "Selected frames from V31C dataset by Tomas Soucek for LSC 2020."),
}};

inline const std::pair<std::string, std::string>& enum_label(eImagesetId val)
{
  return eImagesetId_labels[size_t(val)];
}

enum class eVocabularyId
{
  VIRET_WORDNET_2019,
  GOOGLE_AI_2019,
  BOW_JAN_2020,
  NATIVE_LANGUAGE,
  _COUNT
};

const std::array<std::pair<std::string, std::string>, size_t(eVocabularyId::_COUNT)> eVocabularyId_labels = {{
    std::pair("VIRET_WordNet2019", "With ~1300 phrases."),
    std::pair("Google_AI_2019", "In total ~ 5000 phrases."),
    std::pair("BoW_Jan_2020", "With ~12k words."),
    std::pair("native_language", "Free form native language sentences."),
}};

inline const std::pair<std::string, std::string>& enum_label(eVocabularyId val)
{
  return eVocabularyId_labels[size_t(val)];
}

enum class eDataPackId
{
  NASNET_2019,
  GOOGLENET_2019,
  GOOGLE_AI_2019,
  RESNET_RESNEXT_BOW_JAN_2019,
  _COUNT
};

const std::array<std::pair<std::string, std::string>, size_t(eDataPackId::_COUNT)> eDataPackId_labels = {{
    std::pair("NASNET_2019", "NasNet for 20k subset."),
    std::pair("GOOGLENET_2019", "GoogLeNet for 20k subset."),
    std::pair("GOOGLE_AI_2019", "Google AI results on 20k subset."),
    std::pair("RESNET_RESNEXT_BOW_JAN_2019", "BoW model by Xirong for 20k subset."),
}};

inline const std::pair<std::string, std::string>& enum_label(eDataPackId val)
{
  return eDataPackId_labels[size_t(val)];
}

enum class eModelIds
{
  BOOLEAN,
  VECTOR_SPACE,
  MULT_SUM_MAX,
  BOOLEAN_BUCKET,
  BOW_VBS2020,
  _COUNT
};

const std::array<std::pair<std::string, std::string>, size_t(eModelIds::_COUNT)> eModelIds_labels = {{
    std::pair("Boolean", ""),
    std::pair("Vector_space", ""),
    std::pair("Mult-sum-max", ""),
    std::pair("Boolean_bucket", ""),
    std::pair("BoW_VBS2020", "BoW model by Xirong."),
}};

inline const std::pair<std::string, std::string>& enum_label(eModelIds val) { return eModelIds_labels[size_t(val)]; }

enum class eTransformationIds
{
  LINEAR_01,
  SOFTMAX,
  NO_TRANSFORM,
  _COUNT
};

const std::array<std::pair<std::string, std::string>, size_t(eTransformationIds::_COUNT)> eTransformationIds_labels = {{
    std::pair("linear_01", ""),
    std::pair("softmax", ""),
    std::pair("no_transform", ""),
}};

inline const std::pair<std::string, std::string>& enum_label(eTransformationIds val)
{
  return eTransformationIds_labels[size_t(val)];
}

enum class eModelOptsKeys
{
  MODEL_ID,
  TRANSFORM_ID,
  _COUNT
};

const std::array<std::pair<std::string, std::string>, size_t(eModelOptsKeys::_COUNT)> eModelOptsKeys_labels = {{
    std::pair("model_ID", ""),
    std::pair("transform_ID", ""),
}};

inline const std::pair<std::string, std::string>& enum_label(eModelOptsKeys val)
{
  return eModelOptsKeys_labels[size_t(val)];
}


using ModelKeyValOption = std::pair<std::string, std::string>;
/**********************************************
 **********************************************
 ***********************************************/

using KeywordId = size_t;

template <typename T>
struct Literal
{
  T atom;
  bool neg;
};
using Clause = std::vector<Literal<KeywordId>>;
using CnfFormula = std::vector<Clause>;

/**
 * Basic typenames
 */
using DataName = std::string;
using FrameId = uint32_t;
using VideoId = uint32_t;
using ShotId = uint32_t;
using FrameNumber = uint32_t;

template <typename T>
using Vector = std::vector<T>;

template <typename T>
using Matrix = std::vector<Vector<T>>;

/** User query for testing models.
 *  FORMAT: (user_query, target_frame_ID) */
using UserTestQuery = std::pair<std::vector<CnfFormula>, FrameId>;

template <typename T>
constexpr T ERR_VAL()
{
  return std::numeric_limits<T>::max();
}

/**
 * Base class for all usable data packs.
 */
struct BaseDataPackRef
{
  std::string ID;
  std::string description;
  std::string target_imageset;
};

/**
 * Represents one input dataset containg selected frames we have scorings for.
 */
struct DatasetDataPackRef : public BaseDataPackRef
{
  std::string images_dir;
  std::string imgage_to_ID_fpth;
};

/**
 * Represents input data for 'Viret' based models.
 */
struct ViretDataPackRef : public BaseDataPackRef
{
  /*
   * Substructures
   */
  struct VocabData
  {
    std::string ID;
    std::string description;
    std::string keyword_synsets_fpth;
  };

  struct ScoreData
  {
    std::string presoftmax_scorings_fpth;
    std::string softmax_scorings_fpth;
    std::string deep_features_fpth;
  };

  /*
   * Member variables
   */
  VocabData vocabulary_data;
  ScoreData score_data;
};

using VecMat = std::vector<std::vector<float>>;

/**
 * Represents input data for 'Google AI' based models.
 */
struct GoogleDataPackRef : public BaseDataPackRef
{
  struct VocabData
  {
    std::string ID;
    std::string description;
    std::string keyword_synsets_fpth;
  };

  struct ScoreData
  {
    std::string presoftmax_scorings_fpth;
    std::string softmax_scorings_fpth;
    std::string deep_features_fpth;
  };

  VocabData vocabulary_data;
  ScoreData score_data;
};

/**
 * Represents input data for 'BoW/W2V++' based models.
 */
struct BowDataPackRef : public BaseDataPackRef
{
  struct VocabData
  {
    std::string ID;
    std::string word_to_idx_fpth;
    std::string kw_features_fpth;
    std::string kw_bias_vec_fpth;
    std::string kw_PCA_mat_fpth;
  };

  struct ScoreData
  {
    std::string img_features_fpth;
  };

  VocabData vocabulary_data;
  ScoreData score_data;
};

using DataPackId = std::string;
using PackModelId = std::string;
using PackModelCommands = std::string;
using StringId = std::string;

enum class eDataPackType
{
  VIRET,
  GOOGLE,
  BOW
};

enum class eModelOptType
{
  INT,
  FLOAT,
  STRING
};

struct ModelOption
{
  std::string ID;
  std::string name;
  std::string description;
  eModelOptType type;
  std::vector<std::string> enum_vals;
  std::pair<float, float> range;
};

/**
 * Configuration pack for VIRET based mode
 */
struct ModelInfo
{
  std::string ID;
  std::string name;
  std::string description;
  eDataPackType target_pack_type;
  std::vector<ModelOption> options;
};

struct SelFrame
{
  SelFrame(FrameId ID, FrameId external_ID, const std::string& filename, VideoId videoId, ShotId shotId,
           FrameNumber frameNumber)
      : m_ID(ID),
        m_external_ID(external_ID),
        m_video_ID(videoId),
        m_shot_ID(shotId),
        m_frame_number(frameNumber),
        m_num_successors(0),
        m_filename(filename)
  {
  }

  FrameId m_ID;
  FrameId m_external_ID;

  VideoId m_video_ID;
  ShotId m_shot_ID;
  FrameNumber m_frame_number;

  size_t m_num_successors;
  std::string m_filename;
};

using ImageIdFilenameTuple = std::tuple<FrameId, std::string>;

struct RankingResult
{
  std::vector<FrameId> m_frames;
  FrameId target;
  size_t target_pos;
};

/** User queries come in in as vector of those. */
struct AnnotatorUserQuery
{
  std::string session_ID;
  std::string user_query_encoded;
  std::string user_query_readable;
  FrameId target_frame_ID;
};

struct GameSessionQueryResult
{
  std::string session_ID;
  std::string frame_filename;
  std::string human_readable_query;
  std::string model_top_query;
};

/**
 * Keywords that are possible for given prefix.
 */
struct AutocompleteInputResult
{
  std::vector<const Keyword*> top_keywords;
};

struct ImagesetInfo
{
  const std::string& ID;
  const std::string& directory;
  size_t num_sel_frames;
};

struct LoadedImagesetsInfo
{
  std::vector<ImagesetInfo> imagesets_info;
};

struct DataPackInfo
{
  const std::string& ID;
  const std::string& description;
  const std::string& target_imageset_ID;

  const std::string& vocabulary_ID;
  const std::string& vocabulary_description;
};

struct LoadedDataPacksInfo
{
  std::vector<DataPackInfo> data_packs_info;
};

enum class eMainTempRankingAggregation
{
  cSum,
  cProduct
};

enum class eSuccesorAggregation
{
  cSum,
  cProduct,
  cMax
};

// =====================================
//  NOT REFACTORED CODE BELOW
// =====================================

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
using RelevantImagesResponse = std::tuple<std::vector<const FrameId*>, std::vector<KeywordOccurance>, size_t>;

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

/*
 * User defined literals
 */
constexpr size_t operator""_z(unsigned long long int x) { return static_cast<size_t>(x); }
// std::vector<std::vector<std::pair<bool, size_t>>>

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

}  // namespace image_ranker
#endif  // _IR_COMMON_H_
