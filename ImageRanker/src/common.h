/**
 * \file common.h
 *
 * Common defines, typedefs and types used througout the whole program.
 */

#ifndef _IR_COMMON_H_
#define _IR_COMMON_H_

#include <array>
#include <map>
#include <set>
#include <string>
#include <memory>
#include <vector>
using namespace std::string_literals;

#include "config.h"
#include "custom_exceptions.h"
#include "log.h"
#include "macros.h"

namespace image_ranker
{
struct Keyword;

/**********************************************
 * Typedefs
 ***********************************************/
using KeywordId = size_t;
using DataName = std::string;
using FrameId = uint32_t;
using VideoId = uint32_t;
using ShotId = uint32_t;
using FrameNumber = uint32_t;
using ModelKeyValOption = std::pair<std::string, std::string>;

template <typename T>
using Vector = std::vector<T>;

template <typename T>
using Matrix = std::vector<Vector<T>>;

template <typename T>
struct Literal
{
  T atom;
  bool neg{false};
};
using Clause = std::vector<Literal<KeywordId>>;
using CnfFormula = std::vector<Clause>;

/** User query for testing models.
 *  FORMAT: (user_query, target_frame_ID) */
using UserTestQuery = std::pair<std::vector<CnfFormula>, FrameId>;
using UserTestNativeQuery = std::pair<std::vector<std::string>, FrameId>;

/** Value indicating error/invalid/"NULL" value for the given type. */
template <typename T>
constexpr T ERR_VAL()
{
  return std::numeric_limits<T>::max();
}

/**********************************************
 * Name definitions
 ***********************************************/

enum class eSearchSessionEndStatus
{
  GAVE_UP = 0,
  FOUND_TARGET = 1
};

enum class eSimUserTarget
{
  SINGLE_QUERIES,
  TEMP_QUERIES,
  AUGMENT_REAL_WITH_TEMP,
  _COUNT
};

enum class eDistFunction
{
  EUCLID,
  EUCLID_SQUARED,
  MANHATTAN,
  COSINE,
  COSINE_NONORM
};

enum class eTermFrequency
{
  NATURAL,
  LOGARIGHMIC,
  AUGMENTED,
  _COUNT
};

enum class eInvDocumentFrequency
{
  NONE,
  IDF,
  _COUNT
};

enum class eModelIds
{
  BOOLEAN,
  VECTOR_SPACE,
  MULT_SUM_MAX,
  BOOLEAN_BUCKET,
  W2VV_BOW_VBS2020,
  _COUNT
};

const std::array<std::pair<std::string, std::string>, size_t(eModelIds::_COUNT)> eModelIds_labels = { {
    std::pair("boolean", ""),
    std::pair("vector_space", ""),
    std::pair("mult-sum-max", ""),
    std::pair("boolean_bucket", ""),
    std::pair("w2vv_bow_plain", "BoW model by Xirong."),
} };

inline const std::pair<std::string, std::string>& enum_label(eModelIds val) { return eModelIds_labels.at(size_t(val)); }

enum class eSimUserIds
{
  NO_SIM,
  USER_X_TO_P,
  _COUNT
};

const std::array<std::pair<std::string, std::string>, size_t(eSimUserIds::_COUNT)> eSimUserIds_labels = {
  { std::pair("no_sim_user", ""), std::pair("user_model_x_to_p", "") }
};

inline const std::pair<std::string, std::string>& enum_label(eSimUserIds val)
{
  return eSimUserIds_labels.at(size_t(val));
}

enum class eTransformationIds
{
  LINEAR_01,
  SOFTMAX,
  NO_TRANSFORM,
  _COUNT
};

const std::array<std::pair<std::string, std::string>, size_t(eTransformationIds::_COUNT)> eTransformationIds_labels = {
  {
      std::pair("linear_01", ""),
      std::pair("softmax", ""),
      std::pair("no_transform", ""),
  }
};

inline const std::pair<std::string, std::string>& enum_label(eTransformationIds val)
{
  return eTransformationIds_labels.at(size_t(val));
}

enum class eModelOptsKeys
{
  MODEL_ID,
  TRANSFORM_ID,
  SIM_USER_ID,
  SIM_USER_TYPE,
  MODEL_OPERATIONS,
  MODEL_INNER_OP,
  MODEL_OUTTER_OP,
  MODEL_IGNORE_THRESHOLD,
  MODEL_TRUE_THRESHOLD,
  MODEL_IDF_COEF,
  MODEL_DIST_FN,
  MODEL_TERM_TF,
  MODEL_TERM_IDF,
  MODEL_QUERY_TF,
  MODEL_QUERY_IDF,
  MODEL_SUB_PCA_MEAN,
  _COUNT
};

const std::array<std::pair<std::string, std::string>, size_t(eModelOptsKeys::_COUNT)> eModelOptsKeys_labels = {
  { std::pair("model", ""), std::pair("transform", ""), std::pair("sim_user", ""), std::pair("sim_user_type", ""),
    std::pair("model_operations", ""), std::pair("model_inner_op", ""), std::pair("model_outter_op", ""),
    std::pair("model_ignore_treshold", ""), std::pair("model_true_threshold", ""),
    std::pair("model_IDF_type_idf_coef", ""), std::pair("model_dist_fn", ""), std::pair("model_term_tf", ""),
    std::pair("model_term_idf", ""), std::pair("model_query_tf", ""), std::pair("model_query_idf", ""),
    std::pair("sub_PCA_mean", "") }
};

inline const std::pair<std::string, std::string>& enum_label(eModelOptsKeys val)
{
  return eModelOptsKeys_labels.at(size_t(val));
}

struct UserQueriesStats
{
  float median_num_labels_asigned{ std::numeric_limits<float>::quiet_NaN() };
  float avg_num_labels_asigned{ std::numeric_limits<float>::quiet_NaN() };
};

struct DataParseStats
{
  float median_num_labels_asigned{ std::numeric_limits<float>::quiet_NaN() };
  float avg_num_labels_asigned{ std::numeric_limits<float>::quiet_NaN() };
};

struct DataPackStats
{
  float median_num_labels_asigned{ std::numeric_limits<float>::quiet_NaN() };
  float avg_num_labels_asigned{ std::numeric_limits<float>::quiet_NaN() };

  float median_num_labels_used_in_query{ std::numeric_limits<float>::quiet_NaN() };
  float avg_num_labels_used_in_query{ std::numeric_limits<float>::quiet_NaN() };

  float label_hit_prob{ std::numeric_limits<float>::quiet_NaN() };
};

struct DataInfo
{
  Vector<float> mins;
  Vector<float> maxes;
  std::vector<std::vector<KeywordId>> top_classes;
};

struct FrameFilenameOffsets
{
  size_t v_ID_off;
  size_t v_ID_len;
  size_t s_ID_off;
  size_t s_ID_len;
  size_t fn_ID_off;
  size_t fn_ID_len;
};

/**
 * Base class for all usable data packs.
 */
struct BaseDataPackRef
{
  std::string ID;
  std::string description;
  std::string model_options;
  std::string target_imageset;
};

struct BaseImagesetkRef
{
  std::string ID;
  std::string description;
  std::string target_imageset;
  FrameFilenameOffsets offsets;
};

/**
 * Represents one input dataset containg selected frames we have scorings for.
 */
struct DatasetDataPackRef : public BaseImagesetkRef
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
  bool accumulated{ false };
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
  };

  VocabData vocabulary_data;
  ScoreData score_data;
};

/**
 * Represents input data for 'W2VV++' based models.
 */
struct W2vvDataPackRef : public BaseDataPackRef
{
  struct VocabData
  {
    std::string ID;
    std::string description;
    std::string keyword_synsets_fpth;

    std::string kw_features_fpth;
    size_t kw_features_dim;
    size_t kw_features_data_offset;

    std::string kw_bias_vec_fpth;
    size_t kw_bias_vec_dim;
    size_t kw_bias_vec_data_offset;

    std::string kw_PCA_mat_fpth;
    size_t kw_PCA_mat_dim;
    size_t kw_PCA_mat_data_offset;

    std::string kw_PCA_mean_vec_fpth;
    size_t kw_PCA_mean_vec_dim;
    size_t kw_PCA_mean_vec_data_offset;
  };

  struct ScoreData
  {
    std::string img_features_fpth;
    size_t img_features_dim;
    size_t img_features_offset;
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

/**
 * Represents one selected frame in some imageset.
 */
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

  FrameId m_ID{ ERR_VAL<FrameId>() };
  FrameId m_external_ID{ ERR_VAL<FrameId>() };

  VideoId m_video_ID{ ERR_VAL<VideoId>() };
  ShotId m_shot_ID{ ERR_VAL<ShotId>() };
  FrameNumber m_frame_number{ ERR_VAL<FrameNumber>() };

  size_t m_num_successors{ ERR_VAL<size_t>() };
  std::string m_filename{};
};

using ImageIdFilenameTuple = std::tuple<FrameId, std::string>;

/**
 *  Result of rank_query call.
 */
struct RankingResult
{
  std::vector<FrameId> m_frames;
  FrameId target;
  size_t target_pos;
};

/**
 *  Result of rank_query call with filenames included.
 */
struct RankingResultWithFilenames
{
  std::vector<std::pair<FrameId, std::string>> m_frames;
  FrameId target;
  size_t target_pos;
};

/** User queries come in in as vector of those. */
struct AnnotatorUserQuery
{
  std::string session_ID;

  /** Query containing unique identifiers (ID, or unique string repre).
      EXAMPLE: "&-1+-31+>&-13+-43+" */
  std::vector<std::string> user_query_encoded;

  /** Query in the same form user entered it (cause the same wordnet ID can have multiple string repres).
      EXAMPLE: "cat dog food > grass owl" */
  std::vector<std::string> user_query_readable;

  /** Sequence of images user described */
  std::vector<FrameId> target_sequence_IDs;
};

/** Data that describe user annotations vs classifier annotations */
struct GameSessionQueryResult
{
  std::string session_ID;
  std::string frame_filename;
  std::string human_readable_query;
  std::string model_top_query;
};

/** Data needed for frame detail. */
struct FrameDetailData
{
  FrameId frame_ID;
  std::string data_pack_ID;
  std::string model_options;
  std::vector<const Keyword*> top_keywords;
};

/** Data type for holding quantile aggregated plot data. */
template <typename XType, typename FXType>
struct QuantileLineChartData
{
  std::vector<XType> x;
  std::vector<FXType> y_min;
  std::vector<FXType> y_q1;
  std::vector<FXType> y_q2;
  std::vector<FXType> y_q3;
  std::vector<FXType> y_max;
  std::vector<size_t> count;
};

template <typename XType, typename FXType>
struct MedianLineMultichartData
{
  /** X values.*/
  std::vector<std::vector<XType>> x;

  /** Y values. */
  std::vector<std::vector<FXType>> medians;
};

/** Holds data for median & quantile chart plotting */
struct SearchSessRankChartData
{
  MedianLineMultichartData<size_t, float> median_multichart;
  QuantileLineChartData<size_t, float> aggregate_quantile_chart;
};

/** Data type for holding histogram data for plitting */
template <typename XType, typename FXType>
struct HistogramChartData
{
  /** X values.*/
  std::vector<XType> x;

  /** Y values. */
  std::vector<FXType> fx;
};

/**
 * Keywords that are possible for given prefix.
 */
struct AutocompleteInputResult
{
  std::vector<const Keyword*> top_keywords;
};

/** Data representing one loaded imageset. */
struct ImagesetInfo
{
  std::string ID;
  std::string directory;
  size_t num_sel_frames;
};

/** Data representing all loaded imagesets. */
struct LoadedImagesetsInfo
{
  std::vector<ImagesetInfo> imagesets_info;
};

/** Data representing one loaded data pack. */
struct DataPackInfo
{
  std::string ID;
  std::string description;
  std::string model_options;
  std::string target_imageset_ID;
  size_t num_frames;

  std::string vocabulary_ID;
  std::string vocabulary_description;
};

/** Data representing all loaded data packs. */
struct LoadedDataPacksInfo
{
  std::vector<DataPackInfo> data_packs_info;
};

/**
 * Operations to be used to aggregate main and secondary scorings.
 *
 * \see image_ranker::MultSumMaxModel
 */
enum class eMainTempRankingAggregation
{
  cSum,
  cProduct
};

/**
 * Operations to be used as operation while walking main frame successors in Mult/Sum/Max model.
 *
 * \see image_ranker::MultSumMaxModel
 */
enum class eSuccesorAggregation
{
  cSum,
  cProduct,
  cMax
};

/**
 * Output data for drawing chart.
 */
using ModelTestResult = std::vector<std::pair<uint32_t, uint32_t>>;

/** Supported levels of users */
enum class eUserQueryOrigin
{
  /** Wild public users. */
  PUBLIC = 5,
  /** Non-experts but are close to the field. */
  SEMI_EXPERTS = 10,
  /** Experts - ones working with these tools and data on daily basis. */
  EXPERTS = 15
};

/**
 *  Represents one search action done in interactive search tool in the KISSEval system.
 */
struct InteractiveSearchAction
{
  size_t query_idx = ERR_VAL<size_t>();
  std::string action;
  std::string operand;
  std::string operand_readable;
  size_t final_rank = ERR_VAL<size_t>();
  size_t time = ERR_VAL<size_t>();
  bool is_initial = true;
};

/** Structure holding data after VIRET classnames file parsing.
 *
 * \see image_ranker::FileParser::parse_VIRET_format_keyword_classes_file
 */
struct ViretKeywordClassesParsedData
{
  std::string all_descriptions;
  std::map<size_t, Keyword*> vec_idx_to_keyword;
  std::map<size_t, Keyword*> wordnet_ID_to_keyword;
  std::vector<std::pair<size_t, Keyword*>> desc_index_to_keyword;
  std::vector<std::unique_ptr<Keyword>> keywords;
  std::map<KeywordId, Keyword*> ID_to_keyword;
  std::map<KeywordId, std::set<Keyword*>> ID_to_all_keywords;
};

}  // namespace image_ranker

constexpr std::size_t operator""_z(unsigned long long n) { return n; }

#endif  // _IR_COMMON_H_
