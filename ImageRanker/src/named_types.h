#pragma once

#include <array>
#include <string>
#include <vector>

/**********************************************
 * Name definitions
 ***********************************************/

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

inline const std::pair<std::string, std::string>& enum_label(eModelIds val) { return eModelIds_labels[size_t(val)]; }

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
  return eSimUserIds_labels[size_t(val)];
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
  return eTransformationIds_labels[size_t(val)];
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
  return eModelOptsKeys_labels[size_t(val)];
}
