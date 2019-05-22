#pragma once

#include <stdint.h>

//! In seconds
#define QUERIES_CACHE_LIFETIME 5

#define TRUE_TRESHOLD_FOR_KW_FREQUENCY 0.001f
/*!
  100 => always softmax
  200 => always x^1
*/
#define KW_FREQUENCY_BASE_DATA 200


//! Default settings for main evaluation
#define DEFAULT_RANKING_MODEL RankingModelId::cViretBase
#define DEFAULT_AGG_FUNCTION AggregationId::cXToTheP
#define DEFAULT_MODEL_SETTINGS std::vector<std::string>({"0"s, "0.0f"s, "1"s})
#define DEFAULT_TRANSFORM_SETTINGS std::vector<std::string>({"0"s, "0"s})


#define CHART_DENSITY 100
#define CHART_NUM_X_POINTS 100

/*!
  * Boolean model settings
  */
#define NUM_IMAGES_PER_PAGE 200

#define DEVELOPMENT 1
#define STAGING 0
#define PRODUCTION 0

#define DEFAULT_MODE ImageRanker::Mode::cCollector

//! Will throw exception on LOG_ERROR
#define THROW_ON_ERROR 1

#define MIN_DESC_SEARCH_LENGTH 3

//! What is delimiter for synonyms in data files
#define SYNONYM_DELIMITER '#'

//! What is delimiter in CSV data files
#define CSV_DELIMITER '~'

//! How many suggestions will be returned when called \ref ImageRanker::GetNearKeywords
#define NUM_SUGESTIONS 5ULL

//! If set to 1, loaded data from files will be inserted into PRIMARY db
#define PUSH_DATA_TO_DB 0

#define LOG_DEBUG_HYPERNYMS_EXPANSION 0
#define LOG_DEBUG_IMAGE_RANKING 0

//! Standard logging macro
#define LOG(x) std::cout << x << std::endl
#define LOG_NO_ENDL(x) std::cout << x

//! Basic log error macro
#if THROW_ON_ERROR

  #define LOG_ERROR(x) std::cout << "ERROR: " << x << std::endl; throw std::runtime_error(std::string(x));

#elif

  #define LOG_ERROR(x) std::cout << "ERROR: " << x << std::endl;

#endif

 //! Error value for size_t types
#define SIZE_T_ERROR_VALUE   SIZE_MAX


#define LOG_CALLS 1

/*!
 * Set what databases will be used as primary/secondary
 *
 * OPTIONS:
 * 0: localhost testing
 * 1: localhost data1
 * 2: localhost data2
 * 3: linode data1
 * 4: linode data2
 */
#define PRIMARY_DB_ID 4
#include "credentials.h"
