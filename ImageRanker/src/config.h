#pragma once

#include <stdint.h>

//
// TRECVID SPECIFIC
//
#define TRECVID_MAPPING 1
#define SHOT_REFERENCE_PATH R"(c:\Users\devwe\source\repos\ImageRankerApp\data\trecvid_data\shot_reference\)"


//
// Frame filename format
//
//#define FILENAME_START_INDEX 6 // Trecvid one
#define FILENAME_START_INDEX 0

#define FILENAME_VIDEO_ID_FROM 1
#define FILENAME_VIDEO_ID_LEN 5

#define FILENAME_SHOT_ID_FROM 8
#define FILENAME_SHOT_ID_LEN 5

#define FILENAME_FRAME_NUMBER_FROM 40
#define FILENAME_FRAME_NUMBER_LEN 6

//! If true, successors from whole videos will be considered (e.g. in temporal queries)
#define USE_VIDEOS_AS_SHOTS 1

//! If constructed hypernyms should be ignored
#define IGNORE_CONSTRUCTED_HYPERNYMS false
#define RUN_TESTS_ONLY_ON_NON_EMPTY_POSTREMOVE_HYPERNYM 1

//
// Simulated user & temp queries settings
//
#define SIMULATED_QUERIES_ENUM_OFSET 10000
#define USER_PLUS_SIMULATED_QUERIES_ENUM_OFSET 20000
#define MAX_TEMP_QUERY_OFFSET 5


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
#define DEFAULT_AGG_FUNCTION NetDataTransformation::cXToTheP
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
 * 0: localhost "ir_base_test"
 * 1: localhost "image-ranker-collector-data2"
 * 2: localhost data2
 * 3: Herkules --
 * 4: Herkules "image-ranker-collector-data2"
 */
#define PRIMARY_DB_ID 0
#include "credentials.h"
