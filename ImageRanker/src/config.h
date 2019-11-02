#pragma once

#include <stdint.h>

#define LOG_PRE_AND_PPOST_EXP_RANKS 0

#define PRECOMPUTE_EXPANSION_SUBWORDS 0
#define LOG_DEBUG_PRECOMPUTE_SUBSTRINGS_1 0
#define LOG_DEBUG_PRECOMPUTE_SUBSTRINGS_2 0

#define PARSE_W2V_FILE 0
// 1-> all substrings, 2 -> substring matching whole word, 3 -> w2v, 23 -> 2&3
#define SUBSTRING_EXPANSION_SET 23

#define DO_QUERY_DYNAMIC_EXPANSION 0
#define CONCATS_COEF 1.0f
#define SUBSTRINGS_COEF 1.0f

#define DO_SUBSTRING_EXPANSION 0

// 0 AND, 1 OR
#define SUBSTRING_EXPANSION_TYPE 1
#define LOG_QUERY_EXPANSION 0

#define PARSE_W2V_FILES 0
#define LOG_W2V_EXPANSION_KW_SETS 0
#define W2V_DISTANCE_THRESHOLD 0.5f




#define SOLUTION_DIR R"(c:\Users\devwe\source\repos\ImageRanker\)"s

#define NUM_TOP_KEYWORDS 100_z
//
// Google data setings
//
#define GOOGLE_AI_NO_LABEL_SCORE 0.001f
#define VIRET_TRESHOLD_LINEAR_01 0.001f

#define NO_TRANSFORM_ID 900_z

//
// Keyword types
//
#define DEFAULT_KEYWORD_DATA_TYPE eKeywordsDataType::cViret1
#define DEFAULT_SCORING_DATA_TYPE eImageScoringDataType::cNasNet


//
// Query tests settings
//
#define TEST_QUERIES_ID_MULTIPLIER 1


//
// TRECVID SPECIFIC
//
#define PRECOMPUTE_MAX_BASED_DATA 1
#define TRECVID_MAPPING 1
#define SHOT_REFERENCE_PATH R"(c:\Users\devwe\source\repos\ImageRankerApp\data\trecvid_data\shot_reference\)"
#define DROPPED_SHOTS_FILEPATH R"(c:\Users\devwe\source\repos\ImageRankerApp\data\trecvid_data\dropped_shots.txt)"
#define DEBUG_SHOW_OUR_FRAME_IDS 1


//
// Frame filename format
//
#define MAX_SUCC_CHECK_COUNT 10

#define FILENAME_START_INDEX 6 // Trecvid one
//#define FILENAME_START_INDEX 0

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
#define RUN_TESTS_ONLY_ON_NON_EMPTY_POSTREMOVE_HYPERNYM 0

//
// Simulated user & temp queries settings
//
#define SIMULATED_QUERIES_ENUM_OFSET 10000
#define USER_PLUS_SIMULATED_QUERIES_ENUM_OFSET 20000
#define MAX_TEMP_QUERY_OFFSET 5


//! In seconds
#define QUERIES_CACHE_LIFETIME 0

#define TRUE_TRESHOLD_FOR_KW_FREQUENCY 0.001f
/*!
  100 => always softmax
  200 => always x^1
*/
#define KW_FREQUENCY_BASE_DATA 200


//! Default settings for main evaluation
#define DEFAULT_RANKING_MODEL RankingModelId::cViretBase
#define DEFAULT_AGG_FUNCTION InputDataTransformId::cXToTheP
#define DEFAULT_MODEL_SETTINGS std::vector<std::string>({"0"s, "0.0f"s, "1"s})
#define DEFAULT_TRANSFORM_SETTINGS std::vector<std::string>({"0"s, "0"s})


#define MODEL_TEST_CHART_NUM_X_POINTS 100

/*!
  * Boolean model settings
  */
#define NUM_IMAGES_PER_PAGE 200

#define DEVELOPMENT 1
#define STAGING 0
#define PRODUCTION 0

#define DEFAULT_MODE ImageRanker::eMode::cFullAnalytical

//! Will throw exception on LOG_ERROR
#define THROW_ON_ERROR 1

#define MIN_DESC_SEARCH_LENGTH 3

//! How many suggestions will be returned when called \ref ImageRanker::GetNearKeywords
#define NUM_SUGESTIONS 5ULL


#define LOG_DEBUG_HYPERNYMS_EXPANSION 0
#define LOG_DEBUG_IMAGE_RANKING 0
#define LOG_DEBUG_RUN_TESTS 0

//! Standard logging macro
#define LOG(x) std::cout << x << std::endl

#define LOG_WARN(x) std::cout << x << std::endl

#define LOG_NO_ENDL(x) std::cout << x

//! Basic log error macro
#if THROW_ON_ERROR

  #define LOG_ERROR(x) \
    std::cout << "ERROR: " << x << "(" << __LINE__ << ", " << __FILE__ << ")" << std::endl; \
    throw std::runtime_error(std::string(x));

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
