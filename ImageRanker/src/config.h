#pragma once

#include <stdint.h>

#define COLLECTOR_MODE 1

#define DETECT_AMBIG_KEYWORDS 1
#define PRINT_QUERIES 1 

#define CALC_MIN_MAX_CLAMP_AGG 1
#define CALC_BOOL_AGG 1
#define BOOL_AGG_TRESHOLD 0.5

#define MIN_MAX_CLAMP_TRESHOLD 0.5

//! Default settings for main evaluation
#define DEFAULT_RANKING_MODEL ImageRanker::cViretBase
#define DEFAULT_AGG_FUNCTION ImageRanker::Aggregation::cSoftmax
#define DEFAULT_MODEL_SETTINGS std::vector<std::string>({"0.0f"s, "1"s})


#define CHART_DENSITY 100

/*!
  * Boolean model settings
  */
#define NUM_IMAGES_PER_PAGE 200

#define DEVELOPMENT 1
#define STAGING 0
#define PRODUCTION 0

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

//! If 1, data will be loaded from database(s)
#define GET_DATA_FROM_DB 0
#define DATA_SOURCE_DB Database::cPrimary


//! Standard logging macro
#define LOG(x) std::cout << x << std::endl;

//! Basic log error macro
#if THROW_ON_ERROR

  #define LOG_ERROR(x) std::cout << "ERROR: " << x << std::endl; throw std::runtime_error(std::string(x));

#elif

  #define LOG_ERROR(x) std::cout << "ERROR: " << x << std::endl;

#endif

 //! Error value for size_t types
#define SIZE_T_ERROR_VALUE   SIZE_MAX


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
#define PRIMARY_DB_ID 2
#include "credentials.h"
