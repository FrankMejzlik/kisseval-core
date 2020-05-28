#pragma once

#include <stdint.h>

/** If running in production.
 *
 * Errors will throw production exception instead of debugging ones.
 */
#define IS_PRODUCTION (0)

/** If basic functionality tests will be executed after initialization.
 *    \see image_ranker::Tester */
#define RUN_BASIC_TESTS (1)

/**********************************************
 * Input data
 ***********************************************/

/** Prefix that will be used when looking for data files. */
#define DATA_DIR "./data/"

/** File that defines where each data file is located. */
#define DATA_INFO_FPTH "../data_info.json"

/** Filepath to the SQLite database file. */
#define DB_FPTH "database.db"

/** Delimiter for columns in  VIRET format keyword classes file */
#define CSV_DELIMITER_001 '~'

/** Delimiter for synonyms in VIRET format keyword classes file */
#define SYNONYM_DELIMITER_001 '#'

#define VIRET_FORMAT_NET_DATA_HEADER_SIZE 36

/**********************************************
 * Logging
 ***********************************************/

/** If exception will be thrown on LOGE. */
#if IS_PRODUCTION
#  define THROW_ON_ERROR 0
#else
#  define THROW_ON_ERROR 1
#endif

/** Logging level.
 *
 * Levels:
 *  5 - Verbose
 *  4 - Debug
 *  3 - Info
 *  2 - Warning
 *  1 - Error
 *  0 - None
 */
#define LOG_LEVEL 5

/**********************************************
 * API methods' config
 ***********************************************/

/*
 * Model testing
 */
#define NUM_MODEL_TEST_RESULT_POINTS 100

/*
 * Simulted users
 */
/** \ref image_ranker::top_frame_keywords will return this ammount by default */
#define DEF_SIM_USER_NUM_WORDS_FROM 2
#define DEF_SIM_USER_NUM_WORDS_TO 6
#define DEF_SIM_USER_TARGET eSimUserTarget::SINGLE_QUERIES
#define DEF_SIM_USER_EXPONENT 6.0F

/*
 * Autocomplate
 */
#define DEF_NUM_AUTOCOMPLETE_RESULTS 10

/** Minimal length of prefix to return any aytocomplete suggestions. */
#define MIN_DESC_SEARCH_LENGTH 2

/*
 * Ranker
 */
/** top_frame_keywords will return this ammount by default. */
#define DEF_NUMBER_OF_TOP_KWS 10

/*
 * Statistics
 */
/** Maximum user leve that will be used when generating data for statistics. */
#define DEF_MAX_USER_LEVEL_FOR_DATA 9

/** Minimum number of samples for search session charts. */
#define DEF_MIN_SAMPLES_SS_CHART 20

/**********************************************
 * Data packs & Models
 ***********************************************/

/** Value that zero values will be substitued with */
#define ZERO_WEIGHT 0.01F

#define TEMP_CONTEXT_LOOKUP_LENGTH 3
#define NUM_EXAMPLE_FRAMES 10

/** Number of top classes that will be computed and stored */
#define NUM_TOP_KWS_LOADED 10
