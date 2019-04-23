#pragma once

#include <stdint.h>

#define CALC_MIN_MAX_CLAMP_AGG 1
#define CALC_BOOL_AGG 1
#define BOOL_AGG_TRESHOLD 0.5

#define MIN_MAX_CLAMP_TRESHOLD 0.5

#define DEFAULT_RANKING_MODEL ImageRanker::cBooleanCustom
#define DEFAULT_AGG_FUNCTION ImageRanker::AggregationFunction::cSoftmax

#define DEFAULT_MODEL_SETTINGS std::vector<std::string>()


#define CHART_DENSITY 100

/*!
  * Boolean model settings
  */
#define GENERATE_BOOL_IMAGE_VECTOR_NAIVELY 1 
#define IS_TRUE_TRESHOLD 0.005f


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
#define USE_SECONDARY_DB 0
#define DATA_SOURCE_DB Database::cPrimary

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
#define SECONDARY_DB_ID 2


/*!
  * PRIMARY DB
  */
//! 0: localhost testing
#if PRIMARY_DB_ID == 0

  #define PRIMARY_DB_HOST "localhost"
  #define PRIMARY_DB_PORT 3306ULL
  #define PRIMARY_DB_USERNAME "image-ranker"
  #define PRIMARY_DB_PASSWORD "s5XurJ3uS3E52Gzm"
  #define PRIMARY_DB_DB_NAME "image-ranker-collector-test"

//! 1: localhost data1
#elif PRIMARY_DB_ID == 1

  #define PRIMARY_DB_HOST "localhost"
  #define PRIMARY_DB_PORT 3306ULL
  #define PRIMARY_DB_USERNAME "image-ranker"
  #define PRIMARY_DB_PASSWORD "s5XurJ3uS3E52Gzm"
  #define PRIMARY_DB_DB_NAME "image-ranker-collector-data"    

//! 2: localhost data2
#elif PRIMARY_DB_ID == 2

  #define PRIMARY_DB_HOST "localhost"
  #define PRIMARY_DB_PORT 3306ULL
  #define PRIMARY_DB_USERNAME "image-ranker"
  #define PRIMARY_DB_PASSWORD "s5XurJ3uS3E52Gzm"
  #define PRIMARY_DB_DB_NAME "image-ranker-collector-data2"

//! 3: linode data2
#elif PRIMARY_DB_ID == 3
  
  #define PRIMARY_DB_HOST "li1504-72.members.linode.com"
  #define PRIMARY_DB_PORT 3306ULL
  #define PRIMARY_DB_USERNAME "image-ranker"
  #define PRIMARY_DB_PASSWORD "s5XurJ3uS3E52Gzm"
  #define PRIMARY_DB_DB_NAME "image-ranker-collector-data"

//! 4: linode data2
#elif PRIMARY_DB_ID == 4

  #define PRIMARY_DB_HOST "li1504-72.members.linode.com"
  #define PRIMARY_DB_PORT 3306ULL
  #define PRIMARY_DB_USERNAME "image-ranker"
  #define PRIMARY_DB_PASSWORD "s5XurJ3uS3E52Gzm"
  #define PRIMARY_DB_DB_NAME "image-ranker-collector-data2"

#endif


/*!
  * SECONDARY DB
  */
//! 0: localhost testing
#if SECONDARY_DB_ID == 0

  #define SECONDARY_DB_HOST "localhost"
  #define SECONDARY_DB_PORT 3306ULL
  #define SECONDARY_DB_USERNAME "image-ranker"
  #define SECONDARY_DB_PASSWORD "s5XurJ3uS3E52Gzm"
  #define SECONDARY_DB_DB_NAME "image-ranker-collector-test"

  //! 1: localhost data1
#elif SECONDARY_DB_ID == 1

  #define SECONDARY_DB_HOST "localhost"
  #define SECONDARY_DB_PORT 3306ULL
  #define SECONDARY_DB_USERNAME "image-ranker"
  #define SECONDARY_DB_PASSWORD "s5XurJ3uS3E52Gzm"
  #define SECONDARY_DB_DB_NAME "image-ranker-collector-data"    

  //! 2: localhost data2
#elif SECONDARY_DB_ID == 2

  #define SECONDARY_DB_HOST "localhost"
  #define SECONDARY_DB_PORT 3306ULL
  #define SECONDARY_DB_USERNAME "image-ranker"
  #define SECONDARY_DB_PASSWORD "s5XurJ3uS3E52Gzm"
  #define SECONDARY_DB_DB_NAME "image-ranker-collector-data2"

  //! 3: linode data2
#elif SECONDARY_DB_ID == 3

  #define SECONDARY_DB_HOST "li1504-72.members.linode.com"
  #define SECONDARY_DB_PORT 3306ULL
  #define SECONDARY_DB_USERNAME "image-ranker"
  #define SECONDARY_DB_PASSWORD "s5XurJ3uS3E52Gzm"
  #define SECONDARY_DB_DB_NAME "image-ranker-collector-data"

  //! 4: linode data2
#elif SECONDARY_DB_ID == 4

  #define SECONDARY_DB_HOST "li1504-72.members.linode.com"
  #define SECONDARY_DB_PORT 3306ULL
  #define SECONDARY_DB_USERNAME "image-ranker"
  #define SECONDARY_DB_PASSWORD "s5XurJ3uS3E52Gzm"
  #define SECONDARY_DB_DB_NAME "image-ranker-collector-data2"

#endif


/*!
 * Files with data
 */
#define USING_DATASET 2

#if USING_DATASET == 1
  /***************************
    Dataset 1
  *************************/
  //! Number of images in file data
  #define NUM_ROWS 20000ULL
  //! ID offset from file data
  #define INDEX_OFFSET 50ULL

  #define IMAGES_LIST_FILENAME "dir_images.txt"
  #define COLUMN_INDEX_OF_FILENAME 4
  #define FILES_FILE_LINE_LENGTH 90ULL
  #define KEYWORD_CLASSES_FILENAME "keyword_classes.txt"
  #define DEEP_FEATURES_FILENAME "VBS2019_classification_NasNetMobile_20000.deep-features"
  #define SOFTMAX_BIN_FILE "VBS2019_classification_NasNetLarge_20000.softmax"

  #if DEVELOPMENT

    #define SOLUTION_PATH "c:\\Users\\devwe\\source\\repos\\ImageRanker\\"
    #define DATA_PATH SOLUTION_PATH "data\\"
    #define IMAGES_PATH SOLUTION_PATH "images\\"
    #define COLLECTOR_INPUT_OUTPUT_DATA_PATH SOLUTION_PATH "data\\to_collector\\"

  #elif STAGING

    #define SOLUTION_PATH "/home/devwesp/source/repos/ImageRankerCollector/"
    #define DATA_PATH "/home/devwesp/source/repos/ImageRankerCollector/data/"
    #define IMAGES_PATH "/home/devwesp/source/repos/ImageRankerCollector/public/images/"

  #endif

#elif USING_DATASET == 2

  /*************************
    Dataset 2
  *************************/
  //! Number of images in file data
  #define NUM_ROWS 20000ULL
  //! ID offset from file data
  #define INDEX_OFFSET 1ULL

  //#define IMAGES_LIST_FILENAME "files.txt"
  #define IMAGES_LIST_FILENAME "dir_images.txt"  
  #define COLUMN_INDEX_OF_FILENAME 4
  #define FILES_FILE_LINE_LENGTH 90ULL
  #define KEYWORD_CLASSES_FILENAME "keyword_classes.txt"
  #define DEEP_FEATURES_FILENAME "Studenti_NasNetLarge.deep-features"
  //#define SOFTMAX_BIN_FILE "Studenti_NasNetLarge.softmax"
  #define SOFTMAX_BIN_FILE "Studenti_NasNetLarge.pre-softmax"



  #if DEVELOPMENT

    #define SOLUTION_PATH "c:\\Users\\devwe\\source\\repos\\ImageRanker\\"
    #define DATA_PATH SOLUTION_PATH "data2\\"
    #define IMAGES_PATH SOLUTION_PATH "images\\"
    #define COLLECTOR_INPUT_OUTPUT_DATA_PATH SOLUTION_PATH "data\\to_collector\\"

  #elif STAGING

    #define SOLUTION_PATH "/home/devwesp/source/repos/ImageRankerCollector/"
    #define DATA_PATH "/home/devwesp/source/repos/ImageRankerCollector/data2/"
    #define IMAGES_PATH "/home/devwesp/source/repos/ImageRankerCollector/public/images/"

  #endif

#endif

#define GENERATE_COLLECTOR_INPUT_DATA 1


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
