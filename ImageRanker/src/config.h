#pragma once

#define REPEAT_MAIN 1

#define USE_REMOTE_DB 0

#define USE_DATA_FROM_DATABASE 0

#define NUM_ROWS 20000ULL

#define INDEX_OFFSET 50ULL

#define SOLUTION_PATH "c:\\Users\\devwe\\source\\repos\\ImageRanker\\"

#define DATA_PATH "c:\\Users\\devwe\\source\\repos\\ImageRanker\\data\\"
#define IMAGES_PATH "c:\\Users\\devwe\\source\\repos\\ImageRanker\\data\\images\\"
#define COLLECTOR_INPUT_OUTPUT_DATA_PATH "c:\\Users\\devwe\\source\\repos\\ImageRanker\\data\\to_collector\\"

#define GENERATE_COLLECTOR_INPUT_DATA 1

#define IMAGES_LIST_FILENAME "dir_images.txt"
#define KEYWORD_CLASSES_FILENAME "keyword_classes.txt"
#define DEEP_FEATURES_FILENAME "VBS2019_classification_NasNetMobile_20000.deep-features"
#define SOFTMAX_BIN_FILENAME "VBS2019_classification_NasNetLarge_20000.softmax"

#define SYNONYM_DELIMITER '#'
#define CSV_DELIMITER '~'

#define NUM_SUGESTIONS 5ULL

#define LOG(x) std::cout << x << std::endl

