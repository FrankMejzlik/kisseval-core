// ImageRanker.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "ImageRanker.h"

int main()
{

#if GET_DATA_FROM_DB

  ImageRanker ranker{IMAGES_PATH};

#else 

  //ImageRanker ranker{
  //  IMAGES_PATH,
  //  DATA_PATH SOFTMAX_BIN_FILE,
  //  DATA_PATH DEEP_FEATURES_FILENAME,
  //  DATA_PATH KEYWORD_CLASSES_FILENAME,
  //  DATA_PATH IMAGES_LIST_FILENAME,
  //  COLUMN_INDEX_OF_FILENAME,
  //  FILES_FILE_LINE_LENGTH,
  //  NUM_ROWS,
  //  INDEX_OFFSET
  //};


  ImageRanker ranker{
    R"(C:\Users\devwe\source\repos\ImageRankerCollector\public\images\)",
    R"(C:\Users\devwe\source\repos\ImageRankerCollector\data2\Studenti_NasNetLarge.pre-softmax)",
    R"(C:\Users\devwe\source\repos\ImageRankerCollector\data2\Studenti_NasNetLarge.deep-features)",
    R"(C:\Users\devwe\source\repos\ImageRankerCollector\data2\keyword_classes.txt)",
    R"(C:\Users\devwe\source\repos\ImageRankerCollector\data2\dir_images.txt)",
    4,
    90,
    20000,
    1
  };
  

  std::vector<ImageRanker::GameSessionInputQuery> input;
  input.push_back(ImageRanker::GameSessionInputQuery("123ULL"s, 0ULL, "dog&cat&wood"));
  input.push_back(ImageRanker::GameSessionInputQuery("123U3LL"s, 50ULL, "brick&entity"));

  auto output = ranker.SubmitUserQueriesWithResults(input);


  auto result = ranker.GetNearKeywords("paprika");
  

  for (auto&& r : result)
  {
    std::cout << std::get<1>(r) <<std::endl;
  }

#endif

  return -0;

}
