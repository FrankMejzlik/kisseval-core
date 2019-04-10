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
    R"(C:\Users\devwe\source\repos\ImageRankerCollector\data2\Studenti_NasNetLarge.softmax)",
    R"(C:\Users\devwe\source\repos\ImageRankerCollector\data2\Studenti_NasNetLarge.deep-features)",
    R"(C:\Users\devwe\source\repos\ImageRankerCollector\data2\keyword_classes.txt)",
    R"(C:\Users\devwe\source\repos\ImageRankerCollector\data2\dir_images.txt)",
    4,
    90,
    20000,
    1
  };

  auto result3 = ranker.GetRelevantImages("eagle", 500, (ImageRanker::RankingModel)2, 0);

  
  auto result{ ranker.RunModelTest((ImageRanker::RankingModel)2, (ImageRanker::QueryOrigin)0, std::string("0.005000")) };


  for (auto&& slice : result) 
  {
    std::cout << slice.first << " => " << slice.second << std::endl;
  }

#endif

  return 0;

}
