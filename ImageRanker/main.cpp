// ImageRanker.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "ImageRanker.h"

int main()
{

#if GET_DATA_FROM_DB

  ImageRanker ranker{IMAGES_PATH};

#else 

  ImageRanker ranker{
    IMAGES_PATH,
    DATA_PATH IMAGE_PROB_VECTOR_BIN_FILENAME,
    DATA_PATH DEEP_FEATURES_FILENAME,
    DATA_PATH KEYWORD_CLASSES_FILENAME
  };

  std::vector<ImageRanker::GameSessionInputQuery> input;
  input.push_back(ImageRanker::GameSessionInputQuery(123ULL, 0ULL, "dog&cat&wood"));
  input.push_back(ImageRanker::GameSessionInputQuery(124ULL, 50ULL, "brick&entity"));

  auto output = ranker.SubmitUserQueriesWithResults(input);

#endif

  return -0;

}
