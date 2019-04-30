

#include <iostream>

#include "ImageRanker.h"

int main()
{

#if GET_DATA_FROM_DB

  ImageRanker ranker{IMAGES_PATH};

#else 

  ImageRanker ranker{
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\images\)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\Studenti_NasNetLarge.pre-softmax)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\keyword_classes.txt)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\Studenti_NasNetLarge.softmax)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\Studenti_NasNetLarge.deep-features)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\files.txt)",
  };

  ranker.Initialize();

  //auto result1{ ranker.RunModelTest(
  // (ImageRanker::Aggregation)5,
  // (ImageRanker::RankingModel)3,
  // (ImageRanker::QueryOrigin)0,
  // std::vector<std::string>({"0.0", "0.000", "0", "2"})) };

  //for (auto&& slice : result1)
  //{
  //  std::cout << slice.first << " => " << slice.second << std::endl;
  //}

  auto p1 = ranker.GetRandomImage();
  auto p2 = ranker.GetRandomImage();

  std::vector<ImageRanker::GameSessionInputQuery> tv;
  tv.emplace_back(std::tuple("ses1"s, 1234ULL, "111&222&333&4444"));
  tv.emplace_back(std::tuple("ses2"s, 12364ULL, "9111&9222&3933&94444"));

  auto result = ranker.SubmitUserQueriesWithResults(tv, ImageRanker::cPublic);

#endif


  return 0;

}
