

#include <iostream>
#include <thread>
#include <chrono>
#include "ImageRanker.h"

int main()
{
  ImageRanker ranker{
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\images\)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\Studenti_NasNetLarge.pre-softmax)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\keyword_classes.txt)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\Studenti_NasNetLarge.softmax)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\Studenti_NasNetLarge.deep-features)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\files.txt)",
    1ULL,
    ImageRanker::Mode::cFull
  };

  ranker.Initialize();

  auto result11{ 
    ranker.RunModelTestWrapper(
      AggregationId::cXToTheP, RankingModelId::cBooleanBucket, QueryOriginId::cDeveloper,
      std::vector<std::string>({ "0"s, "0.001"s, "1"s }), std::vector<std::string>({ "1"s })
    )
  };

  for (auto&& slice : result11)
  {
    std::cout << slice.first << " => " << slice.second << std::endl;
  }

  auto result12{
    ranker.RunModelTestWrapper(
      AggregationId::cXToTheP, RankingModelId::cBooleanBucket, QueryOriginId::cDeveloper,
      std::vector<std::string>({ "1"s, "00.0009"s, "1"s }), std::vector<std::string>({ "1"s })
    )
  };

  for (auto&& slice : result12)
  {
    std::cout << slice.first << " => " << slice.second << std::endl;
  }





  return 0;

}
