

#include <iostream>

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

  /*auto result1{ ranker.RunModelTest(
   (ImageRanker::Aggregation)1,
   (ImageRanker::RankingModel)3,
   (ImageRanker::QueryOrigin)0,
   std::vector<std::string>({"0.0", "0.000", "0", "2"})) };

  for (auto&& slice : result1)
  {
    std::cout << slice.first << " => " << slice.second << std::endl;
  }*/


  /*auto result = ranker.RunGridTest(std::vector<ImageRanker::TestSettings>());

  size_t i{ 0ULL };
  for (auto&& setChartDataPair : result)
  {
    const auto& settings{ setChartDataPair.first };
    const auto& chartData{ setChartDataPair.second };

    std::cout << i << ")" << std::endl;
    std::cout << "Aggregation = " << std::get<0>(settings) << ", RankingModel = " << std::get<1>(settings) << ", QueryOrigin" << std::get<2>(settings) << std::endl;

    std::cout << "MODEL SETTINGS: " << std::endl;
    auto setStr{ std::get<3>(settings) };
    for (auto&& str : setStr)
    {
      std::cout << str << std::endl;
    }
    std::cout << "---" << std::endl;

    for (auto&& slice : chartData)
    {
      std::cout << slice.first << " => " << slice.second << std::endl;
    }

    std::cout << "===========================" << std::endl;
    ++i;
  }*/


  auto results = ranker.GetRelevantImagesPlainQuery(
    "07906111&00023100&07893528"s, 100, AggregationId::cSoftmax, 
    RankingModelId::cBooleanBucket, std::vector<std::string>({ "0.869999"s, "1"s }), std::vector<std::string>()
  );




  return 0;

}
