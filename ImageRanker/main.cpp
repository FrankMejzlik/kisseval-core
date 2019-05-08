

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

  auto result11{ 
    ranker.RunModelTestWrapper(
      AggregationId::cXToTheP, RankingModelId::cViretBase, QueryOriginId::cDeveloper,
      std::vector<std::string>({ "0.0"s, "0"s }), std::vector<std::string>({ "1"s })
    )
  };

  for (auto&& slice : result11)
  {
    std::cout << slice.first << " => " << slice.second << std::endl;
  }


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

/*

  auto results = ranker.GetRelevantImagesWrapper(
    "07906111&00023100&07893528"s, 100, 
    200, 3,
    std::vector<std::string>({ "0.0"s, "1"s }), std::vector<std::string>({ "0.2"s }),
    0
  );

  auto results2 = ranker.GetRelevantImagesWrapper(
    "07906111&00023100&07893528"s, 100,
    200, 3,
    std::vector<std::string>({ "0.0"s, "1"s }), std::vector<std::string>({ "1"s }),
    0
  );

  auto results3 = ranker.GetRelevantImagesWrapper(
    "07906111&00023100&07893528"s, 100,
    200, 3,
    std::vector<std::string>({ "0.0"s, "1"s }), std::vector<std::string>({ "2"s }),
    0
  );*/




  return 0;

}
