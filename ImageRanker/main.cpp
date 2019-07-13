

#include <iostream>
#include <thread>
#include <chrono>
#include "ImageRanker.h"

int main()
{
  /*ImageRanker ranker{
    R"(c:\Users\devwe\source\repos\ImageRanker\data\dataset2\images\)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\Studenti_NasNetLarge.pre-softmax)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\keyword_classes.txt)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\Studenti_NasNetLarge.softmax)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\Studenti_NasNetLarge.deep-features)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\files.txt)",
    1ULL,
    ImageRanker::Mode::cSearchTool
  };*/

  ImageRanker ranker{
    R"(c:\Users\devwe\source\repos\ImageRankerApp\public\images\)",
    R"(c:\Users\devwe\source\repos\ImageRankerApp\data\trecvid_data\net_data\nasnet_large.pre-softmax)",
    R"(c:\Users\devwe\source\repos\ImageRankerApp\data\trecvid_data\net_data\keyword_classes.txt)",
    R"()",
    R"()",
    R"(c:\Users\devwe\source\repos\ImageRankerApp\data\trecvid_data\keyframes\filelist.txt)",
    1ULL,
    ImageRanker::Mode::cSearchTool
  };

  ranker.Initialize();

  /*auto result1{
  ranker.GetRelevantImagesWithSuggestedWrapper(
    std::vector<std::string>({"3438257"}), 100,
    NetDataTransformation::cXToTheP, RankingModelId::cViretBase,
    std::vector<std::string>({ "0"s, "0.0"s, "2"s, "1"s, "2"s }), std::vector<std::string>({ "1"s, "0"s }),
    5_z
  )};*/


  auto result2{
  ranker.TrecvidGetRelevantShots(
    std::vector<std::string>({"3438257", "2753044"}), 1000_z,
    NetDataTransformation::cXToTheP, RankingModelId::cViretBase,
    std::vector<std::string>({ "0"s, "0.0"s, "0"s, "1"s, "2"s }), std::vector<std::string>({ "1"s, "0"s }),
    0.0f
  ) };

  


  /*auto result15{
  ranker.RunModelTestWrapper(
    NetDataTransformation::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)20000,
    std::vector<std::string>({ "4"s }), std::vector<std::string>({ "0"s, "0.0"s, "2"s, "1"s, "2"s }), std::vector<std::string>({ "1"s, "0"s })
  )
  };

  for (auto&& i : result15)
  {
    std::cout << i.first << " => " << i.second << std::endl;
  }*/

  std::cout << "========================" << std::endl;
  
  
/*
  auto result11{
  ranker.RunModelTestWrapper(
    AggregationId::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)999,
    std::vector<std::string>({ "0"s, "0.0"s, "0"s }), std::vector<std::string>({ "0"s })
  )
  };

  auto result12{
  ranker.RunModelTestWrapper(
    AggregationId::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)999,
    std::vector<std::string>({ "0"s, "0.0"s, "1"s }), std::vector<std::string>({ "0"s })
  )
  };*/

  /*auto result13{
  ranker.RunModelTestWrapper(
    AggregationId::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)999,
    std::vector<std::string>({ "0"s, "0.0"s, "2"s }), std::vector<std::string>({ "1"s, "0"s })
  )
  };

  for (auto&& i : result13)
  {
    std::cout << i.first << " => " << i.second;
  }

  auto result14{
  ranker.RunModelTestWrapper(
    AggregationId::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)999,
    std::vector<std::string>({ "0"s, "0.0"s, "2"s }), std::vector<std::string>({ "1"s, "1"s })
  )
  };

  for (auto&& i : result14)
  {
    std::cout << i.first << " => " << i.second;
  }*/
  

  auto result{ ranker.GetNearKeywordsWithImages("Bla") };

  //auto result2{ ranker.GetImageKeywordsForInteractiveSearchWithExampleImages(1ULL, 10)};

  
  return 0;

}
